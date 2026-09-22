"""RunPod handler wrapper for permanent render archiving and library browsing."""

from __future__ import annotations

import base64
import copy
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from permanent_storage import (
    ArchiveError,
    archive_generation_result,
    archive_status,
    download_render_video,
    list_renders,
)
from runtime_controls import MINIMAX_RUNTIME_OPTION_NAMES
from worker import handle_job as base_handle_job


LOGGER = logging.getLogger("redgraft-app-worker")
APP_WORKER_VERSION = "redgraft-library-3"

_PRESET_SIGNATURES: dict[str, dict[str, Any]] = {
    "Fast Test": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.40, "mystic_strength": 0.0, "hmnsfw_strength": 0.0, "vagassist_strength": 0.0, "hmpussy_strength": 0.0, "cumshot_strength": 0.0, "steps": 6, "enable_audio": False, "enable_gimm": False},
    "General NSFW": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.50, "mystic_strength": 0.55, "hmnsfw_strength": 0.55, "vagassist_strength": 0.0, "hmpussy_strength": 0.0, "cumshot_strength": 0.0, "steps": 8, "enable_audio": False, "enable_gimm": False},
    "Anatomy Lock": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.45, "mystic_strength": 0.60, "hmnsfw_strength": 0.60, "vagassist_strength": 1.0, "hmpussy_strength": 0.35, "cumshot_strength": 0.0, "steps": 8, "enable_audio": False, "enable_gimm": False},
    "HMNSFW Focus": {"length_seconds": 8, "turbo_strength": 0.50, "m3_strength": 0.35, "mystic_strength": 0.70, "hmnsfw_strength": 1.0, "vagassist_strength": 0.75, "hmpussy_strength": 0.25, "cumshot_strength": 0.0, "steps": 12, "enable_audio": False, "enable_gimm": False},
    "Cumshot I2V": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.40, "mystic_strength": 0.70, "hmnsfw_strength": 0.70, "vagassist_strength": 1.0, "hmpussy_strength": 0.35, "cumshot_strength": 0.70, "steps": 8, "enable_audio": False, "enable_gimm": False},
    "Full Stack": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.50, "mystic_strength": 1.0, "hmnsfw_strength": 1.0, "vagassist_strength": 1.0, "hmpussy_strength": 0.35, "cumshot_strength": 0.70, "steps": 8, "enable_audio": True, "enable_gimm": True},
}


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(right, bool):
        return isinstance(left, bool) and left is right
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return abs(float(left) - float(right)) < 1e-6
        except (TypeError, ValueError):
            return False
    return left == right


def _infer_preset_name(job_input: dict[str, Any]) -> str:
    for name, signature in _PRESET_SIGNATURES.items():
        if all(key in job_input and _same_value(job_input[key], value) for key, value in signature.items()):
            return name
    return "Custom"


def _library_action(job_input: dict[str, Any]) -> dict[str, Any]:
    allowed = {"action", "cursor", "max_keys"}
    if set(job_input) - allowed:
        return {"error": "Library requests support only action, cursor, and max_keys"}
    cursor = job_input.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        return {"error": "input.cursor must be a string"}
    max_keys = job_input.get("max_keys", 300)
    if isinstance(max_keys, bool):
        return {"error": "input.max_keys must be an integer from 20 to 1000"}
    try:
        max_keys = int(max_keys)
    except (TypeError, ValueError):
        return {"error": "input.max_keys must be an integer from 20 to 1000"}
    if max_keys < 20 or max_keys > 1000:
        return {"error": "input.max_keys must be an integer from 20 to 1000"}
    try:
        library = list_renders(cursor=cursor, max_keys=max_keys)
        return {"status": "library", "app_worker_version": APP_WORKER_VERSION, "archive": library}
    except ArchiveError as exc:
        return {"error": str(exc), "archive": {"configured": False}}


def _archive_success(
    *,
    job: dict[str, Any],
    result: dict[str, Any],
    prompt: str,
    preset_name: str,
    request_settings: dict[str, Any],
) -> dict[str, Any]:
    result["app_worker_version"] = APP_WORKER_VERSION
    if result.get("status") != "success":
        return result
    try:
        archived = archive_generation_result(
            job_id=job.get("id"),
            prompt=prompt,
            preset_name=preset_name,
            request_settings=request_settings,
            result=result,
        )
        if archived is None:
            result["archive"] = {
                "configured": False,
                "permanent": False,
                "message": "Permanent object storage is not configured on this RunPod endpoint.",
            }
        else:
            result["archive"] = archived
            # Replace possibly short-lived provider URLs with our own fresh signed
            # private URLs. The underlying bucket objects remain permanent.
            result["videos"] = archived["videos"]
    except ArchiveError as exc:
        LOGGER.exception("Permanent archive metadata failed")
        result["archive"] = {
            "configured": True,
            "permanent": False,
            "error": str(exc),
        }
    return result


def _extension_frame(render_id: str) -> tuple[dict[str, Any], str]:
    """Extract the last decoded frame of an archived render as a JPEG data URI."""
    timeout = max(30, min(300, int(os.getenv("EXTEND_FRAME_TIMEOUT_SECONDS", "120"))))
    with tempfile.TemporaryDirectory(prefix="redgraft-extend-") as folder:
        root = Path(folder)
        source = root / "source.mp4"
        frame = root / "last-frame.jpg"
        metadata, _ = download_render_video(render_id, source)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-sseof",
            "-0.20",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise ArchiveError("Video extension requires ffmpeg in the worker image") from exc
        except subprocess.TimeoutExpired as exc:
            raise ArchiveError("Timed out while extracting the final video frame") from exc
        if completed.returncode != 0 or not frame.is_file() or frame.stat().st_size == 0:
            detail = (completed.stderr or "").strip()[-500:]
            raise ArchiveError("Could not extract the final frame from this render" + (f": {detail}" if detail else ""))
        data = frame.read_bytes()
        if len(data) > 12 * 1024 * 1024:
            raise ArchiveError("The extracted final frame is unexpectedly large")
        return metadata, "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _extend_action(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    """Continue a permanent render by feeding its final frame back into MiniMax I2V."""
    allowed = {"action", "render_id", "prompt", "length_seconds", *MINIMAX_RUNTIME_OPTION_NAMES}
    if set(job_input) - allowed:
        return {"error": "Video extension supports render_id, prompt, length_seconds, and MiniMax runtime controls only"}
    render_id = job_input.get("render_id")
    if not isinstance(render_id, str) or not render_id.strip():
        return {"error": "input.render_id is required for video extension"}
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "input.prompt is required for video extension"}
    prompt = prompt.strip()

    try:
        metadata, image = _extension_frame(render_id.strip())
    except ArchiveError as exc:
        return {"error": str(exc)}

    inherited = metadata.get("request_settings")
    if not isinstance(inherited, dict):
        inherited = {}
    generation_input: dict[str, Any] = {}
    for key in {"length_seconds", *MINIMAX_RUNTIME_OPTION_NAMES}:
        if key in inherited:
            generation_input[key] = inherited[key]
        if key in job_input:
            generation_input[key] = job_input[key]
    generation_input["image"] = image
    generation_input["prompt"] = prompt

    forwarded_job = copy.deepcopy(job)
    forwarded_job["input"] = generation_input
    result = base_handle_job(forwarded_job)
    if not isinstance(result, dict):
        return result
    result["extension"] = {
        "from_render_id": render_id.strip(),
        "method": "last-frame-i2v",
    }

    preset_name = str(metadata.get("preset_name") or "Custom")
    request_settings = {
        key: value for key, value in generation_input.items() if key not in {"image", "prompt"}
    }
    request_settings["extended_from_render_id"] = render_id.strip()
    request_settings["extension_method"] = "last-frame-i2v"
    return _archive_success(
        job=job,
        result=result,
        prompt=prompt,
        preset_name=preset_name,
        request_settings=request_settings,
    )


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return base_handle_job(job)

    action = job_input.get("action")
    if action == "library":
        return _library_action(job_input)
    if action == "extend":
        return _extend_action(job, job_input)

    if action == "status":
        result = base_handle_job(job)
        if isinstance(result, dict) and "error" not in result:
            try:
                result["archive"] = archive_status()
            except ArchiveError as exc:
                result["archive"] = {"configured": False, "error": str(exc)}
            result["app_worker_version"] = APP_WORKER_VERSION
        return result

    # preset_name belongs to the phone app, not the ComfyUI workflow. Newer
    # clients may send it, while older-compatible clients omit it. In the latter
    # case infer the built-in preset from the exact runtime settings.
    preset_name = _infer_preset_name(job_input)
    forwarded_job = job
    if "preset_name" in job_input:
        value = job_input.get("preset_name")
        if not isinstance(value, str) or len(value.strip()) > 60:
            return {"error": "input.preset_name must be a string up to 60 characters"}
        preset_name = value.strip() or preset_name
        forwarded_job = copy.deepcopy(job)
        forwarded_job["input"].pop("preset_name", None)

    result = base_handle_job(forwarded_job)
    if not isinstance(result, dict):
        return result

    original_prompt = str(job_input.get("prompt") or "")
    request_settings = {
        key: value
        for key, value in job_input.items()
        if key not in {"image", "prompt", "preset_name"}
    }
    return _archive_success(
        job=job,
        result=result,
        prompt=original_prompt,
        preset_name=preset_name,
        request_settings=request_settings,
    )

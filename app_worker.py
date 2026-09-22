"""RunPod handler wrapper for prompt enhancement, archiving, library browsing, and video extension."""

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
    download_job_result_video,
    download_render_video,
    list_renders,
    upload_job_video,
)
from prompt_enhancer import PromptEnhancerError, enhance_prompt
from runtime_controls import MINIMAX_RUNTIME_OPTION_NAMES
from worker import handle_job as base_handle_job


LOGGER = logging.getLogger("redgraft-app-worker")
APP_WORKER_VERSION = "redgraft-library-4"

_PRESET_SIGNATURES: dict[str, dict[str, Any]] = {
    "Fast Test": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.40, "mystic_strength": 0.0, "hmnsfw_strength": 0.0, "vagassist_strength": 0.0, "hmpussy_strength": 0.0, "cumshot_strength": 0.0, "steps": 6, "enable_audio": False, "enable_gimm": False},
    "General NSFW": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.50, "mystic_strength": 0.55, "hmnsfw_strength": 0.55, "vagassist_strength": 0.0, "hmpussy_strength": 0.0, "cumshot_strength": 0.0, "steps": 8, "enable_audio": False, "enable_gimm": False},
    "Anatomy Lock": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.45, "mystic_strength": 0.60, "hmnsfw_strength": 0.60, "vagassist_strength": 1.0, "hmpussy_strength": 0.35, "cumshot_strength": 0.0, "steps": 8, "enable_audio": False, "enable_gimm": False},
    "HMNSFW Focus": {"length_seconds": 8, "turbo_strength": 0.50, "m3_strength": 0.35, "mystic_strength": 0.70, "hmnsfw_strength": 1.0, "vagassist_strength": 0.75, "hmpussy_strength": 0.25, "cumshot_strength": 0.0, "steps": 12, "enable_audio": False, "enable_gimm": False},
    "Cumshot I2V": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.40, "mystic_strength": 0.70, "hmnsfw_strength": 0.70, "vagassist_strength": 1.0, "hmpussy_strength": 0.35, "cumshot_strength": 0.70, "steps": 8, "enable_audio": False, "enable_gimm": False},
    "Full Stack": {"length_seconds": 8, "turbo_strength": 0.85, "m3_strength": 0.50, "mystic_strength": 1.0, "hmnsfw_strength": 1.0, "vagassist_strength": 1.0, "hmpussy_strength": 0.35, "cumshot_strength": 0.70, "steps": 8, "enable_audio": True, "enable_gimm": True},
}

_PROMPT_METADATA_FIELDS = {
    "original_prompt",
    "enhanced_prompt",
    "used_prompt",
    "prompt_enhancement_enabled",
    "prompt_enhancement_mode",
    "prompt_enhancement_previewed",
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


def _enhance_action(job_input: dict[str, Any]) -> dict[str, Any]:
    allowed = {"action", "prompt", "mode", "is_extend"}
    if set(job_input) - allowed:
        return {"error": "Prompt enhancement supports only prompt, mode, and is_extend"}
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "input.prompt is required for prompt enhancement"}
    mode = str(job_input.get("mode") or "detailed").strip().lower()
    is_extend = bool(job_input.get("is_extend", False))
    try:
        enhanced = enhance_prompt(prompt, mode=mode, is_extend=is_extend)
    except PromptEnhancerError as exc:
        return {"error": str(exc)}
    return {
        "status": "enhanced_prompt",
        "app_worker_version": APP_WORKER_VERSION,
        "original_prompt": prompt.strip(),
        "enhanced_prompt": enhanced,
        "used_prompt": enhanced,
        "mode": mode,
        "is_extend": is_extend,
    }


def _prompt_metadata(job_input: dict[str, Any], *, is_extend: bool = False) -> tuple[dict[str, Any], str]:
    raw_prompt = job_input.get("prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("input.prompt is required")
    raw_prompt = raw_prompt.strip()
    original = job_input.get("original_prompt")
    if not isinstance(original, str) or not original.strip():
        original = raw_prompt
    else:
        original = original.strip()
    mode = str(job_input.get("prompt_enhancement_mode") or "detailed").strip().lower()
    enabled = bool(job_input.get("prompt_enhancement_enabled", False))
    previewed = bool(job_input.get("prompt_enhancement_previewed", False))
    enhanced = job_input.get("enhanced_prompt")
    if enhanced is not None and not isinstance(enhanced, str):
        raise ValueError("input.enhanced_prompt must be a string")
    enhanced = enhanced.strip() if isinstance(enhanced, str) and enhanced.strip() else None
    used = job_input.get("used_prompt")
    if used is not None and not isinstance(used, str):
        raise ValueError("input.used_prompt must be a string")
    used = used.strip() if isinstance(used, str) and used.strip() else raw_prompt

    # API callers can request automatic enhancement without doing the preview
    # action first. Phone clients normally provide enhanced_prompt after preview.
    if enabled and enhanced is None:
        try:
            enhanced = enhance_prompt(original, mode=mode, is_extend=is_extend)
        except PromptEnhancerError as exc:
            raise ValueError(str(exc)) from exc
        used = enhanced

    return {
        "original_prompt": original,
        "enhanced_prompt": enhanced,
        "used_prompt": used,
        "prompt_enhancement_enabled": enabled,
        "prompt_enhancement_mode": mode,
        "prompt_enhancement_previewed": previewed,
    }, used


def _archive_success(
    *,
    job: dict[str, Any],
    result: dict[str, Any],
    prompt: str,
    preset_name: str,
    request_settings: dict[str, Any],
    prompt_metadata: dict[str, Any] | None = None,
    extension_metadata: dict[str, Any] | None = None,
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
            prompt_metadata=prompt_metadata,
            extension_metadata=extension_metadata,
        )
        if archived is None:
            result["archive"] = {
                "configured": False,
                "permanent": False,
                "message": "Permanent object storage is not configured on this RunPod endpoint.",
            }
        else:
            result["archive"] = archived
            result["videos"] = archived["videos"]
    except ArchiveError as exc:
        LOGGER.exception("Permanent archive metadata failed")
        result["archive"] = {
            "configured": True,
            "permanent": False,
            "error": str(exc),
        }
    return result


def _extract_last_frame(source: Path, frame: Path) -> str:
    timeout = max(30, min(300, int(os.getenv("EXTEND_FRAME_TIMEOUT_SECONDS", "120"))))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-sseof", "-0.20", "-i", str(source), "-frames:v", "1", "-q:v", "2", str(frame),
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
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _prepare_extension_source(render_id: str, root: Path) -> tuple[dict[str, Any], Path, str]:
    source = root / "source.mp4"
    frame = root / "last-frame.jpg"
    metadata, _ = download_render_video(render_id, source)
    return metadata, source, _extract_last_frame(source, frame)


def _has_audio(path: Path) -> bool:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool((completed.stdout or "").strip())


def _concat_videos(first: Path, second: Path, output: Path) -> None:
    """Append the continuation. Try lossless stream copy, then robust re-encode."""
    timeout = max(120, min(1800, int(os.getenv("EXTEND_MERGE_TIMEOUT_SECONDS", "900"))))
    concat_list = output.with_suffix(".txt")
    concat_list.write_text(f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n", encoding="utf-8")
    copy_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", "-movflags", "+faststart", str(output),
    ]
    try:
        copied = subprocess.run(copy_command, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        copied = None
        copy_error = str(exc)
    else:
        copy_error = (copied.stderr or "").strip()[-500:]
    if copied is not None and copied.returncode == 0 and output.is_file() and output.stat().st_size > 0:
        return

    output.unlink(missing_ok=True)
    both_audio = _has_audio(first) and _has_audio(second)
    if both_audio:
        filter_graph = "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]"
        reencode = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(first), "-i", str(second), "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
    else:
        filter_graph = "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]"
        reencode = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(first), "-i", str(second), "-filter_complex", filter_graph,
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-movflags", "+faststart", str(output),
        ]
    try:
        encoded = subprocess.run(reencode, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise ArchiveError("Video extension requires ffmpeg in the worker image") from exc
    except subprocess.TimeoutExpired as exc:
        raise ArchiveError("Timed out while combining the original and continuation videos") from exc
    if encoded.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        detail = (encoded.stderr or copy_error or "").strip()[-800:]
        raise ArchiveError("Could not append the continuation to the original video" + (f": {detail}" if detail else ""))


def _extend_action(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "action", "render_id", "prompt", "length_seconds", *MINIMAX_RUNTIME_OPTION_NAMES,
        *_PROMPT_METADATA_FIELDS,
    }
    if set(job_input) - allowed:
        return {"error": "Video extension supports render_id, prompt, length_seconds, prompt enhancement metadata, and MiniMax runtime controls only"}
    render_id = job_input.get("render_id")
    if not isinstance(render_id, str) or not render_id.strip():
        return {"error": "input.render_id is required for video extension"}
    try:
        prompt_meta, used_prompt = _prompt_metadata(job_input, is_extend=True)
    except ValueError as exc:
        return {"error": str(exc)}

    with tempfile.TemporaryDirectory(prefix="redgraft-extend-") as folder:
        root = Path(folder)
        try:
            metadata, source, image = _prepare_extension_source(render_id.strip(), root)
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
        generation_input["prompt"] = used_prompt

        forwarded_job = copy.deepcopy(job)
        forwarded_job["input"] = generation_input
        result = base_handle_job(forwarded_job)
        if not isinstance(result, dict):
            return result
        if result.get("status") != "success":
            result["app_worker_version"] = APP_WORKER_VERSION
            return result

        continuation_path = root / "continuation.mp4"
        combined_path = root / "extended.mp4"
        try:
            continuation_record = download_job_result_video(job.get("id"), result, continuation_path)
            _concat_videos(source, continuation_path, combined_path)
            combined = upload_job_video(job.get("id"), combined_path, "extended.mp4")
        except ArchiveError as exc:
            # Keep the generated continuation usable if concatenation fails instead
            # of throwing away an expensive successful generation.
            LOGGER.exception("Extension merge failed; archiving continuation segment")
            extension_metadata = {
                "from_render_id": render_id.strip(),
                "method": "last-frame-i2v",
                "merged": False,
                "merge_error": str(exc),
            }
        else:
            result["videos"] = [combined]
            extension_metadata = {
                "from_render_id": render_id.strip(),
                "method": "last-frame-i2v-concat",
                "merged": True,
                "continuation_segment": continuation_record,
                "combined_video": {
                    "key": combined.get("archive_key"),
                    "filename": combined.get("filename"),
                },
            }
        result["extension"] = extension_metadata

        preset_name = str(metadata.get("preset_name") or "Custom")
        request_settings = {
            key: value for key, value in generation_input.items() if key not in {"image", "prompt"}
        }
        request_settings["extended_from_render_id"] = render_id.strip()
        request_settings["extension_method"] = extension_metadata["method"]
        request_settings["extension_merged"] = bool(extension_metadata.get("merged"))
        return _archive_success(
            job=job,
            result=result,
            prompt=used_prompt,
            preset_name=preset_name,
            request_settings=request_settings,
            prompt_metadata=prompt_meta,
            extension_metadata=extension_metadata,
        )


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return base_handle_job(job)

    action = job_input.get("action")
    if action == "library":
        return _library_action(job_input)
    if action == "enhance_prompt":
        return _enhance_action(job_input)
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

    try:
        prompt_meta, used_prompt = _prompt_metadata(job_input)
    except ValueError as exc:
        return {"error": str(exc)}

    preset_name = _infer_preset_name(job_input)
    forwarded_job = copy.deepcopy(job)
    forwarded_input = forwarded_job["input"]
    forwarded_input["prompt"] = used_prompt

    if "preset_name" in job_input:
        value = job_input.get("preset_name")
        if not isinstance(value, str) or len(value.strip()) > 60:
            return {"error": "input.preset_name must be a string up to 60 characters"}
        preset_name = value.strip() or preset_name
    forwarded_input.pop("preset_name", None)
    for key in _PROMPT_METADATA_FIELDS:
        forwarded_input.pop(key, None)

    result = base_handle_job(forwarded_job)
    if not isinstance(result, dict):
        return result

    request_settings = {
        key: value
        for key, value in job_input.items()
        if key not in {"image", "prompt", "preset_name", *_PROMPT_METADATA_FIELDS}
    }
    request_settings["prompt_enhancement_enabled"] = prompt_meta["prompt_enhancement_enabled"]
    request_settings["prompt_enhancement_mode"] = prompt_meta["prompt_enhancement_mode"]
    return _archive_success(
        job=job,
        result=result,
        prompt=used_prompt,
        preset_name=preset_name,
        request_settings=request_settings,
        prompt_metadata=prompt_meta,
    )

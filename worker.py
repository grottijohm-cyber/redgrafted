"""LTX 2.5 image-to-video RunPod worker.

Each request supplies exactly an image and a short prompt. The fixed workflow
uses ComfyUI's native LTX 2.5 prompt enhancer before video generation.
"""

from __future__ import annotations

import base64
import binascii
import copy
import io
import json
import logging
import mimetypes
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

import model_setup
from model_setup import ModelSetupError, ensure_models
from file_integrity import check_deadline
from video_delivery import DeliveryError, inline_video


LOGGER = logging.getLogger("ltx25-worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOW_PATH = Path(os.getenv("WORKFLOW_PATH", "/app/api-workflow.json"))
IMAGE_NODE_ID = "395"
PROMPT_ENHANCER_NODE_ID = "380"
MAIN_SEED_NODE_ID = "339"
OUTPUT_NODE_ID = "75"
COMFY_START_TIMEOUT_SECONDS = int(os.getenv("COMFY_START_TIMEOUT_SECONDS", "900"))
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "6600"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "2"))
MAX_INPUT_IMAGE_BYTES = int(os.getenv("MAX_INPUT_IMAGE_BYTES", str(25 * 1024 * 1024)))
MAX_PROMPT_CHARACTERS = int(os.getenv("MAX_PROMPT_CHARACTERS", "10000"))
MAX_INLINE_OUTPUT_BYTES = min(6_000_000, max(1024, int(os.getenv("MAX_INLINE_OUTPUT_BYTES", "6000000"))))
MAX_RESULT_BYTES = 9_000_000
WORKER_VERSION = "runpod-reliability-1"

FORMAT_TO_EXTENSION = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
    "BMP": ".bmp",
}


class WorkerError(RuntimeError):
    """An expected, user-facing worker failure."""


def load_workflow(path: Path | None = None) -> dict[str, Any]:
    path = path or WORKFLOW_PATH
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError(f"Could not load API workflow at {path}: {exc}") from exc

    if not isinstance(workflow, dict) or not workflow:
        raise WorkerError("The bundled API workflow is empty or is not a JSON object")

    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            raise WorkerError(f"Workflow node {node_id!r} is not in ComfyUI API format")

    expected_types = {
        IMAGE_NODE_ID: "LoadImage",
        PROMPT_ENHANCER_NODE_ID: "TextGenerateLTX2Prompt",
        MAIN_SEED_NODE_ID: "RandomNoise",
        OUTPUT_NODE_ID: "SaveVideo",
    }
    for node_id, class_type in expected_types.items():
        if workflow.get(node_id, {}).get("class_type") != class_type:
            raise WorkerError(f"Workflow node {node_id} is missing or is not {class_type}")

    for node_id, node in workflow.items():
        for input_value in node["inputs"].values():
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[1], int)
                and str(input_value[0]) not in workflow
            ):
                raise WorkerError(
                    f"Workflow node {node_id} references missing node {input_value[0]}"
                )
    return workflow


def _read_limited_response(response: requests.Response, limit: int) -> bytes:
    declared_size = response.headers.get("Content-Length")
    if declared_size:
        try:
            if int(declared_size) > limit:
                raise WorkerError(f"Input image exceeds the {limit}-byte limit")
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise WorkerError(f"Input image exceeds the {limit}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_image_input(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("input.image must be a non-empty URL or base64 string")
    value = value.strip()

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        try:
            with requests.get(
                value,
                stream=True,
                timeout=(15, 180),
                headers={"User-Agent": "runpod-ltx25-i2v-worker/1.0"},
            ) as response:
                response.raise_for_status()
                return _read_limited_response(response, MAX_INPUT_IMAGE_BYTES)
        except requests.RequestException as exc:
            raise WorkerError(f"Could not download input image: {exc}") from exc

    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or ";base64" not in header.lower():
            raise WorkerError("input.image data URI must use base64 encoding")
    else:
        encoded = value

    if len(encoded) > ((MAX_INPUT_IMAGE_BYTES + 2) // 3) * 4 + 1024:
        raise WorkerError("Base64 input image exceeds the size limit")
    encoded = "".join(encoded.split())
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WorkerError("input.image is neither a valid URL nor valid base64") from exc
    if len(image_bytes) > MAX_INPUT_IMAGE_BYTES:
        raise WorkerError(f"Input image exceeds the {MAX_INPUT_IMAGE_BYTES}-byte limit")
    return image_bytes


def _validate_image(image_bytes: bytes) -> tuple[str, str]:
    if not image_bytes:
        raise WorkerError("The input image is empty")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format or ""
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise WorkerError("The supplied data is not a valid supported image") from exc

    extension = FORMAT_TO_EXTENSION.get(image_format.upper())
    if not extension:
        raise WorkerError(f"Unsupported image format: {image_format or 'unknown'}")
    mime_type = mimetypes.types_map.get(extension, "application/octet-stream")
    return extension, mime_type


def wait_for_comfyui(deadline: float | None = None) -> None:
    startup_deadline = time.monotonic() + COMFY_START_TIMEOUT_SECONDS
    deadline = min(deadline, startup_deadline) if deadline is not None else startup_deadline
    last_error = "not reachable"
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if response.ok:
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"ComfyUI did not start in time: {last_error}")


def upload_input_image(image_bytes: bytes, filename: str, mime_type: str) -> str:
    try:
        response = requests.post(
            f"{COMFY_URL}/upload/image",
            files={"image": (filename, image_bytes, mime_type)},
            data={"type": "input", "overwrite": "true"},
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise WorkerError(f"ComfyUI rejected the input image: {exc}") from exc

    uploaded_name = result.get("name", filename)
    subfolder = result.get("subfolder", "")
    return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name


def queue_workflow(workflow: dict[str, Any], client_id: str) -> str:
    try:
        response = requests.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=180,
        )
        if not response.ok:
            raise WorkerError(
                "ComfyUI workflow validation failed "
                f"(HTTP {response.status_code}): {response.text[:8000]}"
            )
        result = response.json()
    except requests.RequestException as exc:
        raise WorkerError(f"Could not queue the ComfyUI workflow: {exc}") from exc

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise WorkerError(f"ComfyUI did not return a prompt_id: {result}")
    return str(prompt_id)


def _history_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for message in reversed(status.get("messages", [])):
            if isinstance(message, list) and len(message) == 2:
                kind, details = message
                if kind == "execution_error" and isinstance(details, dict):
                    return (
                        f"Node {details.get('node_id')} ({details.get('node_type')}): "
                        f"{details.get('exception_message', 'execution failed')}"
                    )
        return "ComfyUI reported an execution error"
    return None


def wait_for_history(prompt_id: str, deadline: float | None = None) -> dict[str, Any]:
    deadline = deadline if deadline is not None else time.monotonic() + JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            entry = response.json().get(prompt_id)
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("History polling failed temporarily: %s", exc)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if entry:
            error = _history_error(entry)
            if error:
                raise WorkerError(error)
            if entry.get("status", {}).get("completed"):
                return entry
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Generation exceeded the {JOB_TIMEOUT_SECONDS}-second job budget")


def _walk_file_descriptors(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            yield value
            return
        for nested in value.values():
            yield from _walk_file_descriptors(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_file_descriptors(nested)


def get_output_descriptors(entry: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = entry.get("outputs", {})
    preferred = outputs.get(OUTPUT_NODE_ID)
    values = [preferred] if preferred else list(outputs.values())
    descriptors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        for descriptor in _walk_file_descriptors(value):
            key = (
                descriptor.get("type", "output"),
                descriptor.get("subfolder", ""),
                descriptor["filename"],
            )
            if key not in seen and key[0] != "temp":
                seen.add(key)
                descriptors.append(descriptor)
    if not descriptors:
        raise WorkerError(
            f"Workflow completed but output node {OUTPUT_NODE_ID} returned no files"
        )
    return descriptors


def _safe_output_path(descriptor: dict[str, Any]) -> Path:
    output_root = Path(os.getenv("COMFY_OUTPUT_DIR", "/comfyui/output")).resolve()
    filename = Path(descriptor["filename"]).name
    subfolder = str(descriptor.get("subfolder", "")).replace("\\", "/").strip("/")
    candidate = (output_root / subfolder / filename).resolve()
    if output_root not in candidate.parents and candidate != output_root:
        raise WorkerError("ComfyUI returned an unsafe output path")
    if not candidate.is_file():
        raise WorkerError(f"Generated output file was not found: {filename}")
    return candidate


def _safe_input_path(uploaded_name: str) -> Path:
    input_root = Path(os.getenv("COMFY_INPUT_DIR", "/comfyui/input")).resolve()
    relative_name = uploaded_name.replace("\\", "/").strip("/")
    candidate = (input_root / relative_name).resolve()
    if input_root not in candidate.parents:
        raise WorkerError("ComfyUI returned an unsafe input path")
    return candidate


def _bucket_configuration() -> dict[str, str | None]:
    keys = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME")
    values = {key: os.getenv(key) for key in keys}
    if any(values.values()) and not all(values.values()):
        missing = ", ".join(key for key, value in values.items() if not value)
        raise WorkerError(f"Object-storage configuration is incomplete; missing: {missing}")
    return values


def publish_output(job_id: str, descriptor: dict[str, Any],
                   deadline: float | None = None, inline_budget: int | None = None) -> dict[str, Any]:
    check_deadline(deadline)
    output_path = _safe_output_path(descriptor)
    mime_type = mimetypes.guess_type(output_path.name)[0] or "application/octet-stream"
    bucket = _bucket_configuration()
    if all(bucket.values()):
        from runpod.serverless.utils import rp_upload
        url = rp_upload.upload_file_to_bucket(
            file_name=output_path.name, file_location=str(output_path),
            bucket_name=bucket["BUCKET_NAME"], prefix=job_id,
            extra_args={"ContentType": mime_type},
        )
        if not isinstance(url, str) or urlparse(url).scheme not in {"https", "http"}:
            raise WorkerError("Output upload did not return a downloadable URL")
        return {"filename": output_path.name, "type": "url", "url": url,
                "mime_type": mime_type, "compressed_for_delivery": False}

    budget = MAX_INLINE_OUTPUT_BYTES if inline_budget is None else min(MAX_INLINE_OUTPUT_BYTES, inline_budget)
    payload, compressed = inline_video(output_path, budget, deadline)
    return {"filename": output_path.name, "type": "base64",
            "data": base64.b64encode(payload).decode("ascii"), "mime_type": mime_type,
            "compressed_for_delivery": compressed, "size_bytes": len(payload)}


def report_progress(job: dict, stage: str) -> None:
    LOGGER.info("Job stage: %s", stage)
    if not os.getenv("RUNPOD_POD_ID") or not job.get("id"):
        return
    try:
        import runpod
        runpod.serverless.progress_update(job, {"stage": stage})
    except Exception:
        LOGGER.warning("Could not send progress update")


def cancel_workflow(prompt_id: str) -> None:
    """Delete this pending prompt and interrupt it only if it owns the GPU."""
    try:
        response = requests.get(f"{COMFY_URL}/queue", timeout=10)
        response.raise_for_status()
        running = response.json().get("queue_running", [])
        requests.post(f"{COMFY_URL}/queue", json={"delete": [prompt_id]}, timeout=10).raise_for_status()
        if any(isinstance(item, list) and len(item) > 1 and item[1] == prompt_id for item in running):
            requests.post(f"{COMFY_URL}/interrupt", json={}, timeout=10).raise_for_status()
    except (requests.RequestException, ValueError):
        LOGGER.warning("Could not confirm cancellation of prompt %s", prompt_id)


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return {"error": "Request must contain an input object"}
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    if "action" in job_input:
        if set(job_input) != {"action"}:
            return {"error": "Setup/status requests contain only input.action"}
        try:
            if job_input["action"] == "status":
                return {"status": "diagnostics", "worker_version": WORKER_VERSION,
                        "models": model_setup.model_status()}
            if job_input["action"] == "setup":
                from bootstrap_hf_repo import bootstrap_bundle
                report_progress(job, "Preparing and publishing model bundle")
                return bootstrap_bundle(deadline)
            return {"error": "Supported actions are setup and status"}
        except (ModelSetupError, TimeoutError) as exc:
            return {"error": str(exc)}
        except Exception:
            LOGGER.exception("Setup/status job failed")
            return {"error": "Setup failed; check worker logs for provider access or upload errors"}

    if set(job_input) - {"image", "prompt"}:
        return {"error": "Only input.image and input.prompt are supported"}
    if "image" not in job_input:
        return {"error": "Missing required input.image"}
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Missing required non-empty input.prompt"}
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        return {"error": f"input.prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit"}
    if model_setup.BOOTSTRAP_MODE:
        return {"error": "This endpoint is in one-time setup mode. Send input.action=setup, then select the completed Cached Model and remove BOOTSTRAP_HF_REPO before generating."}

    input_path, prompt_id = None, None
    generation_finished = False
    output_paths: list[Path] = []
    delivered = False
    try:
        # Catch configuration/template errors before any model download or GPU work.
        _bucket_configuration()
        workflow = copy.deepcopy(load_workflow())
        report_progress(job, "Checking image and model files")
        image_bytes = _decode_image_input(job_input["image"])
        extension, mime_type = _validate_image(image_bytes)
        ensure_models(deadline)
        wait_for_comfyui(deadline)
        check_deadline(deadline)

        job_id = str(job.get("id") or uuid.uuid4())
        safe_job_id = re.sub(r"[^A-Za-z0-9_-]", "-", job_id)[:64]
        filename = f"runpod-{safe_job_id}-{uuid.uuid4().hex[:8]}{extension}"
        uploaded_name = upload_input_image(image_bytes, filename, mime_type)
        input_path = _safe_input_path(uploaded_name)
        workflow[IMAGE_NODE_ID]["inputs"]["image"] = uploaded_name
        workflow[PROMPT_ENHANCER_NODE_ID]["inputs"]["prompt"] = prompt
        seed = secrets.randbits(63)
        workflow[MAIN_SEED_NODE_ID]["inputs"]["noise_seed"] = seed

        report_progress(job, "Enhancing prompt and generating video")
        prompt_id = queue_workflow(workflow, client_id=uuid.uuid4().hex)
        history = wait_for_history(prompt_id, deadline)
        generation_finished = True
        descriptors = get_output_descriptors(history)
        output_paths = [_safe_output_path(item) for item in descriptors]
        report_progress(job, "Preparing video for download")
        # Bound the combined response, including cases with several output files.
        per_file_budget = MAX_INLINE_OUTPUT_BYTES // len(descriptors)
        outputs = [publish_output(safe_job_id, item, deadline, per_file_budget) for item in descriptors]
        videos = [item for item in outputs if item["mime_type"].startswith("video/")]
        result = {"status": "success", "worker_version": WORKER_VERSION,
                  "prompt_id": prompt_id, "seed": seed, "prompt_enhanced": True,
                  "videos": videos or outputs}
        if len(json.dumps(result).encode("utf-8")) > MAX_RESULT_BYTES:
            raise WorkerError("Video response exceeds the total delivery budget")
        check_deadline(deadline)
        delivered = True
        return result
    except TimeoutError as exc:
        if prompt_id and not generation_finished:
            cancel_workflow(prompt_id)
        # Reset the worker after a deadline to prevent orphaned GPU work.
        return {"error": str(exc), "refresh_worker": True}
    except (WorkerError, ModelSetupError, DeliveryError) as exc:
        LOGGER.exception("Job failed")
        if prompt_id and not generation_finished:
            cancel_workflow(prompt_id)
        return {"error": str(exc)}
    except Exception:
        LOGGER.exception("Unexpected job failure")
        if prompt_id and not generation_finished:
            cancel_workflow(prompt_id)
        return {"error": "Unexpected worker failure; check the RunPod worker logs"}
    finally:
        cleanup = output_paths if delivered else []
        if input_path is not None and (prompt_id is None or generation_finished):
            cleanup = [input_path, *cleanup]
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not clean up %s", path.name)
        if output_paths and not delivered:
            LOGGER.warning("Delivery failed. Originals remain on this worker's temporary disk: %s",
                           [path.name for path in output_paths])

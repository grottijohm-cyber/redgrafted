"""LTX image-to-video RunPod worker with version-matched model profiles.

Each request supplies an image and an already prepared prompt. The fixed workflow
passes that text directly to the video text encoder and requests about ten seconds.
"""

from __future__ import annotations

import base64
import binascii
import copy
import io
import json
import logging
import math
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
import runtime_health
from runtime_health import ComfyMonitor, ComfyUnavailableError
from model_setup import ModelSetupError, ensure_models
from file_integrity import check_deadline
from video_delivery import DeliveryError, inline_video


LOGGER = logging.getLogger("ltx25-worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOW_PATH = Path(os.getenv("WORKFLOW_PATH") or f"/app/{model_setup.WORKFLOW_FILENAME}")
IMAGE_NODE_ID = "395"
POSITIVE_PROMPT_NODE_ID = "364"
MAIN_SEED_NODE_ID = "339"
OUTPUT_NODE_ID = "75"
COMFY_START_TIMEOUT_SECONDS = int(os.getenv("COMFY_START_TIMEOUT_SECONDS", "900"))
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "6600"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "2"))
MAX_INPUT_IMAGE_BYTES = int(os.getenv("MAX_INPUT_IMAGE_BYTES", str(25 * 1024 * 1024)))
MAX_PROMPT_CHARACTERS = int(os.getenv("MAX_PROMPT_CHARACTERS", "10000"))
MAX_INLINE_OUTPUT_BYTES = min(6_000_000, max(1024, int(os.getenv("MAX_INLINE_OUTPUT_BYTES", "6000000"))))
MAX_RESULT_BYTES = 9_000_000
COMFY_UNREACHABLE_TIMEOUT_SECONDS = max(1, int(os.getenv("COMFY_UNREACHABLE_TIMEOUT_SECONDS", "60")))
WORKER_VERSION = "runpod-minimax-5"

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
        POSITIVE_PROMPT_NODE_ID: "MiniMaxH3ImageToVideo" if model_setup.MODEL_PROFILE == "minimax" else "CLIPTextEncode",
        MAIN_SEED_NODE_ID: "RandomNoise",
        OUTPUT_NODE_ID: "SaveVideo",
    }
    for node_id, class_type in expected_types.items():
        if workflow.get(node_id, {}).get("class_type") != class_type:
            raise WorkerError(f"Workflow node {node_id} is missing or is not {class_type}")
    if workflow.get("384", {}).get("class_type") not in {"UNETLoader", "CheckpointLoaderSimple"}:
        raise WorkerError("Workflow node 384 must load the configured model")

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
    if not isinstance(workflow[POSITIVE_PROMPT_NODE_ID]["inputs"].get("prompt" if model_setup.MODEL_PROFILE == "minimax" else "text"), str):
        raise WorkerError("The positive prompt must accept prepared text directly")
    workflow_details(workflow)
    return workflow


def workflow_details(workflow: dict[str, Any]) -> dict[str, Any]:
    """Report the requested configuration and check audio/video synchronization."""
    if model_setup.MODEL_PROFILE == "minimax":
        return minimax_workflow_details(workflow)
    try:
        frames = workflow["356"]["inputs"]["length"]
        audio_frames = workflow["366"]["inputs"]["frames_number"]
        fps = float(workflow["370"]["inputs"]["fps"])
        audio_fps = float(workflow["366"]["inputs"]["frame_rate"])
        conditioning_fps = float(workflow["365"]["inputs"]["frame_rate"])
        loader = workflow["384"]
        checkpoint = loader["inputs"]["ckpt_name" if loader["class_type"] == "CheckpointLoaderSimple" else "unet_name"]
        for sampler, guider in (("344", "388"), ("368", "391")):
            if (workflow[sampler]["inputs"]["guider"] != [guider, 0]
                    or workflow[guider]["inputs"]["model"] != ["384", 0]):
                raise WorkerError("Both sampling passes must use the configured checkpoint")
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerError("The video, audio, or checkpoint configuration is incomplete") from exc
    if (type(frames) is not int or frames < 1 or (frames - 1) % 8
            or type(audio_frames) is not int or audio_frames != frames):
        raise WorkerError("Video and audio must use the same 8*n+1 frame count")
    if not math.isfinite(fps) or fps <= 0 or audio_fps != fps or conditioning_fps != fps:
        raise WorkerError("Video, audio, and conditioning frame rates must match")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise WorkerError("The generation checkpoint filename is missing")
    validate_model_configuration(workflow)
    return {"model_profile": model_setup.MODEL_PROFILE,
            "checkpoint": checkpoint, "frames": frames, "fps": fps,
            "duration_seconds": round(frames / fps, 3), "prompt_enhanced": False}



def minimax_workflow_details(workflow: dict[str, Any]) -> dict[str, Any]:
    """Check the MiniMax AV frame grid, encoders, and both adapter connections."""
    validate_model_configuration(workflow)
    try:
        cond = workflow["364"]["inputs"]
        frames, fps = cond["length"], workflow["370"]["inputs"]["fps"]
        if type(frames) is not int or frames < 5 or frames % 17 != 5 or fps != 24:
            raise WorkerError("MiniMax requires 17*n+5 frames at 24 fps")
        for axis in ("width", "height"):
            value = cond[axis]
            if type(value) is not int or value < 32 or value % 32 or workflow["350"]["inputs"][axis] != value:
                raise WorkerError("MiniMax image and canvas dimensions must match and be multiples of 32")
        required = {
            ("364", "clip"): ["387", 0], ("364", "vae"): ["385", 0],
            ("364", "first_frame"): ["350", 0], ("350", "image"): ["395", 0],
            ("390", "model"): ["384", 0], ("391", "model"): ["390", 0],
            ("388", "model"): ["391", 0], ("388", "conditioning"): ["364", 0],
            ("397", "model"): ["391", 0], ("344", "guider"): ["388", 0],
            ("344", "latent_image"): ["364", 1], ("344", "sigmas"): ["397", 0],
            ("374", "samples"): ["344", 0], ("374", "vae"): ["385", 0],
            ("358", "samples"): ["344", 0], ("358", "vae"): ["386", 0],
            ("370", "images"): ["374", 0], ("370", "audio"): ["358", 0],
        }
        if any(workflow[n]["inputs"].get(k) != v for (n, k), v in required.items()):
            raise WorkerError("MiniMax conditioning, adapter, or AV decode connections are invalid")
        if workflow["387"]["inputs"]["type"] != "minimax":
            raise WorkerError("MiniMax requires its matching Qwen text encoder")
        if workflow["397"]["inputs"]["steps"] != 8 or workflow["390"]["inputs"]["strength_model"] != 1.0:
            raise WorkerError("MiniMax eight-step schedule must match its turbo adapter")
        return {"model_profile": "minimax", "checkpoint": workflow["384"]["inputs"]["unet_name"],
                "frames": frames, "fps": fps, "duration_seconds": round(frames / fps, 3),
                "width": cond["width"], "height": cond["height"], "prompt_enhanced": False}
    except (KeyError, TypeError) as exc:
        raise WorkerError("The MiniMax workflow configuration is incomplete") from exc

def validate_model_configuration(workflow: dict[str, Any]) -> None:
    """Reject profile/workflow mismatches before downloads or GPU work."""
    file_fields = {
        "UNETLoader": {"unet_name": "diffusion_models"},
        "CheckpointLoaderSimple": {"ckpt_name": "checkpoints"},
        "CLIPLoader": {"clip_name": "text_encoders"},
        "LTXAVTextEncoderLoader": {"text_encoder": "text_encoders", "ckpt_name": "checkpoints"},
        "LTXVAudioVAELoader": {"ckpt_name": "checkpoints"},
        "VAELoader": {"vae_name": "vae"},
        "LatentUpscaleModelLoader": {"model_name": "latent_upscale_models"},
        "LoraLoaderModelOnly": {"lora_name": "loras"},
    }
    referenced = set()
    try:
        for node in workflow.values():
            for field, folder in file_fields.get(node["class_type"], {}).items():
                referenced.add(f"{folder}/{node['inputs'][field]}")
    except (KeyError, TypeError) as exc:
        raise WorkerError("A model loader is missing its filename") from exc
    if referenced != set(model_setup.ALL_MODEL_PATHS):
        raise WorkerError(f"Workflow model files do not match MODEL_PROFILE={model_setup.MODEL_PROFILE}; remove an old WORKFLOW_PATH override")


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


def wait_for_comfyui(deadline: float | None = None, monitor: ComfyMonitor | None = None) -> None:
    monitor = monitor or ComfyMonitor()
    startup_deadline = time.monotonic() + COMFY_START_TIMEOUT_SECONDS
    deadline = min(deadline, startup_deadline) if deadline is not None else startup_deadline
    last_error = "not reachable"
    while time.monotonic() < deadline:
        monitor.check()
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


def wait_for_history(prompt_id: str, deadline: float | None = None,
                     monitor: ComfyMonitor | None = None) -> dict[str, Any]:
    monitor = monitor or ComfyMonitor()
    deadline = deadline if deadline is not None else time.monotonic() + JOB_TIMEOUT_SECONDS
    unreachable_since = None
    next_memory_log = 0
    while time.monotonic() < deadline:
        monitor.check()
        if time.monotonic() >= next_memory_log:
            LOGGER.info("Generation memory: %s", json.dumps(runtime_health.memory_snapshot()))
            next_memory_log = time.monotonic() + 60
        try:
            response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=(5, 30))
            # A reachable server may be busy. Read timeouts/HTTP failures are
            # not proof that it died and must not trigger the refusal timer.
            unreachable_since = None
            response.raise_for_status()
            entry = response.json().get(prompt_id)
        except requests.ConnectionError as exc:
            monitor.check()
            now = time.monotonic()
            if unreachable_since is None:
                unreachable_since = now
            if now - unreachable_since >= COMFY_UNREACHABLE_TIMEOUT_SECONDS:
                monitor.fail(f"ComfyUI has been unreachable for {COMFY_UNREACHABLE_TIMEOUT_SECONDS} seconds after accepting the job")
            LOGGER.warning("History connection failed temporarily: %s", exc)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        except (requests.RequestException, ValueError) as exc:
            unreachable_since = None
            monitor.check()
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
                        "runtime": runtime_health.diagnostics(),
                        "workflow": workflow_details(load_workflow()),
                        "models": model_setup.model_status()}
            if job_input["action"] == "setup":
                from bootstrap_hf_repo import bootstrap_bundle
                report_progress(job, "Preparing and publishing model bundle")
                return bootstrap_bundle(deadline)
            return {"error": "Supported actions are setup and status"}
        except (WorkerError, ModelSetupError, TimeoutError) as exc:
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
        configuration = workflow_details(workflow)
        report_progress(job, "Checking image and model files")
        image_bytes = _decode_image_input(job_input["image"])
        extension, mime_type = _validate_image(image_bytes)
        ensure_models(deadline)
        monitor = ComfyMonitor()
        LOGGER.info("Job runtime diagnostics: %s", json.dumps(runtime_health.diagnostics()))
        wait_for_comfyui(deadline, monitor)
        check_deadline(deadline)

        job_id = str(job.get("id") or uuid.uuid4())
        safe_job_id = re.sub(r"[^A-Za-z0-9_-]", "-", job_id)[:64]
        filename = f"runpod-{safe_job_id}-{uuid.uuid4().hex[:8]}{extension}"
        uploaded_name = upload_input_image(image_bytes, filename, mime_type)
        input_path = _safe_input_path(uploaded_name)
        workflow[IMAGE_NODE_ID]["inputs"]["image"] = uploaded_name
        workflow[POSITIVE_PROMPT_NODE_ID]["inputs"]["prompt" if model_setup.MODEL_PROFILE == "minimax" else "text"] = prompt
        seed = secrets.randbits(63)
        workflow[MAIN_SEED_NODE_ID]["inputs"]["noise_seed"] = seed

        LOGGER.info("Requested generation configuration: %s", json.dumps(configuration))
        report_progress(job, "Generating video from your prepared prompt")
        prompt_id = queue_workflow(workflow, client_id=uuid.uuid4().hex)
        history = wait_for_history(prompt_id, deadline, monitor)
        generation_finished = True
        descriptors = get_output_descriptors(history)
        output_paths = [_safe_output_path(item) for item in descriptors]
        report_progress(job, "Preparing video for download")
        # Bound the combined response, including cases with several output files.
        per_file_budget = MAX_INLINE_OUTPUT_BYTES // len(descriptors)
        outputs = [publish_output(safe_job_id, item, deadline, per_file_budget) for item in descriptors]
        videos = [item for item in outputs if item["mime_type"].startswith("video/")]
        result = {"status": "success", "worker_version": WORKER_VERSION,
                  "prompt_id": prompt_id, "seed": seed, "prompt_enhanced": False,
                  "workflow": configuration,
                  "videos": videos or outputs}
        if len(json.dumps(result).encode("utf-8")) > MAX_RESULT_BYTES:
            raise WorkerError("Video response exceeds the total delivery budget")
        check_deadline(deadline)
        delivered = True
        return result
    except ComfyUnavailableError as exc:
        # Do not poll/cancel a dead server for another hour. RunPod receives a
        # failed job and a request to replace this unusable worker.
        return {"error": str(exc), "refresh_worker": True,
                "runtime": runtime_health.diagnostics()}
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

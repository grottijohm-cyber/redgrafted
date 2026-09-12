"""Prepare REDGraft LTX 2.5 for RunPod Cached Models.

Official Lightricks LTX 2.5 support files are supplied by RunPod's Cached Model
mount. Only REDGraft and the adult-capable prompt-enhancer extras are downloaded
into the worker's local ComfyUI model directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

LOGGER = logging.getLogger("ltx25-model-setup")
COMFY_MODELS = Path(os.getenv("COMFY_MODELS", "/comfyui/models"))
CACHE_ROOT = Path(os.getenv("RUNPOD_MODEL_CACHE", "/runpod-volume/huggingface-cache/hub"))
DOWNLOAD_WORKERS = int(os.getenv("MODEL_DOWNLOAD_WORKERS", "3"))
MODEL_DISK_SAFETY_BYTES = int(os.getenv("MODEL_DISK_SAFETY_BYTES", str(5 * 1024**3)))


@dataclass(frozen=True)
class ModelFile:
    url: str
    relative_path: str
    min_bytes: int
    token_env: str | None = None


# These are the only large files we download ourselves. The official LTX 2.5
# files below are symlinked from RunPod's Cached Model mount instead.
EXTRA_FILES = (
    ModelFile(
        "https://civitai.com/api/download/models/3250230?fileId=3133376",
        "diffusion_models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
        16_000_000_000,
        "CIVITAI_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/DreamFast/gemma-3-12b-it-heretic-v2/resolve/main/comfyui/gemma-3-12b-it-heretic-v2_int8.safetensors",
        "text_encoders/gemma-3-12b-it-heretic-v2_int8.safetensors",
        13_000_000_000,
        "HF_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/ReubenF10/ComfyUI-Models/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        2_200_000_000,
        "HF_TOKEN",
    ),
)

# Relative paths inside the Lightricks/LTX-2.5 Hugging Face repository.
CACHED_FILES = (
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)


class ModelSetupError(RuntimeError):
    pass


def _find_ltx_snapshot() -> Path:
    repo_root = CACHE_ROOT / "models--Lightricks--LTX-2.5" / "snapshots"
    if not repo_root.is_dir():
        raise ModelSetupError(
            "RunPod Cached Model Lightricks/LTX-2.5 was not found. "
            "Set Cached model to Lightricks/LTX-2.5 and provide the Hugging Face access token."
        )
    snapshots = [p for p in repo_root.iterdir() if p.is_dir()]
    if not snapshots:
        raise ModelSetupError("RunPod cached LTX 2.5 snapshot directory is empty")
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def _link_cached_files() -> None:
    snapshot = _find_ltx_snapshot()
    missing: list[str] = []
    for relative in CACHED_FILES:
        source = snapshot / relative
        target = COMFY_MODELS / relative
        if not source.exists():
            missing.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(source)
        LOGGER.info("Linked cached model %s", relative)
    if missing:
        raise ModelSetupError(
            "The RunPod cached LTX 2.5 snapshot is missing required files: " + ", ".join(missing)
        )


def _ready(target: Path, item: ModelFile) -> bool:
    return target.is_file() and target.stat().st_size >= item.min_bytes


def _headers(item: ModelFile, offset: int = 0) -> dict[str, str]:
    headers = {"User-Agent": "runpod-redgraft-ltx25/2.0"}
    if item.token_env:
        token = os.getenv(item.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return headers


def _download(item: ModelFile) -> None:
    target = COMFY_MODELS / item.relative_path
    if _ready(target, item):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for delay in (0, 10, 20, 40, 80):
        if delay:
            time.sleep(delay)
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(
                item.url,
                headers=_headers(item, offset),
                stream=True,
                timeout=(30, 600),
                allow_redirects=True,
            ) as response:
                if response.status_code in {401, 403}:
                    raise ModelSetupError(
                        f"Access denied while downloading {target.name}. Check {item.token_env or 'provider access'}."
                    )
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                if mode == "wb" and partial.exists():
                    partial.unlink()
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if partial.stat().st_size < item.min_bytes:
                raise ModelSetupError(f"Downloaded {target.name}, but the file is unexpectedly small")
            partial.replace(target)
            LOGGER.info("Downloaded %s", target.name)
            return
        except ModelSetupError:
            raise
        except (OSError, requests.RequestException) as exc:
            LOGGER.warning("Download failed for %s: %s", target.name, exc)
    raise ModelSetupError(f"Could not download required model: {target.name}")


def _check_disk() -> None:
    remaining = 0
    for item in EXTRA_FILES:
        target = COMFY_MODELS / item.relative_path
        if _ready(target, item):
            continue
        partial = target.with_name(target.name + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        remaining += max(0, item.min_bytes - existing)
    free = shutil.disk_usage(COMFY_MODELS).free
    required = remaining + MODEL_DISK_SAFETY_BYTES
    if free < required:
        raise ModelSetupError(
            f"Not enough container disk for REDGraft extras. Need about {required // 1024**3} GiB free; "
            f"only {free // 1024**3} GiB is available. Increase RunPod container disk to at least 45-50 GB."
        )


def ensure_models() -> None:
    COMFY_MODELS.mkdir(parents=True, exist_ok=True)
    _link_cached_files()
    _check_disk()

    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max(1, DOWNLOAD_WORKERS)) as executor:
        futures = {executor.submit(_download, item): item for item in EXTRA_FILES}
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((item.relative_path, exc))

    if failures:
        details = "; ".join(f"{path}: {exc}" for path, exc in failures)
        raise ModelSetupError("REDGraft model preparation failed: " + details)

    LOGGER.info("REDGraft LTX 2.5 model preparation complete")

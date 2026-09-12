"""Prepare the fixed LTX 2.5 model set on a RunPod network volume.

The Docker image stays small so RunPod's 30-minute GitHub build can finish.
The weights persist at /runpod-volume across worker shutdowns.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests


LOGGER = logging.getLogger("ltx25-model-setup")
DEFAULT_MODEL_STORE = Path("/runpod-volume/ltx25-models")
DOWNLOAD_WORKERS = int(os.getenv("MODEL_DOWNLOAD_WORKERS", "3"))
MODEL_DISK_SAFETY_BYTES = int(
    os.getenv("MODEL_DISK_SAFETY_BYTES", str(8 * 1024**3))
)


@dataclass(frozen=True)
class ModelFile:
    url: str
    relative_path: str
    min_bytes: int
    token_env: str | None = None


MODEL_FILES = (
    ModelFile(
        "https://civitai.com/api/download/models/3250230?fileId=3133376",
        "diffusion_models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
        16_000_000_000,
        "CIVITAI_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        15_000_000_000,
        "HF_TOKEN",
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
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        1_400_000_000,
        "HF_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        350_000_000,
        "HF_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        950_000_000,
        "HF_TOKEN",
    ),
)


class ModelSetupError(RuntimeError):
    """Raised when required model files cannot be prepared."""


def model_store() -> Path:
    store = Path(os.getenv("MODEL_STORE", str(DEFAULT_MODEL_STORE)))
    if not store.is_absolute() or store == Path("/") or len(store.parts) < 3:
        raise ModelSetupError(f"Unsafe MODEL_STORE path: {store}")
    return store


def _ready(target: Path, item: ModelFile) -> bool:
    return target.is_file() and target.stat().st_size >= item.min_bytes


def models_ready(store: Path) -> bool:
    return all(_ready(store / item.relative_path, item) for item in MODEL_FILES)


def _download_headers(item: ModelFile, offset: int = 0) -> dict[str, str]:
    headers = {"User-Agent": "runpod-ltx25-i2v-worker/1.0"}
    if item.token_env:
        token = os.getenv(item.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return headers


def _download_file(item: ModelFile, target: Path) -> None:
    if _ready(target, item):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    if partial.is_file() and partial.stat().st_size >= item.min_bytes:
        partial.replace(target)
        return
    backoffs = (0, 10, 20, 40, 80)

    for attempt, delay in enumerate(backoffs, start=1):
        if delay:
            time.sleep(delay)
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(
                item.url,
                headers=_download_headers(item, offset),
                stream=True,
                timeout=(30, 600),
                allow_redirects=True,
            ) as response:
                if response.status_code in {401, 403}:
                    token_hint = item.token_env or "the provider token"
                    raise ModelSetupError(
                        f"Access denied for {target.name}. Check {token_hint} and accept any model license first."
                    )
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if partial.stat().st_size < item.min_bytes:
                raise ModelSetupError(
                    f"Downloaded {target.name}, but the file is unexpectedly small"
                )
            partial.replace(target)
            LOGGER.info("Downloaded %s", target.name)
            return
        except ModelSetupError:
            raise
        except (OSError, requests.RequestException) as exc:
            LOGGER.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                len(backoffs),
                target.name,
                exc,
            )

    raise ModelSetupError(f"Could not download required model: {target.name}")


def ensure_models() -> None:
    if not os.getenv("HF_TOKEN"):
        raise ModelSetupError(
            "HF_TOKEN is required. Accept the Lightricks/LTX-2.5 license on Hugging Face, "
            "create a read token, and add it to the RunPod endpoint environment variables."
        )

    store = model_store()
    store.mkdir(parents=True, exist_ok=True)
    lock_path = store / ".ltx25-download.lock"

    try:
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if models_ready(store):
                return

            remaining_bytes = 0
            for item in MODEL_FILES:
                target = store / item.relative_path
                if _ready(target, item):
                    continue
                partial = target.with_name(f"{target.name}.part")
                existing = partial.stat().st_size if partial.is_file() else 0
                remaining_bytes += max(0, item.min_bytes - existing)

            required_free_bytes = remaining_bytes + MODEL_DISK_SAFETY_BYTES
            free_bytes = shutil.disk_usage(store).free
            if free_bytes < required_free_bytes:
                raise ModelSetupError(
                    "The worker has insufficient container disk for the LTX 2.5 models. "
                    f"At least {required_free_bytes // 1024**3} GiB free is required for the remaining files; "
                    f"only {free_bytes // 1024**3} GiB is available."
                )

            LOGGER.info(
                "Downloading the LTX 2.5 model set into persistent storage at %s. "
                "Later workers will reuse these files.",
                store,
            )
            failures: list[tuple[str, Exception]] = []
            with ThreadPoolExecutor(max_workers=max(1, DOWNLOAD_WORKERS)) as executor:
                futures = {
                    executor.submit(_download_file, item, store / item.relative_path): item
                    for item in MODEL_FILES
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        LOGGER.error("Failed model %s: %s", item.relative_path, exc)
                        failures.append((item.relative_path, exc))

            if failures or not models_ready(store):
                details = "; ".join(f"{path}: {exc}" for path, exc in failures)
                raise ModelSetupError(
                    "One or more LTX 2.5 model downloads failed. " + (details or "Check the worker logs.")
                )
    except OSError as exc:
        raise ModelSetupError(f"Could not access model storage at {store}: {exc}") from exc

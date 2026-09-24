"""Prepare the selected LTX model profile for RunPod.

Preferred production mode: mount one RunPod Cached Model repo containing every
required weight (grottijohm/redgraft-ltx25-runpod). For the one-time bootstrap,
this module can either reuse an official Lightricks/LTX-2.5 cached-model mount or
download only the exact official files required by the workflow. That avoids
RunPod having to initialize the entire upstream LTX-2.5 repository first.
"""
from __future__ import annotations

import fcntl
import logging
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from file_integrity import (IntegrityError, check_deadline, download_model,
                            validate_safetensors, sha256_file)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("ltx25-model-setup")
COMFY_MODELS = Path(os.getenv("COMFY_MODELS", "/comfyui/models"))
CACHE_ROOT = Path(os.getenv("RUNPOD_MODEL_CACHE", "/runpod-volume/huggingface-cache/hub"))
DOWNLOAD_WORKERS = int(os.getenv("MODEL_DOWNLOAD_WORKERS", "3"))
MODEL_DISK_SAFETY_BYTES = int(os.getenv("MODEL_DISK_SAFETY_BYTES", str(8 * 1024**3)))
BUNDLE_REPO = os.getenv("HF_BUNDLE_REPO", "grottijohm/redgraft-ltx25-runpod")
LOCK_PATH = Path(os.getenv("MODEL_SETUP_LOCK", "/tmp/redgraft-model-setup.lock"))
BOOTSTRAP_MODE = os.getenv("BOOTSTRAP_HF_REPO", "0") == "1"
MODEL_PROFILE = os.getenv("MODEL_PROFILE", "redgraft").strip().lower()

OFFICIAL_PATHS = (
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)


@dataclass(frozen=True)
class ModelFile:
    url: str
    relative_path: str
    min_bytes: int
    token_env: str | None = None
    expected_sha256: str | None = None


# These are the exact four upstream LTX-2.5 files used by the ComfyUI workflow.
# The repository is gated, so HF_TOKEN must have accepted/accessed LTX-2.5.
OFFICIAL_DOWNLOAD_FILES = (
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        10_000_000_000,
        "HF_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        100_000_000,
        "HF_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        100_000_000,
        "HF_TOKEN",
    ),
    ModelFile(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        100_000_000,
        "HF_TOKEN",
    ),
)

EXTRA_FILES = (
    ModelFile(
        "https://civitai.com/api/download/models/3250230?fileId=3133376",
        "diffusion_models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
        16_000_000_000,
        "CIVITAI_TOKEN",
    ),
)


REDGRAFT_FILES = OFFICIAL_DOWNLOAD_FILES + EXTRA_FILES

# Pin each MiniMax file to its repository revision and SHA-256.
MINIMAX_FILES = (
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors',
              'diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors', 34038892334, "HF_TOKEN",
              '7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5'),
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
              'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors', 20970379616, "HF_TOKEN",
              '9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779'),
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
              'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 27141342152, "HF_TOKEN",
              'bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923'),
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/vae/minimax_h3_video_vae_fp16.safetensors',
              'vae/minimax_h3_video_vae_fp16.safetensors', 5207808496, "HF_TOKEN",
              '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'),
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/vae/minimax_h3_audio_vae_fp32.safetensors',
              'vae/minimax_h3_audio_vae_fp32.safetensors', 605254808, "HF_TOKEN",
              '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
              'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', 1956193000, "HF_TOKEN",
              '2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e'),
    ModelFile('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors',
              'loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', 1956193000, "HF_TOKEN",
              '5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c'),
    ModelFile('https://huggingface.co/Coconut25/MN/resolve/4b3feeea4514d62e7ec4aa9522becad67522b806/M3_Unlocked_V2.safetensors',
              'loras/M3_Unlocked_V2.safetensors', 172065637, "HF_TOKEN",
              '8138e5ec1c6cc79706f1129311e90dcd04cc0ef708336c494161b09057f34c07'),
)
MODEL_PROFILES = {"redgraft": REDGRAFT_FILES, "minimax": MINIMAX_FILES}

class ModelSetupError(RuntimeError):
    pass


if MODEL_PROFILE not in MODEL_PROFILES:
    raise ModelSetupError("MODEL_PROFILE must be redgraft or minimax; 10eros has been retired. Upload the MiniMax bundle before switching profiles.")
MODEL_FILES = MODEL_PROFILES[MODEL_PROFILE]
ALL_MODEL_PATHS = tuple(item.relative_path for item in MODEL_FILES)
WORKFLOW_FILENAME = "api-workflow-minimax.json" if MODEL_PROFILE == "minimax" else "api-workflow.json"


def _repo_cache_dir(repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return CACHE_ROOT / f"models--{owner}--{name}" / "snapshots"


def _latest_snapshot(repo_id: str) -> Path | None:
    root = _repo_cache_dir(repo_id)
    if not root.is_dir():
        return None
    snapshots = [p for p in root.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def _link_from_snapshot(snapshot: Path, paths: tuple[str, ...], deadline: float | None = None) -> None:
    # Validate the complete set before changing any symlinks.
    items = {item.relative_path: item for item in MODEL_FILES}
    manifest_path = snapshot / "bundle-manifest.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())["files"]
            if not isinstance(manifest, dict):
                raise ValueError("files is not an object")
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelSetupError("Cached bundle has an invalid integrity manifest") from exc
    for relative in paths:
        check_deadline(deadline)
        source = snapshot / relative
        if not source.is_file():
            raise ModelSetupError(f"Cached model snapshot is missing: {relative}")
        try:
            item = items.get(relative)
            if item is None:
                validate_safetensors(source)
            else:
                _validate_model(source, item, deadline)
            if manifest is not None:
                record = manifest.get(relative)
                if (not isinstance(record, dict) or record.get("size") != source.stat().st_size
                        or record.get("sha256") != sha256_file(source, deadline)):
                    raise IntegrityError(f"Cached file does not match bundle manifest: {relative}")
        except (IntegrityError, OSError) as exc:
            raise ModelSetupError(str(exc)) from exc
    for relative in paths:
        source, target = snapshot / relative, COMFY_MODELS / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.is_dir():
            raise ModelSetupError(f"Expected a model file, found a directory: {target}")
        target.unlink(missing_ok=True)
        target.symlink_to(source)
        LOGGER.info("Linked cached model %s", relative)


def _validate_model(target: Path, item: ModelFile, deadline: float | None = None) -> None:
    validate_safetensors(target, item.min_bytes)
    if item.expected_sha256 and sha256_file(target, deadline) != item.expected_sha256:
        raise IntegrityError(f"{target.name} does not match the pinned upstream SHA-256")


def _ready(target: Path, item: ModelFile, deadline: float | None = None) -> bool:
    try:
        _validate_model(target, item, deadline)
        return True
    except (OSError, IntegrityError):
        return False


def _headers(item: ModelFile, offset: int = 0) -> dict[str, str]:
    headers = {"User-Agent": "runpod-redgraft-ltx25/3.2"}
    if item.token_env:
        token = os.getenv(item.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return headers


def _download(item: ModelFile, deadline: float | None = None) -> None:
    check_deadline(deadline)
    target = COMFY_MODELS / item.relative_path
    if _ready(target, item, deadline):
        LOGGER.info("Already ready: %s", target.name)
        return
    # Never write through an existing symlink into a read-only cached snapshot.
    if target.is_symlink():
        target.unlink()
    try:
        download_model(item.url, target, item.min_bytes, _headers(item), deadline)
        _validate_model(target, item, deadline)
    except (IntegrityError, OSError) as exc:
        raise ModelSetupError(str(exc)) from exc


def _check_disk(items: tuple[ModelFile, ...], deadline: float | None = None) -> None:
    remaining = 0
    for item in items:
        target = COMFY_MODELS / item.relative_path
        if _ready(target, item, deadline):
            continue
        partial = target.with_name(target.name + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        remaining += max(0, item.min_bytes - existing)
    free = shutil.disk_usage(COMFY_MODELS).free
    required = remaining + MODEL_DISK_SAFETY_BYTES
    LOGGER.info(
        "Container disk: %.1f GiB free; minimum remaining model space estimate %.1f GiB",
        free / 1024**3,
        required / 1024**3,
    )
    if free < required:
        raise ModelSetupError(
            f"Not enough container disk for model bootstrap. Need at least about {required // 1024**3} GiB free; "
            f"only {free // 1024**3} GiB is available. For direct bootstrap use roughly 100 GB container disk."
        )


def _download_many(items: tuple[ModelFile, ...], deadline: float | None = None) -> None:
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max(1, DOWNLOAD_WORKERS)) as executor:
        futures = {executor.submit(_download, item, deadline): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((item.relative_path, exc))
    if failures:
        details = "; ".join(f"{path}: {exc}" for path, exc in failures)
        raise ModelSetupError("Model preparation failed: " + details)


def _ensure_models_unlocked(deadline: float | None = None) -> None:
    check_deadline(deadline)
    COMFY_MODELS.mkdir(parents=True, exist_ok=True)

    # Production validates the entire selected profile before linking anything.
    # Setup may reuse a partial older bundle and download the new profile's files.
    bundled = _latest_snapshot(BUNDLE_REPO)
    if bundled is not None:
        paths = ALL_MODEL_PATHS
        if BOOTSTRAP_MODE:
            paths = tuple(p for p in paths if (bundled / p).is_file())
        _link_from_snapshot(bundled, paths, deadline)
        if not BOOTSTRAP_MODE:
            LOGGER.info("Using cached bundle %s with profile %s", BUNDLE_REPO, MODEL_PROFILE)
            return

    # Bootstrap path A: if the official cache is already mounted, reuse its four
    # required files and only download the REDGraft checkpoint.
    official = _latest_snapshot("Lightricks/LTX-2.5")
    if official is not None and BOOTSTRAP_MODE and MODEL_PROFILE == "redgraft" and bundled is None:
        LOGGER.info("Using Lightricks/LTX-2.5 cached model as bootstrap source")
        _link_from_snapshot(official, OFFICIAL_PATHS, deadline)
        _check_disk(EXTRA_FILES, deadline)
        _download_many(EXTRA_FILES, deadline)
        LOGGER.info("Bootstrap model preparation complete")
        return

    # Bootstrap path B: no RunPod Cached Model at all. Download only the five
    # exact files used by this worker. This bypasses RunPod's expensive
    # 'initializing model files' stage for the huge upstream LTX repository.
    if BOOTSTRAP_MODE:
        direct_files = MODEL_FILES
        LOGGER.info("Direct bootstrap enabled; downloading only the exact required model files")
        _check_disk(direct_files, deadline)
        _download_many(direct_files, deadline)
        LOGGER.info("Direct bootstrap model preparation complete")
        return

    raise ModelSetupError(
        f"No complete bundle ({BUNDLE_REPO}) is mounted. For one-time setup, set BOOTSTRAP_HF_REPO=1; "
        f"after bootstrap, set RunPod Cached Model to {BUNDLE_REPO}."
    )


def ensure_models(deadline: float | None = None) -> None:
    """Serialize preparation and include lock waiting in the job time budget."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        while True:
            check_deadline(deadline)
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        try:
            _ensure_models_unlocked(deadline)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def model_status() -> dict:
    """Inspect availability without downloading files or loading GPU weights."""
    snapshot = _latest_snapshot(BUNDLE_REPO)
    root = snapshot or COMFY_MODELS
    missing = [p for p in ALL_MODEL_PATHS if not (root / p).is_file()]
    invalid = []
    for item in MODEL_FILES:
        if item.relative_path not in missing and not _ready(root / item.relative_path, item):
            invalid.append(item.relative_path)
    return {
        "model_profile": MODEL_PROFILE,
        "bundle_repo": BUNDLE_REPO,
        "cached_bundle_mounted": snapshot is not None,
        "files_ready": not missing and not invalid,
        "missing_files": missing,
        "invalid_files": invalid,
        "bootstrap_enabled": BOOTSTRAP_MODE,
        "generation_configured": snapshot is not None and not BOOTSTRAP_MODE and not missing and not invalid,
        "validation": "safetensors container structure; inference not tested",
    }

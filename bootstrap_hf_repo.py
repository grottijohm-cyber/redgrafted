"""Prepare the existing model bundle inside a tracked RunPod setup job."""
from __future__ import annotations

import io
import json
import logging
import os
import re

import model_setup
from file_integrity import check_deadline, sha256_file

LOGGER = logging.getLogger("bundle-setup")


def _existing_manifest(repo_info) -> dict:
    """Retain every existing weight's digest without downloading it again."""
    files = {}
    for entry in repo_info.siblings or []:
        if not entry.rfilename.endswith(".safetensors"):
            continue
        lfs = entry.lfs
        digest = getattr(lfs, "sha256", None)
        size = getattr(lfs, "size", None)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or type(size) is not int or size <= 0:
            raise model_setup.ModelSetupError(f"Cannot preserve integrity metadata for existing file: {entry.rfilename}")
        files[entry.rfilename] = {"size": size, "sha256": digest}
    return {"format_version": 1, "files": files}


def bootstrap_bundle(deadline: float | None = None) -> dict:
    if not model_setup.BOOTSTRAP_MODE:
        raise model_setup.ModelSetupError("One-time setup requires BOOTSTRAP_HF_REPO=1")
    token = (os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    repo_id = model_setup.BUNDLE_REPO.strip()
    if not token:
        raise model_setup.ModelSetupError("One-time setup requires HF_WRITE_TOKEN, or an existing HF_TOKEN with write permission")
    if not repo_id or "/" not in repo_id:
        raise model_setup.ModelSetupError("HF_BUNDLE_REPO must be username/repository")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    # Reuse an existing write-capable HF_TOKEN when supplied; a read-only token
    # fails before any large downloads. Fine-grained access is enforced by the Hub.
    identity = api.whoami()
    if identity.get("auth", {}).get("accessToken", {}).get("role") == "read":
        raise model_setup.ModelSetupError("The configured Hugging Face token is read-only; add HF_WRITE_TOKEN with write access to the existing private bundle")
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    repo_info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    if not repo_info.private:
        raise model_setup.ModelSetupError("The model bundle repository must be private")
    manifest = _existing_manifest(repo_info)

    LOGGER.info("Preparing model files for %s", repo_id)
    model_setup.ensure_models(deadline)
    operations = []
    for relative in model_setup.ALL_MODEL_PATHS:
        check_deadline(deadline)
        source = model_setup.COMFY_MODELS / relative
        LOGGER.info("Recording model digest: %s", relative)
        manifest["files"][relative] = {
            "size": source.stat().st_size,
            "sha256": sha256_file(source, deadline),
        }
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(source)))

    # Remove only the requested retired checkpoint, atomically with the new files.
    # Historical Hub revisions remain available for rollback.
    retired = "checkpoints/10Eros_v1.5_DMD_INT8_checkpoint.safetensors"
    if model_setup.MODEL_PROFILE == "minimax" and retired in manifest["files"]:
        from huggingface_hub import CommitOperationDelete
        operations.append(CommitOperationDelete(path_in_repo=retired))
        del manifest["files"][retired]

    manifest["profiles"] = {
        name: [item.relative_path for item in items]
        for name, items in model_setup.MODEL_PROFILES.items()
        if all(item.relative_path in manifest["files"] for item in items)
    }

    readme = (
        "---\nlicense: other\n---\n\n# Private model runtime bundle\n\n"
        "Files used by grottijohm-cyber/redgrafted. Original model licenses and "
        "access conditions continue to apply. The manifest records the uploaded "
        "bytes; it is not an upstream publisher signature.\n\n"
        "Available profiles: " + ", ".join(manifest["profiles"]) + ".\n\n"
        "The minimax profile uses MiniMax H3 FL2VA, its Qwen3-VL text encoder, "
        "matching video/audio VAEs, and two model adapters. Prompts are passed through. "
        "Its upload retires the 10Eros checkpoint from the current revision only; "
        "unrelated weights and historical revisions are retained.\n"
    )
    for name, text in [("README.md", readme), ("bundle-manifest.json", json.dumps(manifest, indent=2))]:
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=io.BytesIO(text.encode())))
    check_deadline(deadline)
    LOGGER.info("Uploading model bundle. This remains part of the active RunPod job.")
    # Publish all required files and the manifest together, without a partial bundle.
    commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                               commit_message=f"Add {model_setup.MODEL_PROFILE} runtime profile",
                               parent_commit=repo_info.sha)
    LOGGER.info("BOOTSTRAP_COMPLETE:%s", repo_id)
    return {
        "status": "setup_complete",
        "bundle_repo": repo_id,
        "revision": commit.oid,
        "model_profile": model_setup.MODEL_PROFILE,
        "cached_model": f"{repo_id}:{commit.oid}",
        "next_step": "Select this exact new revision as RunPod Cached Model, keep MODEL_PROFILE, then remove BOOTSTRAP_HF_REPO and HF_WRITE_TOKEN and redeploy.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(bootstrap_bundle(), indent=2))

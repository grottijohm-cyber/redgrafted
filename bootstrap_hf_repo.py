"""Prepare the existing model bundle inside a tracked RunPod setup job."""
from __future__ import annotations

import io
import json
import logging
import os

import model_setup
from file_integrity import check_deadline, sha256_file

LOGGER = logging.getLogger("bundle-setup")


def bootstrap_bundle(deadline: float | None = None) -> dict:
    if not model_setup.BOOTSTRAP_MODE:
        raise model_setup.ModelSetupError("One-time setup requires BOOTSTRAP_HF_REPO=1")
    token = os.getenv("HF_WRITE_TOKEN", "").strip()
    repo_id = model_setup.BUNDLE_REPO.strip()
    if not token:
        raise model_setup.ModelSetupError("One-time setup requires HF_WRITE_TOKEN")
    if not repo_id or "/" not in repo_id:
        raise model_setup.ModelSetupError("HF_BUNDLE_REPO must be username/repository")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo_id, repo_type="model").private:
        raise model_setup.ModelSetupError("The model bundle repository must be private")

    LOGGER.info("Preparing model files for %s", repo_id)
    model_setup.ensure_models(deadline)
    operations, manifest = [], {"format_version": 1, "files": {}}
    for relative in model_setup.ALL_MODEL_PATHS:
        check_deadline(deadline)
        source = model_setup.COMFY_MODELS / relative
        LOGGER.info("Recording model digest: %s", relative)
        manifest["files"][relative] = {
            "size": source.stat().st_size,
            "sha256": sha256_file(source, deadline),
        }
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(source)))

    readme = (
        "---\nlicense: other\n---\n\n# Private model runtime bundle\n\n"
        "Files used by grottijohm-cyber/redgrafted. Original model licenses and "
        "access conditions continue to apply. The manifest records the uploaded "
        "bytes; it is not an upstream publisher signature.\n"
    )
    for name, text in [("README.md", readme), ("bundle-manifest.json", json.dumps(manifest, indent=2))]:
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=io.BytesIO(text.encode())))
    check_deadline(deadline)
    LOGGER.info("Uploading model bundle. This remains part of the active RunPod job.")
    # Publish all required files and the manifest together, without a partial bundle.
    commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                               commit_message="Publish complete runtime model bundle")
    LOGGER.info("BOOTSTRAP_COMPLETE:%s", repo_id)
    return {
        "status": "setup_complete",
        "bundle_repo": repo_id,
        "revision": commit.oid,
        "next_step": "Set this repository as RunPod Cached Model, then remove BOOTSTRAP_HF_REPO and HF_WRITE_TOKEN and redeploy.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(bootstrap_bundle(), indent=2))

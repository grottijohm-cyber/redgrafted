"""One-time helper to mirror the exact REDGraft worker model set into one HF repo.

Run inside the RunPod worker with:
  BOOTSTRAP_HF_REPO=1
  HF_WRITE_TOKEN=hf_...
  HF_BUNDLE_REPO=grottijohm/redgraft-ltx25-runpod

The worker first prepares/links the required model files, then this script creates
(or reuses) the target Hugging Face model repo and uploads the exact files that
ComfyUI expects. The repo is private by default.
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

from model_setup import ensure_models

TARGETS = (
    ("diffusion_models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors", "diffusion_models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors"),
    ("text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    ("text_encoders/gemma-3-12b-it-heretic-v2_int8.safetensors", "text_encoders/gemma-3-12b-it-heretic-v2_int8.safetensors"),
    ("text_encoders/ltx-2.3_text_projection_bf16.safetensors", "text_encoders/ltx-2.3_text_projection_bf16.safetensors"),
    ("vae/ltx-2.5-video-vae-conv-bf16.safetensors", "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
    ("vae/ltx-2.5-audio-vae-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors"),
    ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
)


def main() -> None:
    token = os.getenv("HF_WRITE_TOKEN", "").strip()
    repo_id = os.getenv("HF_BUNDLE_REPO", "grottijohm/redgraft-ltx25-runpod").strip()
    if not token:
        raise SystemExit("HF_WRITE_TOKEN is required for the one-time bootstrap.")
    if not repo_id or "/" not in repo_id:
        raise SystemExit("HF_BUNDLE_REPO must look like username/repo-name")

    # This creates /comfyui/models entries and reuses RunPod's official LTX cache.
    ensure_models()

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    root = Path("/comfyui/models")
    for relative_source, path_in_repo in TARGETS:
        source = root / relative_source
        if not source.exists():
            raise FileNotFoundError(f"Required model file missing: {source}")
        print(f"Uploading {path_in_repo} ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {Path(path_in_repo).name}",
        )

    readme = f"""---\nlicense: other\n---\n\n# REDGraft LTX 2.5 RunPod bundle\n\nPrivate runtime bundle for the `grottijohm-cyber/redgrafted` RunPod worker.\n\nIt mirrors the exact files required by that worker so RunPod Cached Models can\nmount one repository instead of downloading the REDGraft extras on each fresh\nworker. Review the upstream model licenses/terms before changing visibility or\nredistributing the files.\n"""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add bundle README",
    )
    print(f"BOOTSTRAP_COMPLETE:{repo_id}", flush=True)


if __name__ == "__main__":
    main()

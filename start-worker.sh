#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${RUNPOD_MODEL_CACHE:-/runpod-volume/huggingface-cache/hub}"
BUNDLE_REPO="${HF_BUNDLE_REPO:-grottijohm/redgraft-ltx25-runpod}"
BUNDLE_CACHE="$CACHE_ROOT/models--${BUNDLE_REPO//\//--}/snapshots"
OFFICIAL_CACHE="$CACHE_ROOT/models--Lightricks--LTX-2.5/snapshots"

mkdir -p /comfyui/input /comfyui/output /comfyui/models

if [[ ! -d "$BUNDLE_CACHE" && ! -d "$OFFICIAL_CACHE" ]]; then
    echo "ERROR: No usable RunPod Cached Model was mounted." >&2
    echo "For bootstrap use Lightricks/LTX-2.5. After bootstrap use $BUNDLE_REPO." >&2
    exit 1
fi

echo "Preparing REDGraft LTX 2.5 model files..." >&2
PYTHONPATH=/ python -c 'from model_setup import ensure_models; ensure_models()'

if [[ "${BOOTSTRAP_HF_REPO:-0}" == "1" ]]; then
    echo "One-time Hugging Face bundle bootstrap requested." >&2
    PYTHONPATH=/ python /bootstrap_hf_repo.py
    echo "Hugging Face bundle bootstrap finished. You can now redeploy with Cached Model: $BUNDLE_REPO" >&2
fi

echo "Model preparation complete. Starting ComfyUI + RunPod worker." >&2
exec /start.sh

#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${RUNPOD_MODEL_CACHE:-/runpod-volume/huggingface-cache/hub}"
mkdir -p /comfyui/input /comfyui/output /comfyui/models

if [[ ! -d "$CACHE_ROOT/models--Lightricks--LTX-2.5/snapshots" ]]; then
    echo "ERROR: RunPod Cached Model Lightricks/LTX-2.5 was not mounted." >&2
    echo "Set Cached model to Lightricks/LTX-2.5 and add the Hugging Face access token." >&2
    exit 1
fi

echo "Preparing REDGraft + prompt-enhancer files and linking official cached LTX 2.5 support files..." >&2
PYTHONPATH=/ python -c 'from model_setup import ensure_models; ensure_models()'

echo "Model preparation complete. Starting ComfyUI + RunPod worker." >&2
exec /start.sh

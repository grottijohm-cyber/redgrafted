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

echo "Cached model mount detected. Starting ComfyUI + RunPod worker immediately." >&2

# Do NOT block worker readiness on multi-GB downloads/uploads. If one-time bundle
# bootstrap is requested, run it in the background. model_setup.py uses a
# cross-process lock so an early job and this bootstrap cannot download the same
# files at the same time.
if [[ "${BOOTSTRAP_HF_REPO:-0}" == "1" ]]; then
    echo "Launching one-time Hugging Face bundle bootstrap in background." >&2
    (
        PYTHONUNBUFFERED=1 PYTHONPATH=/ python /bootstrap_hf_repo.py
    ) 2>&1 | sed -u 's/^/[bundle-bootstrap] /' >&2 &
fi

echo "Handing off to base RunPod worker startup." >&2
exec /start.sh

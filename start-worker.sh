#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${RUNPOD_MODEL_CACHE:-/runpod-volume/huggingface-cache/hub}"
BUNDLE_REPO="${HF_BUNDLE_REPO:-grottijohm/redgraft-ltx25-runpod}"
BUNDLE_CACHE="$CACHE_ROOT/models--${BUNDLE_REPO//\//--}/snapshots"
OFFICIAL_CACHE="$CACHE_ROOT/models--Lightricks--LTX-2.5/snapshots"

mkdir -p /comfyui/input /comfyui/output /comfyui/models

if [[ -d "$BUNDLE_CACHE" ]]; then
    echo "Complete REDGraft cached-model bundle detected." >&2
elif [[ -d "$OFFICIAL_CACHE" ]]; then
    echo "Official LTX cached-model bootstrap source detected." >&2
elif [[ "${BOOTSTRAP_HF_REPO:-0}" == "1" ]]; then
    echo "No RunPod Cached Model mounted; direct-download bootstrap mode enabled." >&2
else
    echo "ERROR: No complete REDGraft cached model is mounted." >&2
    echo "For one-time setup, set BOOTSTRAP_HF_REPO=1. After bootstrap, use Cached Model: $BUNDLE_REPO" >&2
    exit 1
fi

echo "Starting ComfyUI + RunPod worker immediately." >&2

# The one-time bootstrap is intentionally backgrounded so RunPod readiness is
# never blocked by large model downloads/uploads. The job handler calls the same
# model setup function and waits on its cross-process lock if a request arrives
# before bootstrap finishes.
if [[ "${BOOTSTRAP_HF_REPO:-0}" == "1" ]]; then
    echo "Launching one-time Hugging Face bundle bootstrap in background." >&2
    (
        PYTHONUNBUFFERED=1 PYTHONPATH=/ python /bootstrap_hf_repo.py
    ) 2>&1 | sed -u 's/^/[bundle-bootstrap] /' >&2 &
fi

echo "Handing off to base RunPod worker startup." >&2
exec /start.sh

#!/usr/bin/env bash
set -euo pipefail

# Prefer RunPod Cached Models when available. This keeps the container small and
# avoids requiring a dedicated network volume just to persist the official LTX
# files. The REDGraft-specific files are prepared by model_setup.py.
CACHE_ROOT="${RUNPOD_MODEL_CACHE:-/runpod-volume/huggingface-cache/hub}"
mkdir -p /comfyui/input /comfyui/output /comfyui/models

if [[ -d "$CACHE_ROOT" ]]; then
    echo "RunPod cached-model storage detected at $CACHE_ROOT" >&2
else
    echo "WARNING: RunPod cached-model storage was not detected yet." >&2
fi

exec /start.sh

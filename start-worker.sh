#!/usr/bin/env bash
set -euo pipefail

mkdir -p /comfyui/input /comfyui/output /comfyui/models

if [[ "${BOOTSTRAP_HF_REPO:-0}" == "1" ]]; then
    echo 'One-time setup mode. Submit a queue job with input: {"action":"setup"}.' >&2
else
    echo "Generation mode. The handler validates the cached model files before each job." >&2
fi

# Preparation runs inside the setup handler so RunPod tracks its lifetime and
# errors. An idle worker could terminate a background uploader during setup.
echo "Starting ComfyUI and the RunPod queue handler." >&2
exec /start.sh

#!/usr/bin/env bash
set -euo pipefail

export MODEL_STORE="${MODEL_STORE:-/runpod-volume/ltx25-models}"

if ! grep -qsE '[[:space:]]/runpod-volume[[:space:]]' /proc/mounts; then
    echo "WARNING: /runpod-volume is not mounted; model downloads will not persist." >&2
fi

mkdir -p "$MODEL_STORE" /comfyui/input /comfyui/output

# Preserve the base image's model directory structure, then point ComfyUI at
# the persistent network-volume directory before ComfyUI starts.
if [[ -d /comfyui/models && ! -L /comfyui/models ]]; then
    cp -a -n /comfyui/models/. "$MODEL_STORE"/
    mv /comfyui/models /comfyui/models-image
fi
ln -sfn "$MODEL_STORE" /comfyui/models

echo "LTX 2.5 models will persist in $MODEL_STORE." >&2

exec /start.sh

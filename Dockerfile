FROM runpod/worker-comfyui:5.10.0-base

# INT8 convrot kernels require the CUDA 13 PyTorch build and compatible host driver.
RUN uv pip install --reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-GIMM-VFI.git /comfyui/custom_nodes/ComfyUI-GIMM-VFI \
    && uv pip install -r /comfyui/custom_nodes/ComfyUI-GIMM-VFI/requirements.txt
RUN mkdir -p /app
COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt
COPY download_embedded_loras.py /app/download_embedded_loras.py
RUN python /app/download_embedded_loras.py

COPY api-workflow.json api-workflow-minimax.json /app/
COPY handler.py model_setup.py worker.py bootstrap_hf_repo.py file_integrity.py video_delivery.py runtime_health.py comfy_launcher.py /
COPY configure_startup.py /app/configure_startup.py
RUN python /app/configure_startup.py
COPY start-worker.sh /start-worker.sh
RUN chmod +x /start-worker.sh

ENV PYTHONUNBUFFERED=1 \
    COMFY_LOG_LEVEL=INFO \
    MODEL_PROFILE=redgraft \
    COMFY_URL=http://127.0.0.1:8188 \
    COMFY_START_TIMEOUT_SECONDS=900 \
    COMFY_MEMORY_PROFILE=conservative \
    COMFY_UNREACHABLE_TIMEOUT_SECONDS=60 \
    JOB_TIMEOUT_SECONDS=6600 \
    MAX_INLINE_OUTPUT_BYTES=6000000 \
    MODEL_DOWNLOAD_WORKERS=3 \
    HF_BUNDLE_REPO=grottijohm/redgraft-ltx25-runpod

CMD ["/start-worker.sh"]

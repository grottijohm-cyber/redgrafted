FROM runpod/worker-comfyui:5.10.0-base

# INT8 convrot kernels require the CUDA 13 PyTorch build and compatible host driver.
RUN uv pip install --reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg cmake build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-GIMM-VFI.git /comfyui/custom_nodes/ComfyUI-GIMM-VFI \
    && uv pip install -r /comfyui/custom_nodes/ComfyUI-GIMM-VFI/requirements.txt
# Re-run ComfyUI's import smoke test after installing GIMM-VFI so a broken custom-node
# dependency fails the image build instead of causing RunPod to roll the deployment back.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# General-purpose 2x Real-ESRGAN model. The same model enhances the uploaded first
# frame before MiniMax conditioning and upscales decoded video frames to 1088x1920.
RUN mkdir -p /comfyui/models/upscale_models \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', '/comfyui/models/upscale_models/RealESRGAN_x2plus.pth')"

# Small CPU-only prompt enhancer. It is deliberately separate from the ComfyUI GPU
# process so prompt rewriting releases its RAM before MiniMax video sampling begins.
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/llama.cpp/build --target llama-cli -j2 \
    && install /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli \
    && rm -rf /tmp/llama.cpp
RUN mkdir -p /app/models/prompt_enhancer \
    && curl -fL --retry 4 --retry-delay 5 \
      -o /app/models/prompt_enhancer/Qwen2.5-3B-Instruct-Uncensored.i1-Q4_K_M.gguf \
      'https://huggingface.co/mradermacher/Qwen2.5-3B-Instruct-Uncensored-i1-GGUF/resolve/main/Qwen2.5-3B-Instruct-Uncensored.i1-Q4_K_M.gguf?download=true'

RUN mkdir -p /app
COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt
COPY download_embedded_loras.py /app/download_embedded_loras.py
RUN python /app/download_embedded_loras.py

COPY api-workflow.json api-workflow-minimax.json /app/
COPY handler.py app_worker.py permanent_storage.py prompt_enhancer.py model_setup.py worker.py bootstrap_hf_repo.py file_integrity.py video_delivery.py runtime_health.py comfy_launcher.py runtime_controls.py comfy_progress.py /
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
    AWS_DEFAULT_REGION=auto \
    PROMPT_ENHANCER_MODEL=/app/models/prompt_enhancer/Qwen2.5-3B-Instruct-Uncensored.i1-Q4_K_M.gguf \
    HF_BUNDLE_REPO=grottijohm/redgraft-ltx25-runpod

CMD ["/start-worker.sh"]

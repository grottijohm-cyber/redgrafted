FROM runpod/worker-comfyui:5.10.0-base

RUN mkdir -p /app
COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

COPY api-workflow.json /app/api-workflow.json
COPY handler.py model_setup.py worker.py bootstrap_hf_repo.py file_integrity.py video_delivery.py runtime_health.py comfy_launcher.py /
COPY configure_startup.py /app/configure_startup.py
RUN python /app/configure_startup.py
COPY start-worker.sh /start-worker.sh
RUN chmod +x /start-worker.sh

ENV PYTHONUNBUFFERED=1 \
    COMFY_LOG_LEVEL=INFO \
    WORKFLOW_PATH=/app/api-workflow.json \
    COMFY_URL=http://127.0.0.1:8188 \
    COMFY_START_TIMEOUT_SECONDS=900 \
    COMFY_MEMORY_PROFILE=conservative \
    COMFY_UNREACHABLE_TIMEOUT_SECONDS=60 \
    JOB_TIMEOUT_SECONDS=6600 \
    MAX_INLINE_OUTPUT_BYTES=6000000 \
    MODEL_DOWNLOAD_WORKERS=3 \
    HF_BUNDLE_REPO=grottijohm/redgraft-ltx25-runpod

CMD ["/start-worker.sh"]

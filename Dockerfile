FROM runpod/worker-comfyui:5.10.0-base

RUN mkdir -p /app
COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

COPY api-workflow.json /app/api-workflow.json
COPY handler.py model_setup.py worker.py bootstrap_hf_repo.py /
COPY start-worker.sh /start-worker.sh
RUN chmod +x /start-worker.sh

ENV PYTHONUNBUFFERED=1 \
    WORKFLOW_PATH=/app/api-workflow.json \
    COMFY_URL=http://127.0.0.1:8188 \
    COMFY_START_TIMEOUT_SECONDS=900 \
    JOB_TIMEOUT_SECONDS=7200 \
    MODEL_DOWNLOAD_WORKERS=3 \
    HF_BUNDLE_REPO=grottijohm/redgraft-ltx25-runpod

CMD ["/start-worker.sh"]

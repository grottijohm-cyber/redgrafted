"""RunPod Queue Serverless entrypoint."""

import runpod

from app_worker import handle_job


def handler(job):
    return handle_job(job)


# Kept at module scope so RunPod's repository scanner and the container runtime
# both recognize the queue worker entrypoint.
runpod.serverless.start({"handler": handler})

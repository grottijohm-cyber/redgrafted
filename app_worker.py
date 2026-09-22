"""RunPod handler wrapper for permanent render archiving and library browsing."""

from __future__ import annotations

import copy
import logging
from typing import Any

from permanent_storage import ArchiveError, archive_generation_result, archive_status, list_renders
from worker import handle_job as base_handle_job


LOGGER = logging.getLogger("redgraft-app-worker")
APP_WORKER_VERSION = "redgraft-library-1"


def _library_action(job_input: dict[str, Any]) -> dict[str, Any]:
    allowed = {"action", "cursor", "max_keys"}
    if set(job_input) - allowed:
        return {"error": "Library requests support only action, cursor, and max_keys"}
    cursor = job_input.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        return {"error": "input.cursor must be a string"}
    max_keys = job_input.get("max_keys", 300)
    if isinstance(max_keys, bool):
        return {"error": "input.max_keys must be an integer from 20 to 1000"}
    try:
        max_keys = int(max_keys)
    except (TypeError, ValueError):
        return {"error": "input.max_keys must be an integer from 20 to 1000"}
    if max_keys < 20 or max_keys > 1000:
        return {"error": "input.max_keys must be an integer from 20 to 1000"}
    try:
        library = list_renders(cursor=cursor, max_keys=max_keys)
        return {"status": "library", "app_worker_version": APP_WORKER_VERSION, "archive": library}
    except ArchiveError as exc:
        return {"error": str(exc), "archive": {"configured": False}}


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return base_handle_job(job)

    action = job_input.get("action")
    if action == "library":
        return _library_action(job_input)

    if action == "status":
        result = base_handle_job(job)
        if isinstance(result, dict) and "error" not in result:
            try:
                result["archive"] = archive_status()
            except ArchiveError as exc:
                result["archive"] = {"configured": False, "error": str(exc)}
            result["app_worker_version"] = APP_WORKER_VERSION
        return result

    # preset_name belongs to the phone app, not the ComfyUI workflow. Remove it
    # before passing the request to the strict base worker, then preserve it in
    # permanent render metadata.
    preset_name = "Custom"
    forwarded_job = job
    if "preset_name" in job_input:
        value = job_input.get("preset_name")
        if not isinstance(value, str) or len(value.strip()) > 60:
            return {"error": "input.preset_name must be a string up to 60 characters"}
        preset_name = value.strip() or "Custom"
        forwarded_job = copy.deepcopy(job)
        forwarded_job["input"].pop("preset_name", None)

    result = base_handle_job(forwarded_job)
    if not isinstance(result, dict):
        return result
    result["app_worker_version"] = APP_WORKER_VERSION
    if result.get("status") != "success":
        return result

    original_prompt = str(job_input.get("prompt") or "")
    request_settings = {
        key: value
        for key, value in job_input.items()
        if key not in {"image", "prompt", "preset_name"}
    }
    try:
        archived = archive_generation_result(
            job_id=job.get("id"),
            prompt=original_prompt,
            preset_name=preset_name,
            request_settings=request_settings,
            result=result,
        )
        if archived is None:
            result["archive"] = {
                "configured": False,
                "permanent": False,
                "message": "Permanent object storage is not configured on this RunPod endpoint.",
            }
        else:
            result["archive"] = archived
            # Replace possibly short-lived provider URLs with our own fresh signed
            # private URLs. The underlying bucket objects remain permanent.
            result["videos"] = archived["videos"]
    except ArchiveError as exc:
        LOGGER.exception("Permanent archive metadata failed")
        result["archive"] = {
            "configured": True,
            "permanent": False,
            "error": str(exc),
        }
    return result

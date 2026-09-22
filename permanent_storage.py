"""Permanent private render archive backed by S3-compatible object storage.

This module is intentionally independent from the ComfyUI workflow. The existing
worker uploads finished videos through RunPod's bucket helper. We store a small
metadata object beside those uploaded videos and regenerate short-lived signed
URLs whenever the phone client asks for the library.

Cloudflare R2 works with the same BUCKET_* environment variables already used by
the worker:

BUCKET_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
BUCKET_ACCESS_KEY_ID=...
BUCKET_SECRET_ACCESS_KEY=...
BUCKET_NAME=...

No bucket credentials are ever returned to the browser.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


class ArchiveError(RuntimeError):
    """Permanent archive operation failed."""


_REQUIRED_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)


def _configuration() -> dict[str, str]:
    values = {key: (os.getenv(key) or "").strip() for key in _REQUIRED_KEYS}
    if any(values.values()) and not all(values.values()):
        missing = ", ".join(key for key, value in values.items() if not value)
        raise ArchiveError(f"Permanent storage configuration is incomplete; missing: {missing}")
    return values


def archive_status() -> dict[str, Any]:
    values = _configuration()
    return {
        "configured": bool(values["BUCKET_NAME"]),
        "provider": "s3-compatible",
        "private": True,
        "bucket": values["BUCKET_NAME"] or None,
        "prefix": _metadata_prefix(),
        "signed_url_seconds": _signed_url_seconds(),
    }


def configured() -> bool:
    return bool(_configuration()["BUCKET_NAME"])


def _metadata_prefix() -> str:
    value = (os.getenv("PERMANENT_ARCHIVE_PREFIX") or "redgraft/renders").strip().strip("/")
    return value or "redgraft/renders"


def _signed_url_seconds() -> int:
    try:
        value = int(os.getenv("PERMANENT_URL_SECONDS", "604800"))
    except ValueError:
        value = 604800
    return max(300, min(604800, value))


def _client():
    values = _configuration()
    if not values["BUCKET_NAME"]:
        raise ArchiveError("Permanent render storage is not configured")
    return boto3.client(
        "s3",
        endpoint_url=values["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=values["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=values["BUCKET_SECRET_ACCESS_KEY"],
        region_name=(os.getenv("BUCKET_REGION") or "auto"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}),
    ), values["BUCKET_NAME"]


def _safe_job_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]", "-", str(value or "job"))[:64]
    return text or "job"


def _signed_video(client, bucket: str, key: str, filename: str) -> dict[str, Any]:
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=_signed_url_seconds(),
        )
    except (BotoCoreError, ClientError) as exc:
        raise ArchiveError(f"Could not create a private render URL: {exc}") from exc
    return {
        "filename": filename,
        "type": "url",
        "url": url,
        "mime_type": "video/mp4",
        "compressed_for_delivery": False,
        "archive_key": key,
        "permanent": True,
    }


def archive_generation_result(
    *,
    job_id: Any,
    prompt: str,
    preset_name: str,
    request_settings: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Write permanent metadata for a successful generation already uploaded to the bucket.

    The base worker uploads each output with ``prefix=<safe_job_id>``. Therefore
    the permanent object key is deterministic: ``<safe_job_id>/<filename>``.
    """
    if not configured():
        return None

    client, bucket = _client()
    safe_job_id = _safe_job_id(job_id)
    videos = result.get("videos") if isinstance(result, dict) else None
    if not isinstance(videos, list) or not videos:
        raise ArchiveError("Generation succeeded but returned no videos to archive")

    archived_videos: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    for item in videos:
        if not isinstance(item, dict) or item.get("type") != "url":
            raise ArchiveError("Permanent storage is configured but the worker did not return bucket URLs")
        filename = str(item.get("filename") or "video.mp4")
        key = f"{safe_job_id}/{filename}"
        video_records.append({"key": key, "filename": filename, "mime_type": item.get("mime_type") or "video/mp4"})
        archived_videos.append(_signed_video(client, bucket, key, filename))

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    render_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"
    metadata_key = f"{_metadata_prefix()}/{render_id}/metadata.json"
    metadata: dict[str, Any] = {
        "archive_version": 1,
        "render_id": render_id,
        "created_at": now,
        "job_id": str(job_id or ""),
        "prompt": prompt,
        "preset_name": preset_name or "Custom",
        "request_settings": request_settings,
        "seed": result.get("seed"),
        "prompt_id": result.get("prompt_id"),
        "worker_version": result.get("worker_version"),
        "workflow": result.get("workflow"),
        "videos": video_records,
    }
    try:
        client.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store",
        )
    except (BotoCoreError, ClientError) as exc:
        raise ArchiveError(f"Could not save permanent render metadata: {exc}") from exc

    return {
        **metadata,
        "metadata_key": metadata_key,
        "videos": archived_videos,
        "permanent": True,
    }


def list_renders(cursor: str | None = None, max_keys: int = 300) -> dict[str, Any]:
    """Return one private library page with fresh signed video URLs."""
    status = archive_status()
    if not status["configured"]:
        return {**status, "renders": [], "cursor": None}

    client, bucket = _client()
    max_keys = max(20, min(1000, int(max_keys)))
    params: dict[str, Any] = {
        "Bucket": bucket,
        "Prefix": f"{_metadata_prefix()}/",
        "MaxKeys": max_keys,
    }
    if cursor:
        params["ContinuationToken"] = cursor
    try:
        page = client.list_objects_v2(**params)
    except (BotoCoreError, ClientError) as exc:
        raise ArchiveError(f"Could not list permanent renders: {exc}") from exc

    metadata_objects = [
        item for item in page.get("Contents", [])
        if isinstance(item, dict) and str(item.get("Key", "")).endswith("/metadata.json")
    ]
    metadata_objects.sort(key=lambda item: item.get("LastModified") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    renders: list[dict[str, Any]] = []
    for item in metadata_objects:
        key = str(item["Key"])
        try:
            response = client.get_object(Bucket=bucket, Key=key)
            metadata = json.loads(response["Body"].read().decode("utf-8"))
        except (BotoCoreError, ClientError, UnicodeDecodeError, json.JSONDecodeError, KeyError):
            # A single damaged metadata object should not hide the rest of the library.
            continue
        videos = []
        for video in metadata.get("videos", []):
            if not isinstance(video, dict) or not video.get("key"):
                continue
            videos.append(_signed_video(
                client,
                bucket,
                str(video["key"]),
                str(video.get("filename") or "video.mp4"),
            ))
        metadata["videos"] = videos
        metadata["metadata_key"] = key
        metadata["permanent"] = True
        renders.append(metadata)

    return {
        **status,
        "renders": renders,
        "cursor": page.get("NextContinuationToken"),
        "is_truncated": bool(page.get("IsTruncated")),
    }

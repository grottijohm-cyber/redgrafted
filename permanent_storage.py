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
from pathlib import Path
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
_RENDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


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


def _max_extension_source_bytes() -> int:
    try:
        value = int(os.getenv("EXTEND_SOURCE_MAX_BYTES", str(1024 * 1024 * 1024)))
    except ValueError:
        value = 1024 * 1024 * 1024
    return max(10 * 1024 * 1024, min(5 * 1024 * 1024 * 1024, value))


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


def _safe_filename(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", str(value or "video.mp4"))[:180]
    return name or "video.mp4"


def _metadata_key(render_id: str) -> str:
    if not isinstance(render_id, str) or not _RENDER_ID_RE.fullmatch(render_id):
        raise ArchiveError("Invalid permanent render ID")
    return f"{_metadata_prefix()}/{render_id}/metadata.json"


def _read_metadata(client, bucket: str, render_id: str) -> dict[str, Any]:
    key = _metadata_key(render_id)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            raise ArchiveError("The permanent render could not be found") from exc
        raise ArchiveError(f"Could not read permanent render metadata: {exc}") from exc
    except (BotoCoreError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise ArchiveError(f"Could not read permanent render metadata: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("render_id") != render_id:
        raise ArchiveError("Permanent render metadata is invalid")
    return metadata


def _signed_video(client, bucket: str, key: str, filename: str) -> dict[str, Any]:
    filename = _safe_filename(filename)
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=_signed_url_seconds(),
        )
        download_url = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ResponseContentType": "video/mp4",
                "ResponseContentDisposition": f'attachment; filename="{filename}"',
            },
            ExpiresIn=_signed_url_seconds(),
        )
    except (BotoCoreError, ClientError) as exc:
        raise ArchiveError(f"Could not create a private render URL: {exc}") from exc
    return {
        "filename": filename,
        "type": "url",
        "url": url,
        "download_url": download_url,
        "mime_type": "video/mp4",
        "compressed_for_delivery": False,
        "archive_key": key,
        "permanent": True,
    }


def _download_key(client, bucket: str, key: str, destination: Path, *, max_bytes: int | None = None) -> None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        size = int(response.get("ContentLength") or 0)
        if max_bytes is not None and size > max_bytes:
            raise ArchiveError("The archived video is too large to process on this worker")
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        body = response["Body"]
        with destination.open("wb") as output:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise ArchiveError("The archived video is too large to process on this worker")
                output.write(chunk)
    except ArchiveError:
        destination.unlink(missing_ok=True)
        raise
    except (BotoCoreError, ClientError, OSError, KeyError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise ArchiveError(f"Could not download archived video: {exc}") from exc
    if not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise ArchiveError("The archived video downloaded as an empty file")


def archive_generation_result(
    *,
    job_id: Any,
    prompt: str,
    preset_name: str,
    request_settings: dict[str, Any],
    result: dict[str, Any],
    prompt_metadata: dict[str, Any] | None = None,
    extension_metadata: dict[str, Any] | None = None,
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
        filename = _safe_filename(item.get("filename") or "video.mp4")
        key = f"{safe_job_id}/{filename}"
        video_records.append({"key": key, "filename": filename, "mime_type": item.get("mime_type") or "video/mp4"})
        archived_videos.append(_signed_video(client, bucket, key, filename))

    pm = prompt_metadata if isinstance(prompt_metadata, dict) else {}
    used_prompt = str(pm.get("used_prompt") or prompt or "")
    original_prompt = str(pm.get("original_prompt") or used_prompt)
    enhanced_prompt = pm.get("enhanced_prompt")
    if enhanced_prompt is not None:
        enhanced_prompt = str(enhanced_prompt)
    enhancement = {
        "enabled": bool(pm.get("prompt_enhancement_enabled", False)),
        "mode": str(pm.get("prompt_enhancement_mode") or "detailed"),
        "previewed": bool(pm.get("prompt_enhancement_previewed", False)),
    }
    ext = extension_metadata if isinstance(extension_metadata, dict) else None
    if ext is None and isinstance(result.get("extension"), dict):
        ext = result.get("extension")

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    render_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"
    metadata_key = f"{_metadata_prefix()}/{render_id}/metadata.json"
    metadata: dict[str, Any] = {
        "archive_version": 2,
        "render_id": render_id,
        "created_at": now,
        "job_id": str(job_id or ""),
        "prompt": used_prompt,
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "used_prompt": used_prompt,
        "prompt_enhancement": enhancement,
        "preset_name": preset_name or "Custom",
        "request_settings": request_settings,
        "seed": result.get("seed"),
        "prompt_id": result.get("prompt_id"),
        "worker_version": result.get("worker_version"),
        "workflow": result.get("workflow"),
        "extension": ext,
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


def download_render_video(
    render_id: str,
    destination: Path,
    video_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download one archived render video to a worker-local temporary path."""
    client, bucket = _client()
    metadata = _read_metadata(client, bucket, render_id)
    videos = metadata.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ArchiveError("The permanent render has no video to extend")
    if isinstance(video_index, bool) or not isinstance(video_index, int) or video_index < 0 or video_index >= len(videos):
        raise ArchiveError("The requested archived video does not exist")
    record = videos[video_index]
    if not isinstance(record, dict) or not record.get("key"):
        raise ArchiveError("The archived video record is invalid")
    _download_key(client, bucket, str(record["key"]), destination, max_bytes=_max_extension_source_bytes())
    return metadata, record


def download_job_result_video(job_id: Any, result: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Download the first video produced by the current generation from the bucket."""
    videos = result.get("videos") if isinstance(result, dict) else None
    if not isinstance(videos, list) or not videos or not isinstance(videos[0], dict):
        raise ArchiveError("Continuation generation returned no downloadable video")
    filename = _safe_filename(videos[0].get("filename") or "video.mp4")
    key = f"{_safe_job_id(job_id)}/{filename}"
    client, bucket = _client()
    _download_key(client, bucket, key, destination, max_bytes=_max_extension_source_bytes())
    return {"key": key, "filename": filename, "mime_type": "video/mp4"}


def upload_job_video(job_id: Any, source: Path, filename: str) -> dict[str, Any]:
    """Upload a worker-created MP4 under the current job prefix and return signed URLs."""
    if not source.is_file() or source.stat().st_size == 0:
        raise ArchiveError("Combined extension video is empty")
    filename = _safe_filename(filename)
    key = f"{_safe_job_id(job_id)}/{filename}"
    client, bucket = _client()
    try:
        with source.open("rb") as body:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="video/mp4",
                CacheControl="private, max-age=0, no-store",
            )
    except (BotoCoreError, ClientError, OSError) as exc:
        raise ArchiveError(f"Could not upload combined extension video: {exc}") from exc
    return _signed_video(client, bucket, key, filename)


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
    except ClientError as exc:
        # Some S3-compatible endpoints reject ListObjectsV2 with NoSuchKey
        # even though the prefix itself need not exist as an object. Retry
        # with the older listing API, which is also supported by S3 and R2.
        if exc.response.get("Error", {}).get("Code") != "NoSuchKey":
            raise ArchiveError(f"Could not list permanent renders: {exc}") from exc
        legacy_params = {"Bucket": bucket, "Prefix": params["Prefix"], "MaxKeys": max_keys}
        if cursor:
            legacy_params["Marker"] = cursor
        try:
            page = client.list_objects(**legacy_params)
        except (BotoCoreError, ClientError) as fallback_exc:
            raise ArchiveError(f"Could not list permanent renders: {fallback_exc}") from fallback_exc
        page["NextContinuationToken"] = page.get("NextMarker") or (
            page["Contents"][-1]["Key"] if page.get("IsTruncated") and page.get("Contents") else None
        )
    except BotoCoreError as exc:
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

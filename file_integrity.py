"""Validate model containers and resume HTTP downloads without joining bad ranges."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import time
from pathlib import Path

import requests

LOGGER = logging.getLogger("model-download")
_VALIDATED: dict[str, tuple[int, int, int]] = {}
_DIGESTS: dict[tuple, str] = {}


class IntegrityError(RuntimeError):
    pass


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("The job's preparation/generation time budget was exceeded")


def validate_safetensors(path: Path, min_bytes: int = 1) -> None:
    """Check the header and declared tensor extents, not just a size threshold.

    This detects incomplete containers. It does not authenticate the publisher
    or replace an upstream cryptographic checksum.
    """
    stat = path.stat()
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    key = str(path.resolve())
    if stat.st_size < min_bytes:
        raise IntegrityError(f"{path.name} is smaller than the expected minimum")
    if _VALIDATED.get(key) == signature:
        return
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing header length")
            header_length = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_length <= min(100_000_000, stat.st_size - 8):
                raise ValueError("invalid header length")
            header = json.loads(stream.read(header_length))
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        extents = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int for v in offsets)
                    or offsets[0] < 0 or offsets[1] < offsets[0]):
                raise ValueError("invalid tensor offsets")
            extents.append(tuple(offsets))
        if not extents:
            raise ValueError("no tensors")
        cursor = 0
        for start, end in sorted(extents):
            if start != cursor:
                raise ValueError("overlapping tensors or a gap in tensor data")
            cursor = end
        if 8 + header_length + cursor != stat.st_size:
            raise ValueError("file size does not match declared tensor data")
    except (ValueError, TypeError, UnicodeError, struct.error) as exc:
        raise IntegrityError(f"{path.name} is not a complete safetensors file: {exc}") from exc
    _VALIDATED[key] = signature


def sha256_file(path: Path, deadline: float | None = None) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key in _DIGESTS:
        return _DIGESTS[key]
    digest = hashlib.sha256()
    last_log = time.monotonic()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            check_deadline(deadline)
            digest.update(chunk)
            if time.monotonic() - last_log >= 20:
                LOGGER.info("Checking file digest: %s", path.name)
                last_log = time.monotonic()
    result = digest.hexdigest()
    _DIGESTS[key] = result
    return result


def download_model(url: str, target: Path, min_bytes: int,
                   headers: dict[str, str], deadline: float | None = None) -> None:
    partial = target.with_name(target.name + ".part")
    etag_file = partial.with_name(partial.name + ".etag")
    target.parent.mkdir(parents=True, exist_ok=True)
    for delay in (0, 2, 5, 10, 20):
        check_deadline(deadline)
        if delay:
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise TimeoutError("Model download exhausted the job time budget")
            time.sleep(delay)
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = {**headers, "Accept-Encoding": "identity"}
        previous_etag = etag_file.read_text() if etag_file.exists() else ""
        # An interrupted legacy download has no validator. Start afresh rather
        # than combine bytes from two revisions of a mutable source URL.
        if offset and not previous_etag:
            partial.unlink()
            offset = 0
        if offset:
            request_headers.update({"Range": f"bytes={offset}-", "If-Range": previous_etag})
        try:
            LOGGER.info("Downloading %s (resume %.2f GiB)", target.name, offset / 1024**3)
            with requests.get(url, headers=request_headers, stream=True,
                              timeout=(30, 60), allow_redirects=True) as response:
                if response.status_code in {401, 403}:
                    raise PermissionError(f"Access denied for {target.name}; check provider token and model access")
                if response.status_code == 416:
                    match = re.fullmatch(r"bytes \*/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise IntegrityError("Remote range does not match the partial file")
                    validate_safetensors(partial, min_bytes)
                    partial.replace(target)
                    etag_file.unlink(missing_ok=True)
                    return
                response.raise_for_status()
                if response.status_code not in {200, 206}:
                    raise IntegrityError("Download did not return file content")
                total = None
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (not match or int(match[1]) != offset
                            or int(match[2]) < offset or int(match[2]) + 1 != int(match[3])):
                        raise IntegrityError("Unexpected Content-Range; refusing to append")
                    total = int(match[3])
                    new_etag = response.headers.get("ETag", "")
                    if previous_etag and new_etag and previous_etag != new_etag:
                        raise IntegrityError("The remote file changed during download")
                    mode = "ab" if offset else "wb"
                else:
                    mode, offset = "wb", 0
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        expected = offset + int(length)
                    except ValueError as exc:
                        raise IntegrityError("Invalid Content-Length") from exc
                    if expected < offset or (total is not None and expected != total):
                        raise IntegrityError("Inconsistent download length")
                    total = expected
                etag = response.headers.get("ETag", "")
                if etag and not etag.startswith("W/"):
                    etag_file.write_text(etag)
                else:
                    etag_file.unlink(missing_ok=True)
                written, last_log = offset, time.monotonic()
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        check_deadline(deadline)
                        if not chunk:
                            continue
                        stream.write(chunk)
                        written += len(chunk)
                        if total is not None and written > total:
                            raise IntegrityError("Received more data than the declared file size")
                        if time.monotonic() - last_log >= 20:
                            LOGGER.info("%s: %.2f GiB downloaded", target.name, written / 1024**3)
                            last_log = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if total is not None and written != total:
                    raise IntegrityError("Download ended before the declared file size")
            validate_safetensors(partial, min_bytes)
            partial.replace(target)
            etag_file.unlink(missing_ok=True)
            return
        except IntegrityError as exc:
            LOGGER.warning("Restarting %s: %s", target.name, exc)
            partial.unlink(missing_ok=True)
            etag_file.unlink(missing_ok=True)
        except TimeoutError:
            raise
        except (OSError, requests.RequestException) as exc:
            if isinstance(exc, PermissionError):
                raise
            # Do not log signed redirect URLs or provider credentials.
            LOGGER.warning("Download interrupted for %s (%s)", target.name, type(exc).__name__)
    raise IntegrityError(f"Could not obtain a complete model file: {target.name}")

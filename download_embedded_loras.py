from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from urllib.request import Request, urlopen

FILES = (
    (
        "https://civarchive.com/api/download/models/3345855",
        "M3_Unlocked_V2.1.safetensors",
        "52cb409bd89bb5e9a69851c03a1197b3a5b5698c745b812e2f9813492cde7167",
    ),
    (
        "https://civarchive.com/api/download/models/3266628",
        "MysticXXX_MMH3-V4.safetensors",
        "fc3e856d14c6c19557c888f48662d591e4794e281233ec0d987be5003068afba",
    ),
    (
        "https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMNSFW-AIO-V2.5.safetensors",
        "HMNSFW-AIO-V2.5.safetensors",
        "a07732a84fd733085eb5d910f602f918fa7a3658117116927e4329f5951a9d2d",
    ),
    (
        "https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/vagassist_e40.safetensors",
        "vagassist_e40.safetensors",
        "2c2fdb66bf558de1aabda504a81d4ada5f4cebc20e8f519dc6ed3bb6d4be8c9a",
    ),
    (
        "https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/hmpussy_v6_epoch30.safetensors",
        "hmpussy_v6_epoch30.safetensors",
        "3080f4fbcbba4fc06bd09240c7eedb6a5128eb0e19feb001cdf97a7a0941a6ee",
    ),
    (
        "https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMCumshot_V1.0.safetensors",
        "HMCumshot_V1.0.safetensors",
        "634c39cfcbfd9421a2d7b5adc62573fc232384c4c75e003c2e11d3408ad0765c",
    ),
)

TARGET = Path(os.getenv("COMFY_MODELS", "/comfyui/models")) / "loras"
TARGET.mkdir(parents=True, exist_ok=True)

def validate_safetensors_header(path: Path) -> None:
    """Reject HTML/error pages and malformed files before baking them into the worker."""
    size = path.stat().st_size
    if size < 16:
        raise RuntimeError(f"{path.name} is too small to be a safetensors file")
    with path.open("rb") as handle:
        raw = handle.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        if header_size <= 1 or header_size > min(size - 8, 100_000_000):
            raise RuntimeError(f"{path.name} has an invalid safetensors header length: {header_size}")
        header = handle.read(header_size)
    try:
        metadata = json.loads(header.decode("utf-8").rstrip(" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path.name} does not contain a valid safetensors JSON header") from exc
    if not isinstance(metadata, dict) or not metadata:
        raise RuntimeError(f"{path.name} has an empty safetensors header")

    # Validate every tensor points at bytes that actually exist. A valid JSON
    # header alone is not enough: truncated/corrupt downloads can still pass it.
    data_size = size - 8 - header_size
    tensors = 0
    max_end = 0
    for key, record in metadata.items():
        if key == "__metadata__":
            continue
        if not isinstance(record, dict):
            raise RuntimeError(f"{path.name} has malformed tensor metadata for {key}")
        offsets = record.get("data_offsets")
        shape = record.get("shape")
        dtype = record.get("dtype")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(v, int) for v in offsets)
                or offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > data_size):
            raise RuntimeError(f"{path.name} has invalid tensor offsets for {key}")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise RuntimeError(f"{path.name} has malformed tensor shape/dtype for {key}")
        tensors += 1
        max_end = max(max_end, offsets[1])
    if tensors < 2:
        raise RuntimeError(f"{path.name} does not contain a usable LoRA tensor set")
    if max_end != data_size:
        raise RuntimeError(
            f"{path.name} payload size does not match safetensors tensor offsets "
            f"({max_end} != {data_size})"
        )



for url, name, expected in FILES:
    path = TARGET / name
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 redgrafted-builder/1.0"})
    with urlopen(request, timeout=180) as response, path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    try:
        validate_safetensors_header(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if expected is not None and actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    print(f"Ready: {name} source_sha256={actual}")

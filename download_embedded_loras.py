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
    (
        "https://huggingface.co/gravedigga/loras/resolve/main/PlagueKind-tiddies-realismslider.safetensors?download=true",
        "PlagueKind-tiddies-realismslider.safetensors",
        "e5c8c275af58663a664ad2922cc10a248bff70b941043375d2c82d9cc55b7030",
    ),
    (
        "https://huggingface.co/buckets/aronzhan/H3-Loras-bucket/resolve/deepthroat_v02.safetensors?download=true",
        "deepthroat_v02.safetensors",
        "1fd239662f6290255b0bb3a220764fb53aab2859378f7fd3024030c1e1991cb2",
    ),
    (
        "https://civarchive.com/api/download/models/3210503",
        "Missionary_MiniMaxH3.safetensors",
        None,
    ),
    (
        "https://civarchive.com/api/download/models/3320641",
        "MMH3_Synth_Pussy.safetensors",
        None,
    ),
    (
        "https://huggingface.co/nyxia/H3-Loras/resolve/main/Pussy4nus_Epoch80.safetensors?download=true",
        "Pussy4nus_Epoch80.safetensors",
        "ebb9339144845b5516aead2f0fddc6ea6a3567e56ddd74953a307c83e7060d89",
    ),
    (
        "https://huggingface.co/buckets/aronzhan/H3-Loras-bucket/resolve/MinimaxH3-Fingering_000002000.safetensors?download=true",
        "MinimaxH3-Fingering_000002000.safetensors",
        "e758e831ff85aeb4c58f3db1b17ed8d0cc9ef8a778ad910efabb3e6e7513b4eb",
    ),
    (
        "https://huggingface.co/dagloop5/LoRA/resolve/main/moawxx_000002000.safetensors?download=true",
        "moawxx_000002000.safetensors",
        "bc0841e216198174ff5937e3ba2f4c9234c163082276cd9f4e5f8889ae12e4e5",
    ),
    (
        "https://huggingface.co/SexGod1979/NaughtyTimes-MiniMax-H3/resolve/main/SexGod_NaughtyTimes_v3_rank64_pruned_NOADALN.safetensors?download=true",
        "SexGod_NaughtyTimes_v3_rank64_pruned_NOADALN.safetensors",
        "22466f81d4dc6a990e810aa2a57edf579015acb8ffc15e3f562ec21e82f9d0dd",
    ),
)

TARGET = Path(os.getenv("COMFY_MODELS", "/comfyui/models")) / "loras"
TARGET.mkdir(parents=True, exist_ok=True)

NEW_H3_LORAS = {
    "PlagueKind-tiddies-realismslider.safetensors",
    "deepthroat_v02.safetensors",
    "Missionary_MiniMaxH3.safetensors",
    "MMH3_Synth_Pussy.safetensors",
    "Pussy4nus_Epoch80.safetensors",
    "MinimaxH3-Fingering_000002000.safetensors",
    "moawxx_000002000.safetensors",
    "SexGod_NaughtyTimes_v3_rank64_pruned_NOADALN.safetensors",
}


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



def normalize_and_validate_h3_lora(path: Path) -> None:
    """Normalize Kohya/hybrid H3 exports to ComfyUI A/B keys and require usable pairs."""
    from safetensors.torch import load_file, save_file

    raw = load_file(str(path), device="cpu")
    keys = set(raw)
    has_kohya = any(
        key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight")
        or key.startswith("lora_unet_")
        for key in keys
    )
    if has_kohya:
        alphas: dict[str, float] = {}
        for key, value in raw.items():
            if not key.endswith(".alpha"):
                continue
            base = key[:-len(".alpha")]
            down = raw.get(f"{base}.lora_down.weight")
            rank = int(down.shape[0]) if down is not None and down.ndim else float(value)
            alphas[base] = float(value) / float(rank)

        converted = {}
        for key, value in raw.items():
            if key.endswith(".lora_down.weight"):
                base = key[:-len(".lora_down.weight")]
                suffix = "lora_A.weight"
            elif key.endswith(".lora_up.weight"):
                base = key[:-len(".lora_up.weight")]
                suffix = "lora_B.weight"
            else:
                continue

            if base.startswith("diffusion_model."):
                comfy_base = base
            elif base.startswith("transformer."):
                comfy_base = "diffusion_model." + base[len("transformer."):]
            elif base.startswith("lora_unet_"):
                parts = base.split("_")
                if len(parts) < 6 or parts[2] != "blocks":
                    continue
                block, layer = parts[3], parts[4]
                target = "_".join(parts[5:])
                comfy_base = f"diffusion_model.blocks.{block}.{layer}.{target}"
            else:
                continue

            if suffix == "lora_B.weight" and base in alphas:
                value = value * alphas[base]
            converted[f"{comfy_base}.{suffix}"] = value.contiguous()

        if not converted:
            raise RuntimeError(f"{path.name} uses a Kohya/hybrid format but converted to no H3 LoRA tensors")
        temp = path.with_suffix(".normalized.safetensors")
        save_file(converted, str(temp))
        temp.replace(path)
        raw = converted
        keys = set(raw)
        print(f"Normalized H3 LoRA keys: {path.name}")

    a = {key[:-len(".lora_A.weight")] for key in keys if key.endswith(".lora_A.weight")}
    b = {key[:-len(".lora_B.weight")] for key in keys if key.endswith(".lora_B.weight")}
    pairs = a & b
    if not pairs:
        raise RuntimeError(
            f"{path.name} has no matched ComfyUI lora_A/lora_B tensor pairs; "
            "refusing to bake a file the H3 LoRA loader cannot use"
        )
    if not any(base.startswith("diffusion_model.") for base in pairs):
        raise RuntimeError(
            f"{path.name} has LoRA pairs but none target diffusion_model.* H3 weights"
        )
    print(f"H3 LoRA compatibility: {path.name} matched_pairs={len(pairs)}")


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
    if name in NEW_H3_LORAS:
        normalize_and_validate_h3_lora(path)
        validate_safetensors_header(path)
    print(f"Ready: {name} source_sha256={actual}")

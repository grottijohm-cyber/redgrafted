from __future__ import annotations

import hashlib
import os
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
        "https://civarchive.com/api/download/models/3229050",
        "PlagueKind-tiddies-realismslider.safetensors",
        None,
    ),
    (
        "https://civarchive.com/api/download/models/3226989",
        "deepthroat_v02.safetensors",
        None,
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
        "https://civarchive.com/api/download/models/3207723",
        "Pussy4nus_Epoch80.safetensors",
        None,
    ),
    (
        "https://civarchive.com/api/download/models/3264127",
        "MinimaxH3-Fingering_000002000.safetensors",
        None,
    ),
    (
        "https://civarchive.com/api/download/models/3228089",
        "moawxx_000002000.safetensors",
        None,
    ),
    (
        "https://civarchive.com/api/download/models/3304489",
        "SexGod_NaughtyTimes_v3_rank64_pruned_NOADALN.safetensors",
        None,
    ),
)

TARGET = Path(os.getenv("COMFY_MODELS", "/comfyui/models")) / "loras"
TARGET.mkdir(parents=True, exist_ok=True)

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
    if expected is not None and actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    print(f"Ready: {name} sha256={actual}")

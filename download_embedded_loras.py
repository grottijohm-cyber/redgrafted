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
)

TARGET = Path(os.getenv("COMFY_MODELS", "/comfyui/models")) / "loras"
TARGET.mkdir(parents=True, exist_ok=True)

for url, name, expected in FILES:
    path = TARGET / name
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 redgrafted-builder/1.0"})
    with urlopen(request, timeout=120) as response, path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    print(f"Ready: {name}")

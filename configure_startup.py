"""Build-time adapter for worker-comfyui 5.10.0's two launch modes."""

from pathlib import Path


def configure(path: Path) -> None:
    original = path.read_text()
    launch = "python -u /comfyui/main.py "
    if original.count(launch) != 2 or 'COMFY_PID_FILE="/tmp/comfyui.pid"' not in original:
        raise RuntimeError("Upstream /start.sh changed; review ComfyUI launch and PID tracking before building")
    path.write_text(original.replace(launch, "python -u /comfy_launcher.py "))


if __name__ == "__main__":
    configure(Path("/start.sh"))

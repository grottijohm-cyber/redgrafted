"""Run upstream ComfyUI in the same PID with container-oriented memory flags."""

import json
import os
import runpy
import sys

import runtime_health

CONSERVATIVE_FLAGS = ["--cache-none", "--disable-pinned-memory",
                      "--disable-async-offload", "--fast-disk"]


def launch(main_path="/comfyui/main.py"):
    profile = os.getenv("COMFY_MEMORY_PROFILE", "conservative")
    if profile not in {"conservative", "default"}:
        raise ValueError("COMFY_MEMORY_PROFILE must be conservative or default")
    flags = CONSERVATIVE_FLAGS if profile == "conservative" else []
    sys.argv = [main_path, *flags, *sys.argv[1:]]
    sys.path.insert(0, os.path.dirname(main_path))
    print("LTX worker container diagnostics: " + json.dumps(runtime_health.diagnostics()), flush=True)
    print(f"LTX worker ComfyUI memory profile: {profile}; flags: {' '.join(flags)}", flush=True)
    # Dynamic VRAM remains enabled so large models can stay disk-backed.
    # run_path preserves the PID written by the upstream /start.sh script.
    runpy.run_path(main_path, run_name="__main__")


if __name__ == "__main__":
    launch()

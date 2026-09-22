"""Per-job MiniMax runtime controls.

The model files remain baked into the worker image. These helpers only patch the
in-memory ComfyUI workflow for the current request, so changing sliders or AV
switches does not require rebuilding the Docker image.
"""

from __future__ import annotations

import math
from typing import Any


LORA_OPTIONS: dict[str, tuple[str, float]] = {
    "turbo_strength": ("390", 0.85),
    "m3_strength": ("391", 0.50),
    "mystic_strength": ("392", 1.00),
    "hmnsfw_strength": ("393", 1.00),
    "vagassist_strength": ("400", 1.00),
    "hmpussy_strength": ("401", 0.35),
    "cumshot_strength": ("402", 0.70),
}
MINIMAX_RUNTIME_OPTION_NAMES = frozenset({*LORA_OPTIONS, "steps", "enable_audio", "enable_gimm"})


def _number(value: Any, name: str, default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"input.{name} must be a number from {minimum} to {maximum}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input.{name} must be a number from {minimum} to {maximum}") from exc
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"input.{name} must be between {minimum} and {maximum}")
    return round(parsed, 4)


def _boolean(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"input.{name} must be true or false")


def _inject_ai_upscale(workflow: dict[str, Any]) -> None:
    """Enhance the source image before conditioning and 2x the decoded video."""
    # These are native ComfyUI nodes. The Real-ESRGAN weights are baked into the
    # worker image so generation never has to download the upscaler at runtime.
    workflow["405"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": "RealESRGAN_x2plus.pth"},
    }
    workflow["406"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["405", 0],
            "image": ["395", 0],
        },
    }
    workflow["408"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["405", 0],
            "image": ["374", 0],
        },
    }
    try:
        # Input photo: Real-ESRGAN first, then normalize to MiniMax's native
        # 544x960 conditioning canvas.
        workflow["350"]["inputs"]["image"] = ["406", 0]
        # Video: upscale decoded frames before optional GIMM interpolation.
        workflow["399"]["inputs"]["images"] = ["408", 0]
    except (KeyError, TypeError) as exc:
        raise ValueError("MiniMax AI-upscale routing nodes are missing") from exc


def apply_minimax_runtime_options(workflow: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    """Patch slider values into one MiniMax workflow copy and return the applied settings."""
    applied: dict[str, Any] = {}
    for name, (node_id, default) in LORA_OPTIONS.items():
        value = _number(job_input.get(name), name, default, 0.0, 1.5)
        try:
            workflow[node_id]["inputs"]["strength_model"] = value
        except (KeyError, TypeError) as exc:
            raise ValueError(f"MiniMax runtime control node {node_id} is missing") from exc
        applied[name] = value

    raw_steps = job_input.get("steps")
    if raw_steps is None:
        steps = 8
    elif isinstance(raw_steps, bool):
        raise ValueError("input.steps must be an integer from 4 to 16")
    else:
        try:
            numeric_steps = float(raw_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("input.steps must be an integer from 4 to 16") from exc
        if not numeric_steps.is_integer() or numeric_steps < 4 or numeric_steps > 16:
            raise ValueError("input.steps must be an integer from 4 to 16")
        steps = int(numeric_steps)
    try:
        workflow["397"]["inputs"]["steps"] = steps
    except (KeyError, TypeError) as exc:
        raise ValueError("MiniMax scheduler node 397 is missing") from exc
    applied["steps"] = steps

    _inject_ai_upscale(workflow)

    audio_enabled = _boolean(job_input.get("enable_audio"), "enable_audio", True)
    gimm_enabled = _boolean(job_input.get("enable_gimm"), "enable_gimm", True)
    try:
        create_video = workflow["370"]["inputs"]
        # Video frames are always AI-upscaled first. When GIMM is enabled it
        # interpolates the already-upscaled frames; when disabled we encode the
        # 2x Real-ESRGAN frames directly at the native 24 fps.
        if gimm_enabled:
            create_video["images"] = ["399", 0]
            create_video["fps"] = 48.0
        else:
            create_video["images"] = ["408", 0]
            create_video["fps"] = 24.0
        if audio_enabled:
            create_video["audio"] = ["358", 0]
        else:
            create_video.pop("audio", None)
    except (KeyError, TypeError) as exc:
        raise ValueError("MiniMax CreateVideo node 370 is missing") from exc

    applied["enable_audio"] = audio_enabled
    applied["enable_gimm"] = gimm_enabled
    applied["ai_upscale"] = "RealESRGAN_x2plus"
    applied["output_width"] = 1088
    applied["output_height"] = 1920
    return applied

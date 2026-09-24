"""Per-job MiniMax runtime controls.

LoRA sliders are rebuilt into a model chain for every request. A strength of
exactly zero removes that loader node from the submitted workflow, so disabled
LoRAs are genuinely bypassed rather than loaded at strength 0.
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
    "realism_strength": ("410", 0.0),
    "deepthroat_strength": ("411", 0.0),
    "civ3210503_strength": ("412", 0.0),
    "civ3320641_strength": ("413", 0.0),
    "pussy4nus_strength": ("414", 0.0),
    "fingering_strength": ("415", 0.0),
    "moawxx_strength": ("416", 0.0),
    "naughtytimes_strength": ("417", 0.0),
    "astro_strength": ("420", 0.0),
    "icy_real_strength": ("421", 0.0),
    "hogtied_strength": ("422", 0.0),
    "upskirt_strength": ("423", 0.0),
    "all_tied_up_strength": ("424", 0.0),
    "doggy_pov_strength": ("425", 0.0),
}
QUALITY_PROFILES: dict[str, tuple[int, int]] = {
    "fast": (544, 960),
    "balanced": (672, 1184),
    "quality": (768, 1344),
}
MINIMAX_RUNTIME_OPTION_NAMES = frozenset({
    *LORA_OPTIONS, "after_midnight_strength", "steps", "enable_audio", "enable_gimm", "enable_ai_upscale", "quality_mode"
})


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


def reference_lora_strength(job_input: dict[str, Any]) -> float:
    return _number(job_input.get("after_midnight_strength"), "after_midnight_strength", 0.0, 0.0, 2.0)


def _inject_ai_upscale(workflow: dict[str, Any]) -> None:
    workflow["405"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": "RealESRGAN_x2plus.pth"},
    }
    workflow["406"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["405", 0], "image": ["395", 0]},
    }
    workflow["408"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["405", 0], "image": ["374", 0]},
    }
    workflow["350"]["inputs"]["image"] = ["406", 0]
    workflow["399"]["inputs"]["images"] = ["408", 0]


def _apply_lora_chain(workflow: dict[str, Any], job_input: dict[str, Any]) -> dict[str, float]:
    applied: dict[str, float] = {}
    previous = "384"
    for name, (node_id, default) in LORA_OPTIONS.items():
        value = _number(job_input.get(name), name, default, 0.0, 2.0)
        applied[name] = value
        if value == 0.0:
            workflow.pop(node_id, None)
            continue
        try:
            node = workflow[node_id]
            node["inputs"]["model"] = [previous, 0]
            node["inputs"]["strength_model"] = value
        except (KeyError, TypeError) as exc:
            raise ValueError(f"MiniMax runtime control node {node_id} is missing") from exc
        previous = node_id
    try:
        workflow["394"]["inputs"]["model"] = [previous, 0]
    except (KeyError, TypeError) as exc:
        raise ValueError("MiniMax SigmaShift node 394 is missing") from exc
    return applied


def apply_minimax_runtime_options(workflow: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = _apply_lora_chain(workflow, job_input)
    applied["after_midnight_strength"] = reference_lora_strength(job_input)

    quality_mode = str(job_input.get("quality_mode") or "fast").strip().lower()
    if quality_mode not in QUALITY_PROFILES:
        raise ValueError("input.quality_mode must be fast, balanced, or quality")
    width, height = QUALITY_PROFILES[quality_mode]
    workflow["350"]["inputs"]["width"] = width
    workflow["350"]["inputs"]["height"] = height
    workflow["364"]["inputs"]["width"] = width
    workflow["364"]["inputs"]["height"] = height
    applied["quality_mode"] = quality_mode
    applied["generation_width"] = width
    applied["generation_height"] = height

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
    workflow["397"]["inputs"]["steps"] = steps
    applied["steps"] = steps

    audio_enabled = _boolean(job_input.get("enable_audio"), "enable_audio", True)
    gimm_enabled = _boolean(job_input.get("enable_gimm"), "enable_gimm", True)
    upscale_enabled = _boolean(job_input.get("enable_ai_upscale"), "enable_ai_upscale", False)

    if upscale_enabled:
        _inject_ai_upscale(workflow)
    else:
        workflow["350"]["inputs"]["image"] = ["395", 0]
        workflow["399"]["inputs"]["images"] = ["374", 0]
        for node_id in ("405", "406", "408"):
            workflow.pop(node_id, None)

    try:
        create_video = workflow["370"]["inputs"]
        if gimm_enabled:
            create_video["images"] = ["399", 0]
            create_video["fps"] = 48.0
        else:
            create_video["images"] = ["408", 0] if upscale_enabled else ["374", 0]
            create_video["fps"] = 24.0
        if audio_enabled:
            create_video["audio"] = ["358", 0]
        else:
            create_video.pop("audio", None)
    except (KeyError, TypeError) as exc:
        raise ValueError("MiniMax CreateVideo node 370 is missing") from exc

    applied["enable_audio"] = audio_enabled
    applied["enable_gimm"] = gimm_enabled
    applied["enable_ai_upscale"] = upscale_enabled
    applied["ai_upscale"] = "RealESRGAN_x2plus" if upscale_enabled else None
    applied["output_width"] = width * 2 if upscale_enabled else width
    applied["output_height"] = height * 2 if upscale_enabled else height
    return applied

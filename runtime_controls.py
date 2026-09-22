"""Per-job MiniMax runtime controls.

The model files remain baked into the worker image. These helpers only patch the
in-memory ComfyUI workflow for the current request, so changing a slider does not
require rebuilding the Docker image.
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
MINIMAX_RUNTIME_OPTION_NAMES = frozenset({*LORA_OPTIONS, "steps"})


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
    return applied

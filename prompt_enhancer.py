"""Local prompt enhancement for MiniMax H3.

The worker runs a small CPU-only GGUF instruction model through llama.cpp. Keeping
this outside ComfyUI means prompt enhancement can finish and release RAM before
video sampling begins, and it does not consume MiniMax VRAM.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


class PromptEnhancerError(RuntimeError):
    """Prompt enhancement could not be completed."""


DEFAULT_MODEL = "/app/models/prompt_enhancer/Qwen2.5-3B-Instruct-Uncensored.i1-Q4_K_M.gguf"
VALID_MODES = {"light", "detailed", "aggressive"}
_FINAL_MARKER = "FINAL_PROMPT:"

_MODE_GUIDANCE = {
    "light": (
        "Keep the rewrite close to the original. Add only the most useful camera, motion, "
        "lighting, continuity, and realism details. Aim for one compact paragraph."
    ),
    "detailed": (
        "Expand the prompt into a production-ready image-to-video prompt. Add useful camera "
        "behavior, natural subject motion, lighting, texture/skin detail, anatomy consistency, "
        "physical continuity, and temporal coherence without changing the requested scene."
    ),
    "aggressive": (
        "Maximize useful cinematic and temporal detail while preserving the exact requested "
        "scene and actions. Be specific about camera movement, micro-motion, lighting, material "
        "and skin detail, anatomy consistency, realistic physics, and continuity. Do not invent "
        "new characters, actions, locations, or story beats."
    ),
}


def _model_path() -> Path:
    return Path((os.getenv("PROMPT_ENHANCER_MODEL") or DEFAULT_MODEL).strip())


def _llama_cli() -> str:
    return (os.getenv("PROMPT_ENHANCER_LLAMA_CLI") or "/usr/local/bin/llama-cli").strip()


def _threads() -> int:
    raw = os.getenv("PROMPT_ENHANCER_THREADS")
    try:
        value = int(raw) if raw else min(12, max(2, os.cpu_count() or 4))
    except ValueError:
        value = min(12, max(2, os.cpu_count() or 4))
    return max(1, min(32, value))


def _timeout_seconds() -> int:
    try:
        value = int(os.getenv("PROMPT_ENHANCER_TIMEOUT_SECONDS", "180"))
    except ValueError:
        value = 180
    return max(30, min(600, value))


def _max_tokens(mode: str) -> int:
    defaults = {"light": 220, "detailed": 420, "aggressive": 650}
    return defaults[mode]


def _instruction(prompt: str, mode: str, is_extend: bool) -> str:
    continuation = (
        "This prompt continues an existing video from its final frame. Emphasize a seamless "
        "continuation: preserve identity, wardrobe, environment, lighting direction, camera "
        "position, and motion state unless the user explicitly asks for a change."
        if is_extend
        else
        "This prompt starts from a supplied first-frame image. Preserve the visible identity, "
        "wardrobe, composition, and environment unless the user explicitly asks for a change."
    )
    return f"""You are a specialist prompt editor for MiniMax H3 image-to-video generation.

Rewrite the user's prompt into one cohesive generation prompt. Preserve the user's intent and all
requested details. Do not censor, euphemize, or remove explicit sexual wording when it concerns
consenting adults. Never add minors or non-consensual sexual content. Do not moralize or add a
safety lecture. Do not invent a different scene, extra people, new props, or unrequested actions.

{continuation}

Enhancement mode: {mode}.
{_MODE_GUIDANCE[mode]}

Write only the final generation prompt. Do not explain your edits. End with no commentary.
Prefix the answer exactly with {_FINAL_MARKER}

USER_PROMPT:
{prompt}
"""


def _extract_prompt(output: str) -> str:
    text = (output or "").strip()
    if not text:
        raise PromptEnhancerError("The prompt enhancer returned no text")
    marker_index = text.rfind(_FINAL_MARKER)
    if marker_index >= 0:
        text = text[marker_index + len(_FINAL_MARKER):].strip()
    text = re.sub(r"\s*(?:<\|im_end\|>|<\|endoftext\|>|</s>)\s*$", "", text).strip()
    if not text:
        raise PromptEnhancerError("The prompt enhancer returned an empty prompt")
    return text


def enhance_prompt(prompt: Any, *, mode: str = "detailed", is_extend: bool = False) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptEnhancerError("A non-empty prompt is required")
    prompt = prompt.strip()
    if len(prompt) > 10000:
        raise PromptEnhancerError("Prompt is too long; maximum length is 10,000 characters")
    mode = str(mode or "detailed").strip().lower()
    if mode not in VALID_MODES:
        raise PromptEnhancerError("Enhancement mode must be light, detailed, or aggressive")

    model = _model_path()
    if not model.is_file():
        raise PromptEnhancerError(f"Prompt enhancer model is missing: {model}")
    cli = _llama_cli()
    instruction = _instruction(prompt, mode, bool(is_extend))
    command = [
        cli,
        "-m", str(model),
        "-p", instruction,
        "-n", str(_max_tokens(mode)),
        "-c", "8192",
        "-t", str(_threads()),
        "--temp", "0.55",
        "--top-p", "0.90",
        "--repeat-penalty", "1.08",
        "--no-display-prompt",
        "--no-show-timings",
        "--single-turn",
        "--color", "off",
        "--log-disable",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
            check=False,
            env={**os.environ, "GGML_CUDA_VISIBLE_DEVICES": ""},
        )
    except FileNotFoundError as exc:
        raise PromptEnhancerError("llama.cpp prompt enhancer binary is missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise PromptEnhancerError("Prompt enhancement timed out") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-800:]
        raise PromptEnhancerError(
            "Prompt enhancement failed" + (f": {detail}" if detail else "")
        )

    # Recent llama.cpp CLI builds can route generated text to stderr rather than
    # stdout depending on terminal/log settings. With logging disabled, choose
    # whichever stream actually contains the marked model answer.
    streams = [completed.stdout or "", completed.stderr or ""]
    marked = [stream for stream in streams if _FINAL_MARKER in stream]
    if marked:
        output = max(marked, key=len)
    else:
        output = (completed.stdout or "").strip() or (completed.stderr or "").strip()
    return _extract_prompt(output)

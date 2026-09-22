"""Forward useful ComfyUI execution progress to RunPod job progress updates."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse, urlunparse


LOGGER = logging.getLogger("comfy-progress")


class ComfyProgressTracker:
    """Listen to ComfyUI's websocket and translate node progress into 0-100%."""

    NODE_STAGES: dict[str, tuple[str, int]] = {
        "406": ("Enhancing input image", 5),
        "364": ("Preparing MiniMax conditioning", 8),
        "344": ("Sampling video", 10),
        "374": ("Decoding video", 76),
        "358": ("Decoding audio", 82),
        "408": ("AI upscaling video", 84),
        "398": ("Loading frame interpolation", 91),
        "399": ("Interpolating frames", 92),
        "370": ("Encoding video", 98),
        "75": ("Saving video", 99),
    }

    def __init__(
        self,
        comfy_url: str,
        job: dict[str, Any],
        prompt_id: str,
        client_id: str,
        reporter: Callable[..., None],
    ) -> None:
        self.comfy_url = comfy_url
        self.job = job
        self.prompt_id = prompt_id
        self.client_id = client_id
        self.reporter = reporter
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_node = ""
        self._last_percent = 0
        self._last_signature: tuple[str, int, str] | None = None
        self._socket = None

    def start(self) -> "ComfyProgressTracker":
        self._thread = threading.Thread(target=self._run, name="comfy-progress", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _websocket_url(self) -> str:
        parsed = urlparse(self.comfy_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, "/ws", "", f"clientId={self.client_id}", ""))

    def _emit(self, stage: str, percent: int, detail: str = "") -> None:
        percent = max(self._last_percent, min(99, max(0, int(percent))))
        signature = (stage, percent, detail)
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self._last_percent = percent
        self.reporter(self.job, stage, progress=percent, detail=detail or None)

    def _prompt_matches(self, data: dict[str, Any]) -> bool:
        supplied = data.get("prompt_id")
        return supplied is None or str(supplied) == self.prompt_id

    def _handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        data = message.get("data")
        if not isinstance(data, dict) or not self._prompt_matches(data):
            return

        if kind == "executing":
            node = data.get("node")
            if node is None:
                return
            self._current_node = str(node)
            stage = self.NODE_STAGES.get(self._current_node)
            if stage:
                self._emit(stage[0], stage[1])
            return

        if kind != "progress":
            return
        try:
            value = float(data.get("value"))
            maximum = float(data.get("max"))
        except (TypeError, ValueError):
            return
        if maximum <= 0:
            return
        ratio = max(0.0, min(1.0, value / maximum))

        if self._current_node == "344":
            percent = 10 + round(ratio * 65)
            self._emit("Sampling video", percent, f"Step {int(value)} / {int(maximum)}")
        elif self._current_node == "399":
            percent = 92 + round(ratio * 5)
            self._emit("Interpolating frames", percent, f"{round(ratio * 100)}% of interpolation")
        else:
            stage = self.NODE_STAGES.get(self._current_node)
            if stage:
                self._emit(stage[0], stage[1])

    def _run(self) -> None:
        try:
            import websocket

            sock = websocket.create_connection(self._websocket_url(), timeout=5)
            sock.settimeout(1)
            self._socket = sock
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    if not self._stop.is_set():
                        LOGGER.debug("ComfyUI progress websocket ended", exc_info=True)
                    break
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    self._handle(message)
        except Exception:
            # Progress is best-effort; generation must continue if the websocket
            # is unavailable or the base image changes its websocket behavior.
            if not self._stop.is_set():
                LOGGER.warning("Could not attach ComfyUI progress websocket", exc_info=True)
        finally:
            self._socket = None

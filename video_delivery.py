"""Create a bounded MP4 response when object storage is not configured."""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from pathlib import Path


class DeliveryError(RuntimeError):
    pass


def _run(arguments: list[str], deadline: float | None) -> str:
    remaining = 240 if deadline is None else min(240, deadline - time.monotonic())
    if remaining <= 0:
        raise DeliveryError("Video delivery exceeded the job time budget")
    try:
        result = subprocess.run(arguments, capture_output=True, text=True,
                                timeout=remaining, check=True)
        return result.stdout
    except FileNotFoundError as exc:
        raise DeliveryError("Video delivery requires ffmpeg and ffprobe in the image") from exc
    except subprocess.TimeoutExpired as exc:
        raise DeliveryError("Video compression exceeded its time budget") from exc
    except subprocess.CalledProcessError as exc:
        raise DeliveryError("Video compression failed: " + exc.stderr[-1200:]) from exc


def inline_video(path: Path, max_bytes: int, deadline: float | None = None) -> tuple[bytes, bool]:
    if path.stat().st_size <= max_bytes:
        return path.read_bytes(), False
    if path.suffix.lower() != ".mp4":
        raise DeliveryError("This output is too large to return inline; configure object storage")
    try:
        info = json.loads(_run(["ffprobe", "-v", "error", "-show_entries",
                               "format=duration", "-of", "json", str(path)], deadline))
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid duration")
    except (KeyError, ValueError, TypeError) as exc:
        raise DeliveryError("Could not determine the generated video's duration") from exc

    # Reserve room for audio, the MP4 container, and bitrate estimation variation.
    video_bitrate = int(max_bytes * 0.80 * 8 / duration) - 96_000
    if video_bitrate < 64_000:
        raise DeliveryError("Video is too long for an inline response; configure object storage")
    with tempfile.TemporaryDirectory(prefix="runpod-video-") as directory:
        output = Path(directory) / "inline.mp4"
        common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                  "-i", str(path), "-map", "0:v:0", "-c:v", "libx264", "-preset", "fast",
                  "-pix_fmt", "yuv420p", "-b:v", str(video_bitrate),
                  "-passlogfile", str(Path(directory) / "encode")]
        _run(common + ["-pass", "1", "-an", "-f", "null", "-"], deadline)
        _run(common + ["-pass", "2", "-map", "0:a:0?", "-c:a", "aac", "-b:a", "96k",
                       "-movflags", "+faststart", str(output)], deadline)
        if output.stat().st_size > max_bytes:
            raise DeliveryError("Compressed video still exceeds the response limit; configure object storage")
        return output.read_bytes(), True

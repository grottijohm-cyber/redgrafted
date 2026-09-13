import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from video_delivery import inline_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class VideoDeliveryTests(unittest.TestCase):
    def test_oversized_video_returns_playable_mp4_with_audio_within_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=640x360:rate=24:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-threads", "2",
                "-c:a", "aac", "-shortest", str(source),
            ], check=True, timeout=60, capture_output=True)
            budget = 150_000
            self.assertGreater(source.stat().st_size, budget)
            payload, compressed = inline_video(source, budget)
            self.assertTrue(compressed)
            self.assertLessEqual(len(payload), budget)
            self.assertTrue(source.is_file())
            delivered = Path(directory) / "delivered.mp4"
            delivered.write_bytes(payload)
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                "-of", "json", str(delivered),
            ], check=True, timeout=10, capture_output=True, text=True)
            info = json.loads(probe.stdout)
            self.assertEqual({s["codec_type"] for s in info["streams"]}, {"video", "audio"})
            self.assertAlmostEqual(float(info["format"]["duration"]), 4, delta=0.2)


if __name__ == "__main__":
    unittest.main()

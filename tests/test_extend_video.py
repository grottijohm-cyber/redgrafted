import unittest
from unittest.mock import patch

import app_worker


class ExtendVideoTests(unittest.TestCase):
    def test_extend_uses_last_frame_and_inherits_render_settings(self):
        metadata = {
            "render_id": "render-123",
            "preset_name": "Full Stack",
            "request_settings": {
                "length_seconds": 8,
                "turbo_strength": 0.85,
                "m3_strength": 0.5,
                "mystic_strength": 1.0,
                "hmnsfw_strength": 1.0,
                "vagassist_strength": 1.0,
                "hmpussy_strength": 0.35,
                "cumshot_strength": 0.7,
                "steps": 8,
                "enable_audio": True,
                "enable_gimm": True,
            },
        }
        generated = {
            "status": "success",
            "worker_version": "runpod-minimax-test",
            "seed": 7,
            "prompt_id": "prompt-1",
            "workflow": {"model_profile": "minimax"},
            "videos": [{"type": "url", "filename": "extension.mp4", "url": "https://temporary.example/extension.mp4", "mime_type": "video/mp4"}],
        }
        archived = {
            "render_id": "render-extended",
            "permanent": True,
            "videos": [{"type": "url", "filename": "extension.mp4", "url": "https://signed.example/extension.mp4", "mime_type": "video/mp4"}],
        }
        job = {
            "id": "extend-job",
            "input": {
                "action": "extend",
                "render_id": "render-123",
                "prompt": "Continue the motion smoothly.",
                "length_seconds": 12,
            },
        }

        with patch.object(
            app_worker,
            "_extension_frame",
            return_value=(metadata, "data:image/jpeg;base64,ZmFrZQ=="),
        ), patch.object(
            app_worker, "base_handle_job", return_value=generated.copy()
        ) as base, patch.object(
            app_worker, "archive_generation_result", return_value=archived
        ) as archive:
            result = app_worker.handle_job(job)

        forwarded = base.call_args.args[0]["input"]
        self.assertEqual(forwarded["image"], "data:image/jpeg;base64,ZmFrZQ==")
        self.assertEqual(forwarded["prompt"], "Continue the motion smoothly.")
        self.assertEqual(forwarded["length_seconds"], 12)
        self.assertEqual(forwarded["turbo_strength"], 0.85)
        self.assertEqual(forwarded["enable_gimm"], True)
        self.assertNotIn("action", forwarded)
        self.assertNotIn("render_id", forwarded)

        archive_settings = archive.call_args.kwargs["request_settings"]
        self.assertEqual(archive_settings["extended_from_render_id"], "render-123")
        self.assertEqual(archive_settings["extension_method"], "last-frame-i2v")
        self.assertEqual(archive.call_args.kwargs["preset_name"], "Full Stack")
        self.assertEqual(result["extension"]["from_render_id"], "render-123")
        self.assertEqual(result["videos"], archived["videos"])
        self.assertEqual(result["app_worker_version"], "redgraft-library-3")

    def test_extend_rejects_arbitrary_source_urls(self):
        result = app_worker.handle_job({
            "input": {
                "action": "extend",
                "render_id": "render-123",
                "prompt": "Continue.",
                "video_url": "https://example.com/untrusted.mp4",
            }
        })
        self.assertIn("error", result)
        self.assertIn("supports", result["error"])


if __name__ == "__main__":
    unittest.main()

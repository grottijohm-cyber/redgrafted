import unittest
from pathlib import Path
from unittest.mock import patch

import app_worker


class ExtendVideoTests(unittest.TestCase):
    def test_extend_uses_last_frame_inherits_settings_and_archives_combined_video(self):
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
        combined_video = {
            "type": "url",
            "filename": "extended.mp4",
            "url": "https://signed.example/extended.mp4",
            "download_url": "https://signed.example/download-extended.mp4",
            "mime_type": "video/mp4",
            "archive_key": "extend-job/extended.mp4",
        }
        archived = {
            "render_id": "render-extended",
            "permanent": True,
            "videos": [combined_video],
        }
        job = {
            "id": "extend-job",
            "input": {
                "action": "extend",
                "render_id": "render-123",
                "prompt": "A polished continuation prompt.",
                "original_prompt": "Continue the motion smoothly.",
                "enhanced_prompt": "A polished continuation prompt.",
                "used_prompt": "A polished continuation prompt.",
                "prompt_enhancement_enabled": True,
                "prompt_enhancement_mode": "detailed",
                "prompt_enhancement_previewed": True,
                "length_seconds": 12,
            },
        }

        with patch.object(
            app_worker,
            "_prepare_extension_source",
            return_value=(metadata, Path("/tmp/source.mp4"), "data:image/jpeg;base64,ZmFrZQ=="),
        ), patch.object(
            app_worker, "base_handle_job", return_value=generated.copy()
        ) as base, patch.object(
            app_worker,
            "download_job_result_video",
            return_value={"key": "extend-job/extension.mp4", "filename": "extension.mp4", "mime_type": "video/mp4"},
        ), patch.object(
            app_worker, "_concat_videos"
        ) as concat, patch.object(
            app_worker, "upload_job_video", return_value=combined_video
        ), patch.object(
            app_worker, "archive_generation_result", return_value=archived
        ) as archive:
            result = app_worker.handle_job(job)

        forwarded = base.call_args.args[0]["input"]
        self.assertEqual(forwarded["image"], "data:image/jpeg;base64,ZmFrZQ==")
        self.assertEqual(forwarded["prompt"], "A polished continuation prompt.")
        self.assertEqual(forwarded["length_seconds"], 12)
        self.assertEqual(forwarded["turbo_strength"], 0.85)
        self.assertEqual(forwarded["enable_gimm"], True)
        self.assertNotIn("action", forwarded)
        self.assertNotIn("render_id", forwarded)
        concat.assert_called_once()

        archive_settings = archive.call_args.kwargs["request_settings"]
        self.assertEqual(archive_settings["extended_from_render_id"], "render-123")
        self.assertEqual(archive_settings["extension_method"], "last-frame-i2v-concat")
        self.assertTrue(archive_settings["extension_merged"])
        self.assertEqual(archive.call_args.kwargs["preset_name"], "Full Stack")
        self.assertEqual(archive.call_args.kwargs["prompt_metadata"]["original_prompt"], "Continue the motion smoothly.")
        self.assertTrue(archive.call_args.kwargs["extension_metadata"]["merged"])
        self.assertEqual(result["extension"]["from_render_id"], "render-123")
        self.assertEqual(result["videos"], archived["videos"])
        self.assertEqual(result["app_worker_version"], "redgraft-library-6")

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

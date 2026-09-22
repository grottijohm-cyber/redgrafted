import unittest
from unittest.mock import patch

import app_worker


class PresetInferenceTests(unittest.TestCase):
    def test_infers_general_nsfw_without_ui_only_preset_name(self):
        payload = {
            "image": "https://example.com/image.jpg",
            "prompt": "Prompt",
            "length_seconds": 8,
            "turbo_strength": 0.85,
            "m3_strength": 0.50,
            "mystic_strength": 0.55,
            "hmnsfw_strength": 0.55,
            "vagassist_strength": 0.0,
            "hmpussy_strength": 0.0,
            "cumshot_strength": 0.0,
            "steps": 8,
            "enable_audio": False,
            "enable_gimm": False,
        }
        self.assertEqual(app_worker._infer_preset_name(payload), "General NSFW")

    def test_custom_settings_remain_custom(self):
        payload = {
            "length_seconds": 9,
            "turbo_strength": 0.83,
            "m3_strength": 0.47,
            "mystic_strength": 0.51,
            "hmnsfw_strength": 0.49,
            "vagassist_strength": 0.0,
            "hmpussy_strength": 0.0,
            "cumshot_strength": 0.0,
            "steps": 8,
            "enable_audio": False,
            "enable_gimm": False,
        }
        self.assertEqual(app_worker._infer_preset_name(payload), "Custom")

    def test_archiver_uses_inferred_preset_when_client_omits_label(self):
        job = {
            "id": "job-1",
            "input": {
                "image": "https://example.com/image.jpg",
                "prompt": "Prompt",
                "length_seconds": 8,
                "turbo_strength": 0.85,
                "m3_strength": 0.40,
                "mystic_strength": 0.0,
                "hmnsfw_strength": 0.0,
                "vagassist_strength": 0.0,
                "hmpussy_strength": 0.0,
                "cumshot_strength": 0.0,
                "steps": 6,
                "enable_audio": False,
                "enable_gimm": False,
            },
        }
        base = {
            "status": "success",
            "videos": [{"filename": "render.mp4", "type": "url", "url": "https://temporary"}],
        }
        archived = {
            "permanent": True,
            "render_id": "r1",
            "videos": [{"filename": "render.mp4", "type": "url", "url": "https://signed"}],
        }
        with patch.object(app_worker, "base_handle_job", return_value=base), patch.object(
            app_worker, "archive_generation_result", return_value=archived
        ) as archive:
            result = app_worker.handle_job(job)
        self.assertEqual(archive.call_args.kwargs["preset_name"], "Fast Test")
        self.assertTrue(result["archive"]["permanent"])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import app_worker


class PromptEnhancerRemovalTests(unittest.TestCase):
    def test_enhance_action_is_disabled(self):
        result = app_worker.handle_job({"input": {"action": "enhance_prompt", "prompt": "Test."}})
        self.assertIn("error", result)
        self.assertIn("disabled", result["error"].lower())

    def test_generation_uses_raw_prompt_even_if_old_client_sends_enhancement_metadata(self):
        generated = {"status": "success", "videos": []}
        job = {
            "id": "job-1",
            "input": {
                "image": "https://example.com/image.jpg",
                "prompt": "Raw prompt.",
                "original_prompt": "Original prompt.",
                "enhanced_prompt": "Enhanced prompt.",
                "used_prompt": "Enhanced prompt.",
                "prompt_enhancement_enabled": True,
                "prompt_enhancement_mode": "detailed",
                "prompt_enhancement_previewed": True,
            },
        }
        with patch.object(app_worker, "base_handle_job", return_value=generated.copy()) as base, patch.object(
            app_worker, "archive_generation_result", return_value=None
        ) as archive:
            result = app_worker.handle_job(job)

        forwarded = base.call_args.args[0]["input"]
        self.assertEqual(forwarded["prompt"], "Raw prompt.")
        for key in app_worker._PROMPT_METADATA_FIELDS:
            self.assertNotIn(key, forwarded)
        prompt_meta = archive.call_args.kwargs["prompt_metadata"]
        self.assertEqual(prompt_meta["original_prompt"], "Raw prompt.")
        self.assertIsNone(prompt_meta["enhanced_prompt"])
        self.assertFalse(prompt_meta["prompt_enhancement_enabled"])
        self.assertEqual(result["app_worker_version"], "redgraft-library-5")


if __name__ == "__main__":
    unittest.main()

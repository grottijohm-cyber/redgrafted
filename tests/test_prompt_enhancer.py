import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app_worker
import prompt_enhancer


class PromptEnhancerTests(unittest.TestCase):
    def test_enhance_action_returns_original_and_enhanced_prompt(self):
        with patch.object(app_worker, "enhance_prompt", return_value="Enhanced cinematic prompt.") as enhance:
            result = app_worker.handle_job({
                "input": {
                    "action": "enhance_prompt",
                    "prompt": "Original prompt.",
                    "mode": "detailed",
                    "is_extend": True,
                }
            })
        enhance.assert_called_once_with("Original prompt.", mode="detailed", is_extend=True)
        self.assertEqual(result["original_prompt"], "Original prompt.")
        self.assertEqual(result["enhanced_prompt"], "Enhanced cinematic prompt.")
        self.assertEqual(result["used_prompt"], "Enhanced cinematic prompt.")
        self.assertEqual(result["app_worker_version"], "redgraft-library-4")

    def test_generation_strips_prompt_metadata_before_base_worker(self):
        generated = {
            "status": "success",
            "worker_version": "worker-test",
            "videos": [{"type": "url", "filename": "render.mp4", "url": "https://temp"}],
        }
        archived = {"permanent": True, "videos": [{"type": "url", "filename": "render.mp4", "url": "https://signed"}]}
        job = {
            "id": "job-1",
            "input": {
                "image": "data:image/png;base64,AA==",
                "prompt": "Enhanced prompt.",
                "original_prompt": "Original prompt.",
                "enhanced_prompt": "Enhanced prompt.",
                "used_prompt": "Enhanced prompt.",
                "prompt_enhancement_enabled": True,
                "prompt_enhancement_mode": "aggressive",
                "prompt_enhancement_previewed": True,
                "steps": 8,
            },
        }
        with patch.object(app_worker, "base_handle_job", return_value=generated) as base, patch.object(
            app_worker, "archive_generation_result", return_value=archived
        ) as archive:
            app_worker.handle_job(job)

        forwarded = base.call_args.args[0]["input"]
        self.assertEqual(forwarded["prompt"], "Enhanced prompt.")
        self.assertNotIn("original_prompt", forwarded)
        self.assertNotIn("enhanced_prompt", forwarded)
        self.assertNotIn("prompt_enhancement_enabled", forwarded)
        prompt_meta = archive.call_args.kwargs["prompt_metadata"]
        self.assertEqual(prompt_meta["original_prompt"], "Original prompt.")
        self.assertEqual(prompt_meta["used_prompt"], "Enhanced prompt.")
        self.assertEqual(prompt_meta["prompt_enhancement_mode"], "aggressive")

    def test_extract_prompt_uses_final_marker(self):
        text = "noise\nFINAL_PROMPT: A clean final prompt. <|im_end|>"
        self.assertEqual(prompt_enhancer._extract_prompt(text), "A clean final prompt.")

    def test_local_enhancer_builds_cpu_llama_command(self):
        completed = MagicMock(returncode=0, stdout="FINAL_PROMPT: Better prompt.", stderr="")
        with patch.object(prompt_enhancer, "_model_path", return_value=Path("/tmp/model.gguf")), patch.object(
            Path, "is_file", return_value=True
        ), patch.object(prompt_enhancer.subprocess, "run", return_value=completed) as run:
            result = prompt_enhancer.enhance_prompt("Original", mode="light", is_extend=False)
        self.assertEqual(result, "Better prompt.")
        command = run.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("/tmp/model.gguf", command)
        self.assertIn("-t", command)


if __name__ == "__main__":
    unittest.main()

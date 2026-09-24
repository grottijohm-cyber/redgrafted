import copy
import json
import unittest
from pathlib import Path

import model_setup
from runtime_controls import apply_minimax_runtime_options
from worker import _configure_ref2va_workflow


class Ref2VATests(unittest.TestCase):
    def workflow(self):
        return json.loads(Path("api-workflow-minimax.json").read_text(encoding="utf-8"))

    def test_ref2va_model_is_optional_cached_bundle_addon(self):
        paths = {item.relative_path for item in model_setup.MINIMAX_REFERENCE_FILES}
        self.assertIn(
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            paths,
        )
        self.assertIn(
            "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            paths,
        )
        base_paths = {item.relative_path for item in model_setup.MINIMAX_FILES}
        self.assertNotIn(
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            base_paths,
        )

    def test_ref2va_replaces_fl2va_conditioner_and_lora_chain(self):
        workflow = self.workflow()
        apply_minimax_runtime_options(workflow, {"quality_mode": "balanced"})
        applied = _configure_ref2va_workflow(
            workflow, {"reference_size": "match"}
        )
        self.assertEqual(workflow["384"]["inputs"]["unet_name"],
                         "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        self.assertEqual(workflow["364"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(workflow["364"]["inputs"]["ref_images.ref_image_0"], ["395", 0])
        self.assertEqual((workflow["364"]["inputs"]["width"], workflow["364"]["inputs"]["height"]),
                         (672, 1184))
        self.assertEqual(workflow["390"]["inputs"]["lora_name"],
                         "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
        self.assertEqual(workflow["390"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(workflow["388"]["inputs"]["model"], ["390", 0])
        self.assertEqual(workflow["397"]["inputs"]["model"], ["390", 0])
        self.assertEqual(workflow["397"]["inputs"]["steps"], 4)
        self.assertEqual(workflow["352"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(applied["reference_size"], "match")
        for node_id in ("391","392","393","394","400","401","402",
                        "410","411","412","413","414","415","416","417"):
            self.assertNotIn(node_id, workflow)

    def test_ref2va_rejects_unknown_reference_size(self):
        workflow = self.workflow()
        with self.assertRaisesRegex(Exception, "reference_size"):
            _configure_ref2va_workflow(workflow, {"reference_size": "huge"})


if __name__ == "__main__":
    unittest.main()

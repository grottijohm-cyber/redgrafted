import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import model_setup
from runtime_controls import LORA_OPTIONS, apply_minimax_runtime_options
from worker import _configure_ref2va_workflow


class Ref2VATests(unittest.TestCase):
    def workflow(self):
        return json.loads(Path("api-workflow-minimax.json").read_text(encoding="utf-8"))

    def test_ref2va_model_is_optional_cached_bundle_addon(self):
        paths = {item.relative_path for item in model_setup.MINIMAX_REFERENCE_FILES}
        revision = "7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1"
        self.assertEqual(len(model_setup.MINIMAX_REFERENCE_FILES), 2)
        for item in model_setup.MINIMAX_REFERENCE_FILES:
            self.assertIn(
                f"huggingface.co/Comfy-Org/MiniMax-H3/resolve/{revision}/",
                item.url,
            )
            self.assertEqual(len(item.expected_sha256), 64)
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

    def test_status_reports_missing_reference_assets_separately(self):
        reference_files = tuple(
            replace(item, min_bytes=1) for item in model_setup.MINIMAX_REFERENCE_FILES
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(model_setup, "MODEL_PROFILE", "minimax"),
                patch.object(model_setup, "MINIMAX_REFERENCE_FILES", reference_files),
                patch.object(model_setup, "_latest_snapshot", return_value=root),
            ):
                status = model_setup.model_status()
                self.assertFalse(status["reference_assets_present"])
                self.assertEqual(
                    status["missing_reference_files"],
                    [item.relative_path for item in reference_files],
                )

                for item in reference_files:
                    target = root / item.relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"x")

                status = model_setup.model_status()
                self.assertTrue(status["reference_assets_present"])
                self.assertEqual(status["missing_reference_files"], [])

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
        self.assertEqual(workflow["388"]["inputs"]["model"], ["435", 0])
        self.assertEqual(workflow["397"]["inputs"]["model"], ["435", 0])
        self.assertEqual(workflow["397"]["inputs"]["steps"], 4)
        self.assertEqual(workflow["352"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(applied["reference_size"], "match")
        for node_id in ("391","392","393","394","400","401","402",
                        "410","411","412","413","414","415","416","417",
                        "420","421","422","423","424","425","426"):
            self.assertNotIn(node_id, workflow)

    def test_after_midnight_is_applied_only_in_reference_mode(self):
        workflow = self.workflow()
        options = {name: 0 for name in LORA_OPTIONS}
        options["after_midnight_strength"] = 0.75
        apply_minimax_runtime_options(workflow, options)
        _configure_ref2va_workflow(workflow, {"after_midnight_strength": 0.75})
        self.assertEqual(workflow["450"]["inputs"]["lora_name"],
                         "AfterMidnight_ref2va_h3_sexytime_rank64-v1.2.safetensors")
        self.assertEqual(workflow["450"]["inputs"]["strength_model"], 0.75)
        self.assertEqual(workflow["388"]["inputs"]["model"], ["450", 0])
        self.assertEqual(workflow["397"]["inputs"]["model"], ["450", 0])

    def test_selected_h3_loras_are_chained_in_reference_mode(self):
        workflow = self.workflow()
        options = {name: 0 for name in LORA_OPTIONS}
        options.update({"civ3210503_strength": 0.6, "doggy_pov_strength": 0.4})
        apply_minimax_runtime_options(workflow, options)
        applied = _configure_ref2va_workflow(workflow, {})
        self.assertEqual(workflow["430"]["inputs"], {
            "model": ["390", 0],
            "lora_name": "H3_Mis_Insrt_v07.safetensors",
            "strength_model": 0.6,
        })
        self.assertEqual(workflow["431"]["inputs"]["model"], ["430", 0])
        self.assertEqual(workflow["431"]["inputs"]["lora_name"],
                         "hm_nsfw_POV_doggy_only_v16_r32_384_minimax-h3_epoch170.safetensors")
        self.assertEqual(workflow["388"]["inputs"]["model"], ["431", 0])
        self.assertEqual(applied["reference_loras"], [
            "H3_Mis_Insrt_v07.safetensors",
            "hm_nsfw_POV_doggy_only_v16_r32_384_minimax-h3_epoch170.safetensors",
        ])

    def test_ref2va_rejects_unknown_reference_size(self):
        workflow = self.workflow()
        with self.assertRaisesRegex(Exception, "reference_size"):
            _configure_ref2va_workflow(workflow, {"reference_size": "huge"})


if __name__ == "__main__":
    unittest.main()

import copy
import json
import unittest
from pathlib import Path

from runtime_controls import apply_minimax_runtime_options


class RuntimeControlTests(unittest.TestCase):
    def workflow(self):
        return json.loads(Path("api-workflow-minimax.json").read_text(encoding="utf-8"))

    def test_defaults_bypass_new_optional_loras_and_leave_upscale_off(self):
        workflow = self.workflow()
        applied = apply_minimax_runtime_options(workflow, {})
        self.assertEqual(applied["turbo_strength"], 0.85)
        self.assertEqual(applied["m3_strength"], 0.5)
        self.assertEqual(applied["cumshot_strength"], 0.7)
        for key in (
            "realism_strength", "deepthroat_strength", "civ3210503_strength",
            "civ3320641_strength", "pussy4nus_strength", "fingering_strength",
            "moawxx_strength", "naughtytimes_strength",
            "astro_strength", "icy_real_strength", "hogtied_strength",
            "upskirt_strength", "all_tied_up_strength", "doggy_pov_strength",
            "after_midnight_strength",
        ):
            self.assertEqual(applied[key], 0.0)
        for node_id in ("410", "411", "412", "413", "414", "415", "416", "417",
                        "420", "421", "422", "423", "424", "425"):
            self.assertNotIn(node_id, workflow)
        self.assertFalse(applied["enable_ai_upscale"])
        self.assertNotIn("405", workflow)
        self.assertEqual(applied["output_width"], 544)
        self.assertEqual(applied["output_height"], 960)

    def test_quality_modes_patch_h3_canvas_on_32_pixel_grid(self):
        expected = {
            "fast": (544, 960),
            "balanced": (672, 1184),
            "quality": (768, 1344),
        }
        for mode, size in expected.items():
            with self.subTest(mode=mode):
                workflow = self.workflow()
                applied = apply_minimax_runtime_options(workflow, {"quality_mode": mode})
                self.assertEqual((workflow["364"]["inputs"]["width"], workflow["364"]["inputs"]["height"]), size)
                self.assertEqual((workflow["350"]["inputs"]["width"], workflow["350"]["inputs"]["height"]), size)
                self.assertEqual((applied["generation_width"], applied["generation_height"]), size)
                self.assertEqual(size[0] % 32, 0)
                self.assertEqual(size[1] % 32, 0)

    def test_new_optional_lora_can_be_selected_without_loading_others(self):
        workflow = self.workflow()
        applied = apply_minimax_runtime_options(workflow, {
            "astro_strength": 0.65, "doggy_pov_strength": 0.8,
        })
        self.assertEqual(workflow["420"]["inputs"]["strength_model"], 0.65)
        self.assertEqual(workflow["425"]["inputs"]["model"], ["420", 0])
        self.assertEqual(workflow["394"]["inputs"]["model"], ["425", 0])
        self.assertNotIn("421", workflow)
        self.assertEqual(applied["doggy_pov_strength"], 0.8)

    def test_zero_strength_really_bypasses_lora_nodes(self):
        workflow = self.workflow()
        applied = apply_minimax_runtime_options(
            workflow,
            {
                "turbo_strength": 0,
                "m3_strength": 0.5,
                "mystic_strength": 0,
                "hmnsfw_strength": 0,
                "vagassist_strength": 0,
                "hmpussy_strength": 0,
                "cumshot_strength": 0,
                "fingering_strength": 0.9,
                "naughtytimes_strength": 0.7,
            },
        )
        for node_id in ("390", "392", "393", "400", "401", "402"):
            self.assertNotIn(node_id, workflow)
        self.assertEqual(workflow["391"]["inputs"]["model"], ["384", 0])
        self.assertEqual(workflow["415"]["inputs"]["model"], ["391", 0])
        self.assertEqual(workflow["417"]["inputs"]["model"], ["415", 0])
        self.assertEqual(workflow["394"]["inputs"]["model"], ["417", 0])
        self.assertEqual(applied["fingering_strength"], 0.9)

    def test_ai_upscale_switch_enhances_photo_and_video(self):
        workflow = self.workflow()
        applied = apply_minimax_runtime_options(
            workflow, {"enable_ai_upscale": True, "enable_gimm": False}
        )
        self.assertTrue(applied["enable_ai_upscale"])
        self.assertEqual(workflow["405"]["class_type"], "UpscaleModelLoader")
        self.assertEqual(workflow["350"]["inputs"]["image"], ["406", 0])
        self.assertEqual(workflow["408"]["inputs"]["image"], ["374", 0])
        self.assertEqual(workflow["370"]["inputs"]["images"], ["408", 0])
        self.assertEqual(workflow["370"]["inputs"]["fps"], 24.0)
        self.assertEqual(applied["output_width"], 1088)
        self.assertEqual(applied["output_height"], 1920)

    def test_upscale_off_routes_native_frames_through_optional_gimm(self):
        workflow = self.workflow()
        apply_minimax_runtime_options(
            workflow, {"enable_ai_upscale": False, "enable_gimm": True}
        )
        self.assertNotIn("405", workflow)
        self.assertEqual(workflow["350"]["inputs"]["image"], ["395", 0])
        self.assertEqual(workflow["399"]["inputs"]["images"], ["374", 0])
        self.assertEqual(workflow["370"]["inputs"]["images"], ["399", 0])
        self.assertEqual(workflow["370"]["inputs"]["fps"], 48.0)

    def test_rejects_out_of_range_values(self):
        for payload in (
            {"m3_strength": -0.1},
            {"fingering_strength": 2.1},
            {"steps": 3},
            {"steps": 6.5},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                apply_minimax_runtime_options(self.workflow(), payload)


if __name__ == "__main__":
    unittest.main()

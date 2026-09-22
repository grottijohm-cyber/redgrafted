import copy
import json
import unittest
from pathlib import Path

from runtime_controls import apply_minimax_runtime_options


class RuntimeControlTests(unittest.TestCase):
    def workflow(self):
        return json.loads(Path("api-workflow-minimax.json").read_text(encoding="utf-8"))

    def test_defaults_preserve_bundled_settings(self):
        workflow = self.workflow()
        applied = apply_minimax_runtime_options(workflow, {})
        self.assertEqual(applied["turbo_strength"], 0.85)
        self.assertEqual(applied["m3_strength"], 0.5)
        self.assertEqual(applied["mystic_strength"], 1.0)
        self.assertEqual(applied["hmnsfw_strength"], 1.0)
        self.assertEqual(applied["vagassist_strength"], 1.0)
        self.assertEqual(applied["hmpussy_strength"], 0.35)
        self.assertEqual(applied["cumshot_strength"], 0.7)
        self.assertEqual(applied["steps"], 8)

    def test_each_slider_patches_only_the_job_workflow(self):
        original = self.workflow()
        workflow = copy.deepcopy(original)
        applied = apply_minimax_runtime_options(
            workflow,
            {
                "turbo_strength": 0.7,
                "m3_strength": 0.25,
                "mystic_strength": 0,
                "hmnsfw_strength": 0.4,
                "vagassist_strength": 0.6,
                "hmpussy_strength": 0.2,
                "cumshot_strength": 0.5,
                "steps": 6,
            },
        )
        self.assertEqual(workflow["390"]["inputs"]["strength_model"], 0.7)
        self.assertEqual(workflow["391"]["inputs"]["strength_model"], 0.25)
        self.assertEqual(workflow["392"]["inputs"]["strength_model"], 0.0)
        self.assertEqual(workflow["393"]["inputs"]["strength_model"], 0.4)
        self.assertEqual(workflow["400"]["inputs"]["strength_model"], 0.6)
        self.assertEqual(workflow["401"]["inputs"]["strength_model"], 0.2)
        self.assertEqual(workflow["402"]["inputs"]["strength_model"], 0.5)
        self.assertEqual(workflow["397"]["inputs"]["steps"], 6)
        self.assertEqual(applied["steps"], 6)
        self.assertEqual(original["397"]["inputs"]["steps"], 8)

    def test_rejects_out_of_range_values(self):
        for payload in ({"m3_strength": -0.1}, {"mystic_strength": 1.6}, {"steps": 3}, {"steps": 6.5}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                apply_minimax_runtime_options(self.workflow(), payload)


if __name__ == "__main__":
    unittest.main()

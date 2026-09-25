import copy
import base64
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

import model_setup
import worker
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
                patch.object(model_setup, "COMFY_MODELS", root / "models"),
            ):
                status = model_setup.model_status()
                self.assertFalse(status["reference_assets_present"])
                self.assertEqual(
                    status["missing_reference_files"],
                    [item.relative_path for item in reference_files],
                )

                for item in reference_files:
                    target = root / "models" / item.relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"x")

                status = model_setup.model_status()
                self.assertTrue(status["reference_assets_present"])
                self.assertEqual(status["missing_reference_files"], [])

    def test_missing_reference_assets_download_from_pinned_official_repo_on_demand(self):
        reference_files = tuple(
            replace(item, min_bytes=1) for item in model_setup.MINIMAX_REFERENCE_FILES
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "old-bundle"
            snapshot.mkdir()
            with (
                patch.object(model_setup, "MODEL_PROFILE", "minimax"),
                patch.object(model_setup, "BOOTSTRAP_MODE", False),
                patch.object(model_setup, "MINIMAX_REFERENCE_FILES", reference_files),
                patch.object(model_setup, "MINIMAX_REFERENCE_PATHS", tuple(x.relative_path for x in reference_files)),
                patch.object(model_setup, "COMFY_MODELS", root / "models"),
                patch.object(model_setup, "_latest_snapshot", return_value=snapshot),
                patch.object(model_setup, "_check_disk") as check_disk,
                patch.object(model_setup, "_download_many") as download_many,
            ):
                model_setup.ensure_reference_models()

            check_disk.assert_called_once_with(reference_files, None)
            download_many.assert_called_once_with(reference_files, None)
            self.assertTrue(all("huggingface.co/Comfy-Org/MiniMax-H3/resolve/" in x.url
                                for x in reference_files))

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

    def test_reference_workflow_has_valid_links_and_prepareable_lora_sources(self):
        workflow = self.workflow()
        options = {name: 0 for name in LORA_OPTIONS}
        options.update({"all_tied_up_strength": 0.7, "after_midnight_strength": 0.8})
        apply_minimax_runtime_options(workflow, options)
        _configure_ref2va_workflow(workflow, options)
        for node_id, node in workflow.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    self.assertIn(str(value[0]), workflow, f"node {node_id} has a broken link")
        active = {node["inputs"]["lora_name"] for node in workflow.values()
                  if node["class_type"] == "LoraLoaderModelOnly"}
        self.assertIn("all-tied-up-mh3-e70-az420.safetensors", active)
        self.assertIn("AfterMidnight_ref2va_h3_sexytime_rank64-v1.2.safetensors", active)
        bundled = {Path(item.relative_path).name: item for item in model_setup.BUNDLED_H3_LORAS}
        self.assertTrue(active & bundled.keys())
        for name in active & bundled.keys():
            self.assertEqual(len(bundled[name].expected_sha256), 64)
        tied = bundled["all-tied-up-mh3-e70-az420.safetensors"]
        self.assertEqual(tied.token_env, "HF_TOKEN")
        self.assertIn("grottijohm/redgraft-ltx25-runpod/resolve/354c0f2702214c226d6dd0148944e6bb37fb9d86/",
                      tied.url)
        self.assertEqual(tied.expected_sha256,
                         "f87bb957cdee03716bbeaf06bca3e2c33c45db4a28b8da508d0ddeae1b425ad8")

    def test_reference_request_prepares_assets_and_reaches_comfy_queue(self):
        picture = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(picture, format="PNG")
        image = "data:image/png;base64," + base64.b64encode(picture.getvalue()).decode("ascii")
        with patch.object(model_setup, "MODEL_PROFILE", "minimax"), \
                patch.object(model_setup, "MODEL_FILES", model_setup.MINIMAX_FILES), \
                patch.object(model_setup, "ALL_MODEL_PATHS", tuple(item.relative_path for item in model_setup.MINIMAX_FILES)), \
                patch.object(model_setup, "BOOTSTRAP_MODE", False), \
                patch.object(worker, "WORKFLOW_PATH", Path("api-workflow-minimax.json")), \
                patch.object(worker, "_bucket_configuration"), \
                patch.object(worker, "ensure_models") as base_models, \
                patch.object(model_setup, "ensure_selected_loras") as optional_loras, \
                patch.object(model_setup, "ensure_reference_models") as reference_models, \
                patch.object(worker, "wait_for_comfyui"), \
                patch.object(worker, "upload_input_image", return_value="test.png"), \
                patch.object(worker, "queue_workflow", return_value="prompt-1") as queue, \
                patch.object(worker, "wait_for_history", return_value={"outputs": {}}), \
                patch.object(worker, "get_output_descriptors", return_value=[{"filename": "video.mp4"}]), \
                patch.object(worker, "_safe_output_path", return_value=Path("/tmp/ref2v-test.mp4")), \
                patch.object(worker, "publish_output", return_value={
                    "filename": "video.mp4", "url": "https://example.com/video.mp4",
                    "type": "url", "mime_type": "video/mp4"}):
            result = worker.handle_job({"id": "ref2v-test", "input": {
                "image": image, "reference_images": [image], "prompt": "A steady shot",
                "generation_mode": "reference", "all_tied_up_strength": 0.7,
                "after_midnight_strength": 0.8}})
        self.assertEqual(result.get("status"), "success", result)
        base_models.assert_called_once()
        reference_models.assert_called_once()
        optional_loras.assert_called_once()
        workflow = queue.call_args.args[0]
        self.assertEqual(workflow["364"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(workflow["364"]["inputs"]["ref_images.ref_image_1"], ["419", 0])
        selected = {node["inputs"]["lora_name"] for node in workflow.values()
                    if node["class_type"] == "LoraLoaderModelOnly"}
        self.assertIn("all-tied-up-mh3-e70-az420.safetensors", selected)
        self.assertIn("AfterMidnight_ref2va_h3_sexytime_rank64-v1.2.safetensors", selected)


if __name__ == "__main__":
    unittest.main()

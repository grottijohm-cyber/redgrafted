import ast
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import model_setup
import worker


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(output, format="PNG")
    return output.getvalue()


class WorkerTests(unittest.TestCase):
    def test_runpod_entrypoint_is_detectable(self):
        source = Path("handler.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        handlers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "handler"
        ]
        self.assertEqual(len(handlers), 1)
        self.assertIn('runpod.serverless.start({"handler": handler})', source)

    def test_workflow_is_native_ltx25_i2v_api_graph(self):
        workflow = worker.load_workflow(Path("api-workflow.json"))
        self.assertEqual(workflow["395"]["class_type"], "LoadImage")
        self.assertNotIn("380", workflow)
        self.assertNotIn("393", workflow)
        self.assertEqual(workflow["364"]["class_type"], "CLIPTextEncode")
        self.assertIsInstance(workflow["364"]["inputs"]["text"], str)
        self.assertEqual(workflow["364"]["inputs"]["clip"], ["387", 0])
        self.assertEqual(workflow["356"]["inputs"]["length"], 241)
        self.assertEqual(workflow["366"]["inputs"]["frames_number"], 241)
        self.assertEqual(workflow["75"]["class_type"], "SaveVideo")
        self.assertEqual(workflow["356"]["inputs"]["length"] % 8, 1)
        self.assertEqual(workflow["356"]["inputs"]["width"] % 32, 0)
        self.assertEqual(workflow["356"]["inputs"]["height"] % 32, 0)
        self.assertEqual(workflow["370"]["inputs"]["fps"], 24.0)

        class_types = {node["class_type"] for node in workflow.values()}
        self.assertNotIn("VHS_VideoCombine", class_types)
        self.assertNotIn("PrimitiveStringMultiline", class_types)
        self.assertNotIn("ComfySwitchNode", class_types)
        self.assertNotIn("TextGenerateLTX2Prompt", class_types)

    def test_every_workflow_link_points_to_an_existing_node(self):
        workflow = json.loads(Path("api-workflow.json").read_text(encoding="utf-8"))
        for node in workflow.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2:
                    self.assertIn(str(value[0]), workflow)

    def test_workflow_model_names_are_downloaded(self):
        workflow = json.loads(Path("api-workflow.json").read_text(encoding="utf-8"))
        workflow_names = {
            Path(value).name
            for node in workflow.values()
            for value in node["inputs"].values()
            if isinstance(value, str) and value.endswith(".safetensors")
        }
        download_names = {
            Path(item.relative_path).name for item in model_setup.MODEL_FILES
        }
        self.assertEqual(len(model_setup.MODEL_FILES), 5)
        self.assertEqual(workflow_names, download_names)
        self.assertIn(
            "redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
            download_names,
        )
        self.assertNotIn("gemma-3-12b-it-heretic-v2_int8.safetensors", download_names)
        self.assertNotIn("ltx-2.3_text_projection_bf16.safetensors", download_names)



    def test_both_sampling_passes_use_the_reported_checkpoint(self):
        workflow = worker.load_workflow(Path("api-workflow.json"))
        for sampler, guider in (("344", "388"), ("368", "391")):
            self.assertEqual(workflow[sampler]["inputs"]["guider"], [guider, 0])
            self.assertEqual(workflow[guider]["inputs"]["model"], ["384", 0])
            self.assertEqual(workflow[guider]["inputs"]["positive"], ["365", 0])
        self.assertEqual(workflow["365"]["inputs"]["positive"], ["364", 0])
        info = worker.workflow_details(workflow)
        self.assertEqual(info["checkpoint"], workflow["384"]["inputs"]["unet_name"])
        self.assertFalse(info["prompt_enhanced"])

    def test_rejects_unsynchronized_duration_before_generation(self):
        changes = [
            ("366", "frames_number", 121),
            ("366", "frame_rate", 25),
            ("365", "frame_rate", 25),
            ("370", "fps", 0),
            ("356", "length", 240),
        ]
        for node, field, value in changes:
            with self.subTest(node=node, field=field), tempfile.TemporaryDirectory() as directory:
                workflow = json.loads(Path("api-workflow.json").read_text())
                workflow[node]["inputs"][field] = value
                path = Path(directory) / "workflow.json"
                path.write_text(json.dumps(workflow))
                with self.assertRaises(worker.WorkerError):
                    worker.load_workflow(path)

    def test_cannot_report_a_checkpoint_disconnected_from_the_sampler(self):
        workflow = worker.load_workflow(Path("api-workflow.json"))
        workflow["391"]["inputs"]["model"] = ["387", 0]
        with self.assertRaisesRegex(worker.WorkerError, "Both sampling passes"):
            worker.workflow_details(workflow)

    def test_decodes_raw_base64_and_data_uri(self):
        data = png_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        self.assertEqual(worker._decode_image_input(encoded), data)
        self.assertEqual(
            worker._decode_image_input(f"data:image/png;base64,{encoded}"), data
        )

    def test_rejects_non_image_bytes(self):
        with self.assertRaises(worker.WorkerError):
            worker._validate_image(b"not an image")

    @patch.object(worker, "publish_output")
    @patch.object(worker, "_safe_output_path")
    @patch.object(worker, "get_output_descriptors")
    @patch.object(worker, "wait_for_history")
    @patch.object(worker, "queue_workflow")
    @patch.object(worker, "upload_input_image")
    @patch.object(worker, "wait_for_comfyui")
    @patch.object(worker, "ensure_models")
    def test_job_patches_image_prompt_and_random_seed(
        self,
        ensure_models,
        wait_for_comfyui,
        upload_input_image,
        queue_workflow,
        wait_for_history,
        get_output_descriptors,
        safe_output_path,
        publish_output,
    ):
        upload_input_image.return_value = "job.png"
        queue_workflow.return_value = "prompt-1"
        wait_for_history.return_value = {"outputs": {}}
        get_output_descriptors.return_value = [{"filename": "result.mp4"}]
        safe_output_path.return_value = Path("/tmp/result.mp4")
        publish_output.return_value = {
            "filename": "result.mp4",
            "type": "url",
            "url": "https://example.com/result.mp4",
            "mime_type": "video/mp4",
        }

        prepared_prompt = "The subject turns toward the camera.\nA red scarf moves gently in the wind."
        with patch.object(worker, "WORKFLOW_PATH", Path("api-workflow.json")):
            result = worker.handle_job(
                {
                    "id": "job-1",
                    "input": {
                        "image": "data:image/png;base64,"
                        + base64.b64encode(png_bytes()).decode("ascii"),
                        "prompt": prepared_prompt,
                    },
                }
            )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["prompt_enhanced"])
        self.assertEqual(result["workflow"]["frames"], 241)
        self.assertEqual(result["workflow"]["fps"], 24.0)
        self.assertAlmostEqual(result["workflow"]["duration_seconds"], 10.042, places=3)
        self.assertEqual(result["workflow"]["checkpoint"],
                         "redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors")
        queued = queue_workflow.call_args.args[0]
        self.assertEqual(queued["395"]["inputs"]["image"], "job.png")
        self.assertEqual(
            queued["364"]["inputs"]["text"],
            prepared_prompt,
        )
        self.assertIsInstance(queued["339"]["inputs"]["noise_seed"], int)
        self.assertNotIn("380", queued)
        self.assertNotIn("393", queued)

    def test_rejects_extra_inputs(self):
        result = worker.handle_job(
            {"input": {"image": "abc", "prompt": "valid", "seed": 1}}
        )
        self.assertIn("Unsupported generation input", result["error"])

    def test_requires_prompt(self):
        result = worker.handle_job({"input": {"image": "abc"}})
        self.assertIn("input.prompt", result["error"])


if __name__ == "__main__":
    unittest.main()

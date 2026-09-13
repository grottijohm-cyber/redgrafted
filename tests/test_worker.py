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
        self.assertEqual(workflow["380"]["class_type"], "TextGenerateLTX2Prompt")
        self.assertEqual(workflow["393"]["class_type"], "DualCLIPLoader")
        self.assertEqual(workflow["75"]["class_type"], "SaveVideo")
        self.assertEqual(workflow["356"]["inputs"]["length"] % 8, 1)
        self.assertEqual(workflow["356"]["inputs"]["width"] % 32, 0)
        self.assertEqual(workflow["356"]["inputs"]["height"] % 32, 0)
        self.assertEqual(workflow["370"]["inputs"]["fps"], 24.0)

        class_types = {node["class_type"] for node in workflow.values()}
        self.assertNotIn("VHS_VideoCombine", class_types)
        self.assertNotIn("PrimitiveStringMultiline", class_types)
        self.assertNotIn("ComfySwitchNode", class_types)

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
        self.assertEqual(len(model_setup.MODEL_FILES), 7)
        self.assertTrue(workflow_names <= download_names)
        self.assertIn(
            "redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
            download_names,
        )
        self.assertIn("gemma-3-12b-it-heretic-v2_int8.safetensors", download_names)


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

        with patch.object(worker, "WORKFLOW_PATH", Path("api-workflow.json")):
            result = worker.handle_job(
                {
                    "id": "job-1",
                    "input": {
                        "image": "data:image/png;base64,"
                        + base64.b64encode(png_bytes()).decode("ascii"),
                        "prompt": "The subject turns toward the camera.",
                    },
                }
            )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["prompt_enhanced"])
        queued = queue_workflow.call_args.args[0]
        self.assertEqual(queued["395"]["inputs"]["image"], "job.png")
        self.assertEqual(
            queued["380"]["inputs"]["prompt"],
            "The subject turns toward the camera.",
        )
        self.assertIsInstance(queued["339"]["inputs"]["noise_seed"], int)

    def test_rejects_extra_inputs(self):
        result = worker.handle_job(
            {"input": {"image": "abc", "prompt": "valid", "seed": 1}}
        )
        self.assertIn("Only input.image and input.prompt", result["error"])

    def test_requires_prompt(self):
        result = worker.handle_job({"input": {"image": "abc"}})
        self.assertIn("input.prompt", result["error"])


if __name__ == "__main__":
    unittest.main()

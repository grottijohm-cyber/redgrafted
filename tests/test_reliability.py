import base64
import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from PIL import Image

import file_integrity
import bootstrap_hf_repo
import model_setup
import worker


def tensor_bytes(data=b"abcd"):
    header = json.dumps({"x": {"dtype": "U8", "shape": [len(data)],
                                "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


class Response:
    def __init__(self, body=b"", status=200, headers=None):
        self.body, self.status_code, self.headers = body, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise worker.requests.HTTPError("test HTTP error")

    def iter_content(self, chunk_size):
        yield self.body


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "model.safetensors"
        self.data = tensor_bytes()

    def partial(self, data):
        p = self.target.with_name(self.target.name + ".part")
        p.write_bytes(data)
        p.with_name(p.name + ".etag").write_text('"version-one"')

    def full_response(self):
        return Response(self.data, headers={"Content-Length": str(len(self.data)), "ETag": '"version-one"'})

    def download(self, responses):
        with patch.object(file_integrity.requests, "get", side_effect=responses) as get:
            with patch.object(file_integrity.time, "sleep"):
                file_integrity.download_model("https://example.com/model", self.target, 10, {})
        self.assertEqual(self.target.read_bytes(), self.data)
        return get

    def test_truncated_tensor_rejected_even_above_minimum_size(self):
        self.target.write_bytes(self.data[:-1])
        with self.assertRaises(file_integrity.IntegrityError):
            file_integrity.validate_safetensors(self.target, 10)

    def test_valid_range_is_resumed(self):
        self.partial(self.data[:20])
        get = self.download([Response(self.data[20:], 206, {
            "Content-Range": f"bytes 20-{len(self.data)-1}/{len(self.data)}",
            "Content-Length": str(len(self.data)-20), "ETag": '"version-one"'})])
        self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=20-")

    def test_wrong_range_restarts_instead_of_appending(self):
        self.partial(self.data[:20])
        get = self.download([Response(self.data[10:], 206, {
            "Content-Range": f"bytes 10-{len(self.data)-1}/{len(self.data)}"}), self.full_response()])
        self.assertEqual(get.call_count, 2)
        self.assertNotIn("Range", get.call_args.kwargs["headers"])

    def test_server_ignoring_range_replaces_partial(self):
        self.partial(self.data[:20])
        self.download([self.full_response()])

    def test_finished_partial_is_validated_on_416(self):
        self.partial(self.data)
        self.download([Response(status=416, headers={"Content-Range": f"bytes */{len(self.data)}"})])

    def test_short_response_is_not_promoted(self):
        self.download([Response(self.data[:-1], headers={"Content-Length": str(len(self.data))}),
                       self.full_response()])

    def test_cached_bundle_digest_detects_same_size_corruption(self):
        root = Path(self.directory.name)
        snapshot, models = root / "snapshot", root / "models"
        snapshot.mkdir()
        data = tensor_bytes(b"abcd")
        (snapshot / "a.safetensors").write_bytes(tensor_bytes(b"abce"))
        (snapshot / "bundle-manifest.json").write_text(json.dumps({"files": {
            "a.safetensors": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}))
        with patch.object(model_setup, "COMFY_MODELS", models):
            with self.assertRaises(model_setup.ModelSetupError):
                model_setup._link_from_snapshot(snapshot, ("a.safetensors",))
        self.assertFalse(models.exists())

    def test_missing_file_does_not_partially_link_bundle(self):
        root = Path(self.directory.name)
        snapshot, models = root / "snapshot", root / "models"
        snapshot.mkdir()
        (snapshot / "a.safetensors").write_bytes(self.data)
        with patch.object(model_setup, "COMFY_MODELS", models):
            with self.assertRaises(model_setup.ModelSetupError):
                model_setup._link_from_snapshot(snapshot, ("a.safetensors", "missing.safetensors"))
        self.assertFalse(models.exists())


    def test_old_bundle_manifest_remains_usable_without_enhancer_files(self):
        root = Path(self.directory.name)
        snapshot, models = root / "snapshot", root / "models"
        snapshot.mkdir()
        records = {}
        for relative in model_setup.ALL_MODEL_PATHS:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.data)
            records[relative] = {"size": len(self.data),
                                 "sha256": hashlib.sha256(self.data).hexdigest()}
        # Legacy records must not require loading or even mounting the retired files.
        for relative in ("text_encoders/gemma-3-12b-it-heretic-v2_int8.safetensors",
                         "text_encoders/ltx-2.3_text_projection_bf16.safetensors"):
            records[relative] = {"size": 10000000000, "sha256": "old-unneeded-record"}
        (snapshot / "bundle-manifest.json").write_text(json.dumps({"files": records}))
        small_files = tuple(model_setup.ModelFile(item.url, item.relative_path, 1, item.token_env)
                            for item in model_setup.MODEL_FILES)
        with patch.object(model_setup, "COMFY_MODELS", models), \
                patch.object(model_setup, "MODEL_FILES", small_files):
            model_setup._link_from_snapshot(snapshot, model_setup.ALL_MODEL_PATHS)
        for relative in model_setup.ALL_MODEL_PATHS:
            self.assertTrue((models / relative).is_symlink())
        self.assertFalse((models / "text_encoders/gemma-3-12b-it-heretic-v2_int8.safetensors").exists())


@contextlib.contextmanager
def generation_mocks():
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        root = Path(directory)
        input_file, output_file = root / "input.png", root / "output.mp4"
        image = io.BytesIO()
        Image.new("RGB", (4, 4), "blue").save(image, format="PNG")
        input_file.write_bytes(image.getvalue())
        output_file.write_bytes(b"video")
        stack.enter_context(patch.dict(os.environ, {}, clear=True))
        stack.enter_context(patch.object(model_setup, "BOOTSTRAP_MODE", False))
        for name, value in {
            "ensure_models": None, "wait_for_comfyui": None,
            "upload_input_image": "input.png", "_safe_input_path": input_file,
            "queue_workflow": "prompt-1", "wait_for_history": {"outputs": {}},
            "get_output_descriptors": [{"filename": "output.mp4"}],
            "_safe_output_path": output_file,
        }.items():
            stack.enter_context(patch.object(worker, name, return_value=value))
        stack.enter_context(patch.object(worker, "WORKFLOW_PATH", Path("api-workflow.json")))
        job = {"id": "job-1", "input": {"image": base64.b64encode(image.getvalue()).decode(),
                "prompt": "Clouds move slowly over a mountain."}}
        yield job, input_file, output_file


class JobTests(unittest.TestCase):
    def test_delivery_failure_preserves_generated_video(self):
        with generation_mocks() as (job, input_file, output_file):
            with patch.object(worker, "publish_output", side_effect=worker.WorkerError("upload failed")):
                result = worker.handle_job(job)
            self.assertIn("upload failed", result["error"])
            self.assertTrue(output_file.is_file())
            self.assertFalse(input_file.exists())

    def test_success_cleans_files_after_serializing_video(self):
        with generation_mocks() as (job, input_file, output_file):
            result = worker.handle_job(job)
            self.assertEqual(result["status"], "success")
            self.assertEqual(base64.b64decode(result["videos"][0]["data"]), b"video")
            self.assertFalse(output_file.exists())
            self.assertFalse(input_file.exists())

    def test_timeout_cancels_prompt_and_refreshes_worker(self):
        with generation_mocks() as (job, _, __):
            with patch.object(worker, "wait_for_history", side_effect=TimeoutError("budget exceeded")):
                with patch.object(worker, "cancel_workflow") as cancel:
                    result = worker.handle_job(job)
            cancel.assert_called_once_with("prompt-1")
            self.assertTrue(result["refresh_worker"])

    def test_preparation_and_generation_share_deadline(self):
        with generation_mocks() as (job, _, __):
            worker.handle_job(job)
            deadline = worker.ensure_models.call_args.args[0]
            self.assertEqual(deadline, worker.wait_for_comfyui.call_args.args[0])
            self.assertEqual(deadline, worker.wait_for_history.call_args.args[1])

    def test_incomplete_bucket_fails_before_model_setup(self):
        with generation_mocks() as (job, _, __):
            with patch.dict(os.environ, {"BUCKET_NAME": "test"}):
                result = worker.handle_job(job)
            self.assertIn("configuration is incomplete", result["error"])
            worker.ensure_models.assert_not_called()

    def test_setup_job_waits_for_bootstrap_and_returns_result(self):
        with patch("bootstrap_hf_repo.bootstrap_bundle", return_value={"status": "setup_complete"}) as prepare:
            result = worker.handle_job({"input": {"action": "setup"}})
        prepare.assert_called_once()
        self.assertEqual(result["status"], "setup_complete")

    def test_setup_failure_is_returned_to_queue(self):
        with patch("bootstrap_hf_repo.bootstrap_bundle", side_effect=model_setup.ModelSetupError("access denied")):
            result = worker.handle_job({"input": {"action": "setup"}})
        self.assertIn("access denied", result["error"])

    def test_status_does_not_download_models(self):
        with patch.object(model_setup, "model_status", return_value={"files_ready": False}):
            with patch.object(worker, "ensure_models") as prepare, \
                    patch.object(worker, "WORKFLOW_PATH", Path("api-workflow.json")):
                result = worker.handle_job({"input": {"action": "status"}})
        prepare.assert_not_called()
        self.assertFalse(result["models"]["files_ready"])
        self.assertEqual(result["workflow"]["frames"], 241)
        self.assertFalse(result["workflow"]["prompt_enhanced"])

    def test_setup_mode_prevents_accidental_generation(self):
        with generation_mocks() as (job, _, __):
            with patch.object(model_setup, "BOOTSTRAP_MODE", True):
                result = worker.handle_job(job)
            self.assertIn("one-time setup mode", result["error"])
            worker.ensure_models.assert_not_called()


class BundlePublicationTests(unittest.TestCase):
    def test_complete_bundle_and_manifest_are_published_in_one_commit(self):
        api = Mock()
        api.repo_info.return_value.private = True
        api.create_commit.return_value.oid = "commit-123"
        hub = types.SimpleNamespace(HfApi=Mock(return_value=api),
                                    CommitOperationAdd=lambda **kwargs: types.SimpleNamespace(**kwargs))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = model_setup.ALL_MODEL_PATHS
            for relative in paths:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tensor_bytes())
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {"huggingface_hub": hub}))
                stack.enter_context(patch.dict(os.environ, {"HF_WRITE_TOKEN": "test-only-token"}))
                stack.enter_context(patch.object(model_setup, "BOOTSTRAP_MODE", True))
                stack.enter_context(patch.object(model_setup, "COMFY_MODELS", root))
                stack.enter_context(patch.object(model_setup, "ensure_models"))
                result = bootstrap_hf_repo.bootstrap_bundle()
            api.create_commit.assert_called_once()
            operations = api.create_commit.call_args.kwargs["operations"]
            self.assertEqual({op.path_in_repo for op in operations},
                             set(paths) | {"README.md", "bundle-manifest.json"})
            manifest = next(op for op in operations if op.path_in_repo == "bundle-manifest.json")
            files = json.loads(manifest.path_or_fileobj.getvalue())["files"]
            for relative in paths:
                self.assertEqual(files[relative]["sha256"], hashlib.sha256(tensor_bytes()).hexdigest())
                self.assertEqual(files[relative]["size"], len(tensor_bytes()))
            self.assertEqual(result["revision"], "commit-123")
            self.assertEqual(result["status"], "setup_complete")

    def test_public_repository_is_rejected_before_model_preparation_or_upload(self):
        api = Mock()
        api.repo_info.return_value.private = False
        hub = types.SimpleNamespace(HfApi=Mock(return_value=api), CommitOperationAdd=Mock())
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"huggingface_hub": hub}))
            stack.enter_context(patch.dict(os.environ, {"HF_WRITE_TOKEN": "test-only-token"}))
            stack.enter_context(patch.object(model_setup, "BOOTSTRAP_MODE", True))
            prepare = stack.enter_context(patch.object(model_setup, "ensure_models"))
            with self.assertRaisesRegex(model_setup.ModelSetupError, "must be private"):
                bootstrap_hf_repo.bootstrap_bundle()
        prepare.assert_not_called()
        api.create_commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()

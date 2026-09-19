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
from unittest.mock import Mock, patch

import bootstrap_hf_repo
import model_setup
import worker


def tensor_bytes(payload=b"abcd"):
    header = json.dumps({"weight": {"dtype": "U8", "shape": [len(payload)],
                                     "data_offsets": [0, len(payload)]}}).encode()
    return struct.pack("<Q", len(header)) + header + payload


@contextlib.contextmanager
def ten_eros_profile():
    with patch.object(model_setup, "MODEL_PROFILE", "10eros"), \
            patch.object(model_setup, "MODEL_FILES", model_setup.TEN_EROS_FILES), \
            patch.object(model_setup, "ALL_MODEL_PATHS", tuple(x.relative_path for x in model_setup.TEN_EROS_FILES)):
        yield


class ModelProfileTests(unittest.TestCase):
    def test_10eros_uses_its_own_conditioning_and_vaes_in_both_passes(self):
        with ten_eros_profile():
            graph = worker.load_workflow(Path("api-workflow-10eros.json"))
            info = worker.workflow_details(graph)
            self.assertEqual(info["model_profile"], "10eros")
            self.assertEqual(info["frames"], 241)
            self.assertEqual(info["fps"], 24)
            self.assertFalse(info["prompt_enhanced"])
            for node_id in ("357", "349", "348", "374"):
                self.assertEqual(graph[node_id]["inputs"]["vae"], ["384", 2])
            for node_id in ("388", "391"):
                self.assertEqual(graph[node_id]["inputs"]["model"], ["384", 0])
            self.assertEqual(graph["387"]["inputs"]["ckpt_name"], info["checkpoint"])
            self.assertEqual(graph["386"]["inputs"]["ckpt_name"], info["checkpoint"])
            self.assertEqual(graph["364"]["class_type"], "CLIPTextEncode")
            self.assertFalse(any("LoraLoader" in n["class_type"] for n in graph.values()))

    def test_cross_version_workflow_override_is_rejected(self):
        with ten_eros_profile():
            with self.assertRaisesRegex(worker.WorkerError, "do not match MODEL_PROFILE"):
                worker.load_workflow(Path("api-workflow.json"))
        with self.assertRaisesRegex(worker.WorkerError, "do not match MODEL_PROFILE"):
            worker.load_workflow(Path("api-workflow-10eros.json"))

    def test_wrong_vae_or_double_distillation_is_rejected(self):
        with ten_eros_profile():
            graph = worker.load_workflow(Path("api-workflow-10eros.json"))
            graph["374"]["inputs"]["vae"] = ["386", 0]
            with self.assertRaisesRegex(worker.WorkerError, "video VAE"):
                worker.workflow_details(graph)
            graph["374"]["inputs"]["vae"] = ["384", 2]
            graph["extra"] = {"class_type": "LoraLoaderModelOnly", "inputs": {}}
            with self.assertRaisesRegex(worker.WorkerError, "already includes"):
                worker.workflow_details(graph)

    def test_upstream_digest_rejects_complete_but_wrong_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            target.write_bytes(tensor_bytes(b"abce"))
            item = model_setup.ModelFile("https://example.com/model", target.name, 1,
                                         expected_sha256=hashlib.sha256(tensor_bytes()).hexdigest())
            self.assertFalse(model_setup._ready(target, item))
            with self.assertRaisesRegex(model_setup.IntegrityError, "pinned upstream"):
                model_setup._validate_model(target, item)

    def test_setup_reuses_existing_files_and_fetches_only_missing_profile_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, models = root / "snapshot", root / "models"
            snapshot.mkdir()
            (snapshot / "existing.safetensors").write_bytes(tensor_bytes())
            items = tuple(model_setup.ModelFile("https://example.com/" + name, name, 1)
                          for name in ("existing.safetensors", "new.safetensors"))

            def fetch(url, target, *_):
                target.write_bytes(tensor_bytes())

            with patch.object(model_setup, "MODEL_PROFILE", "10eros"), \
                    patch.object(model_setup, "BOOTSTRAP_MODE", True), \
                    patch.object(model_setup, "COMFY_MODELS", models), \
                    patch.object(model_setup, "ALL_MODEL_PATHS", tuple(x.relative_path for x in items)), \
                    patch.object(model_setup, "MODEL_FILES", items), \
                    patch.object(model_setup, "_latest_snapshot", return_value=snapshot), \
                    patch.object(model_setup, "_check_disk"), \
                    patch.object(model_setup, "download_model", side_effect=fetch) as download:
                model_setup._ensure_models_unlocked()
            download.assert_called_once()
            self.assertTrue(download.call_args.args[0].endswith("new.safetensors"))
            self.assertEqual((models / "existing.safetensors").resolve(), snapshot / "existing.safetensors")
            self.assertTrue((models / "new.safetensors").is_file())

    def test_bundle_extension_retains_existing_records_and_checks_parent_commit(self):
        data = tensor_bytes()
        digest = hashlib.sha256(data).hexdigest()
        old_file = types.SimpleNamespace(rfilename="old.safetensors",
                                         lfs=types.SimpleNamespace(sha256=digest, size=len(data)))
        api = Mock()
        api.whoami.return_value = {"auth": {"accessToken": {"role": "write"}}}
        api.repo_info.return_value = types.SimpleNamespace(private=True, siblings=[old_file], sha="parent-123")
        api.create_commit.return_value.oid = "updated-456"
        hub = types.SimpleNamespace(HfApi=Mock(return_value=api),
                                    CommitOperationAdd=lambda **kwargs: types.SimpleNamespace(**kwargs))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new.safetensors").write_bytes(data)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    patch.dict(os.environ, {"HF_TOKEN": "test-only-write-token"}, clear=True), \
                    patch.object(model_setup, "BOOTSTRAP_MODE", True), \
                    patch.object(model_setup, "COMFY_MODELS", root), \
                    patch.object(model_setup, "ALL_MODEL_PATHS", ("new.safetensors",)), \
                    patch.object(model_setup, "ensure_models"):
                result = bootstrap_hf_repo.bootstrap_bundle()
        kwargs = api.create_commit.call_args.kwargs
        self.assertEqual(kwargs["parent_commit"], "parent-123")
        manifest = next(x for x in kwargs["operations"] if x.path_in_repo == "bundle-manifest.json")
        files = json.loads(manifest.path_or_fileobj.getvalue())["files"]
        self.assertEqual(set(files), {"old.safetensors", "new.safetensors"})
        self.assertEqual(files["old.safetensors"]["sha256"], digest)
        self.assertTrue(result["cached_model"].endswith(":updated-456"))
        self.assertNotIn("old.safetensors", [x.path_in_repo for x in kwargs["operations"]])

    def test_read_token_fails_before_downloading_anything(self):
        api = Mock()
        api.whoami.return_value = {"auth": {"accessToken": {"role": "read"}}}
        hub = types.SimpleNamespace(HfApi=Mock(return_value=api), CommitOperationAdd=Mock())
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch.dict(os.environ, {"HF_TOKEN": "test-only-read-token"}, clear=True), \
                patch.object(model_setup, "BOOTSTRAP_MODE", True), \
                patch.object(model_setup, "ensure_models") as prepare:
            with self.assertRaisesRegex(model_setup.ModelSetupError, "read-only"):
                bootstrap_hf_repo.bootstrap_bundle()
        prepare.assert_not_called()
        api.create_commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()

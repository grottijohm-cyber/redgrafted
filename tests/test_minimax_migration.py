import base64
import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from huggingface_hub import CommitOperationDelete
import bootstrap_hf_repo
import model_setup
import worker
from test_model_profiles import minimax_profile, tensor_bytes


class MigrationTests(unittest.TestCase):
    def test_retirement_is_atomic_with_validated_upload_and_never_on_prepare_failure(self):
        retired = 'checkpoints/10Eros_v1.5_DMD_INT8_checkpoint.safetensors'
        old = types.SimpleNamespace(rfilename=retired,
                                    lfs=types.SimpleNamespace(sha256='a'*64, size=10))
        for failure in (False, True):
            with self.subTest(prepare_failure=failure), tempfile.TemporaryDirectory() as folder:
                api = Mock()
                api.whoami.return_value = {'auth': {'accessToken': {'role': 'write'}}}
                api.repo_info.return_value = types.SimpleNamespace(private=True, siblings=[old, types.SimpleNamespace(rfilename='diffusion_models/old-redgraft.safetensors', lfs=old.lfs)], sha='previous')
                api.create_commit.return_value.oid = 'next'
                root = Path(folder)
                (root/'new.safetensors').write_bytes(tensor_bytes())
                with patch('huggingface_hub.HfApi', return_value=api), \
                     patch.dict(os.environ, {'HF_WRITE_TOKEN': 'test-token'}), \
                     patch.object(model_setup, 'MODEL_PROFILE', 'minimax'), \
                     patch.object(model_setup, 'BOOTSTRAP_MODE', True), \
                     patch.object(model_setup, 'COMFY_MODELS', root), \
                     patch.object(model_setup, 'ALL_MODEL_PATHS', ('new.safetensors',)), \
                     patch.object(model_setup, 'ensure_models', side_effect=model_setup.ModelSetupError('incomplete') if failure else None):
                    if failure:
                        with self.assertRaises(model_setup.ModelSetupError):
                            bootstrap_hf_repo.bootstrap_bundle()
                        api.create_commit.assert_not_called()
                    else:
                        bootstrap_hf_repo.bootstrap_bundle()
                        ops = api.create_commit.call_args.kwargs['operations']
                        self.assertEqual([op.path_in_repo for op in ops if isinstance(op, CommitOperationDelete)], [retired, 'diffusion_models/old-redgraft.safetensors'])
                        manifest = next(op for op in ops if op.path_in_repo == 'bundle-manifest.json')
                        self.assertEqual(
                    set(json.loads(manifest.path_or_fileobj.getvalue())['files']),
                    {
                        'new.safetensors',
                        'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
                        'loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors',
                    },
                )
                        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'], 'previous')

    def test_custom_job_prompt_reaches_minimax_conditioner_unchanged(self):
        data = io.BytesIO()
        Image.new('RGB', (8, 8), 'blue').save(data, format='PNG')
        prompt = 'A person waves to the camera in a continuous shot.'
        with minimax_profile(), contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(worker, 'WORKFLOW_PATH', Path('api-workflow-minimax.json')))
            stack.enter_context(patch.object(worker, 'ensure_models'))
            stack.enter_context(patch.object(worker, 'wait_for_comfyui'))
            stack.enter_context(patch.object(worker, 'upload_input_image', return_value='job.png'))
            stack.enter_context(patch.object(worker, '_safe_input_path', return_value=None))
            queue = stack.enter_context(patch.object(worker, 'queue_workflow', return_value='prompt-id'))
            stack.enter_context(patch.object(worker, 'wait_for_history', return_value={}))
            stack.enter_context(patch.object(worker, 'get_output_descriptors', return_value=[{}]))
            stack.enter_context(patch.object(worker, '_safe_output_path', return_value=Path('/tmp/nonexistent-video.mp4')))
            stack.enter_context(patch.object(worker, 'publish_output', return_value={'mime_type': 'video/mp4', 'url': 'https://example.com/video.mp4'}))
            result = worker.handle_job({'input': {'image': base64.b64encode(data.getvalue()).decode(), 'prompt': prompt}})
            self.assertEqual(result.get('status'), 'success', result)
            self.assertEqual(queue.call_args.args[0]['364']['inputs']['prompt'], prompt)
            self.assertNotIn('text', queue.call_args.args[0]['364']['inputs'])

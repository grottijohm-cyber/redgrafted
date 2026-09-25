import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import app_worker
import permanent_storage
from botocore.exceptions import ClientError


class PermanentStorageTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "BUCKET_ENDPOINT_URL": "https://example.r2.cloudflarestorage.com",
                "BUCKET_ACCESS_KEY_ID": "access",
                "BUCKET_SECRET_ACCESS_KEY": "secret",
                "BUCKET_NAME": "renders",
                "BUCKET_REGION": "auto",
                "PERMANENT_ARCHIVE_PREFIX": "redgraft/renders",
            },
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_archive_generation_writes_metadata_and_returns_signed_video(self):
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://signed.example/video.mp4"
        with patch.object(permanent_storage, "_client", return_value=(client, "renders")):
            archived = permanent_storage.archive_generation_result(
                job_id="job-123",
                prompt="A slow camera move",
                preset_name="General NSFW",
                request_settings={"steps": 8, "enable_gimm": False},
                result={
                    "status": "success",
                    "seed": 42,
                    "prompt_id": "prompt-1",
                    "worker_version": "runpod-minimax-test",
                    "workflow": {"runtime_options": {"steps": 8}},
                    "videos": [
                        {
                            "filename": "render.mp4",
                            "type": "url",
                            "url": "https://temporary.example/render.mp4",
                            "mime_type": "video/mp4",
                        }
                    ],
                },
            )

        self.assertTrue(archived["permanent"])
        self.assertEqual(archived["videos"][0]["archive_key"], "job-123/render.mp4")
        self.assertEqual(archived["videos"][0]["url"], "https://signed.example/video.mp4")
        self.assertEqual(client.put_object.call_count, 1)
        args = client.put_object.call_args.kwargs
        self.assertEqual(args["Bucket"], "renders")
        self.assertTrue(args["Key"].startswith("redgraft/renders/"))
        self.assertTrue(args["Key"].endswith("/metadata.json"))
        metadata = json.loads(args["Body"].decode("utf-8"))
        self.assertEqual(metadata["prompt"], "A slow camera move")
        self.assertEqual(metadata["preset_name"], "General NSFW")
        self.assertEqual(metadata["request_settings"]["steps"], 8)
        self.assertEqual(metadata["videos"][0]["key"], "job-123/render.mp4")

    def test_list_renders_refreshes_private_signed_urls(self):
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "redgraft/renders/r1/metadata.json",
                    "LastModified": datetime.now(timezone.utc),
                }
            ],
            "IsTruncated": False,
        }
        metadata = {
            "render_id": "r1",
            "created_at": "2026-09-22T00:00:00Z",
            "prompt": "Prompt",
            "videos": [{"key": "job-1/render.mp4", "filename": "render.mp4"}],
        }
        body = MagicMock()
        body.read.return_value = json.dumps(metadata).encode("utf-8")
        client.get_object.return_value = {"Body": body}
        client.generate_presigned_url.return_value = "https://signed.example/fresh.mp4"

        with patch.object(permanent_storage, "_client", return_value=(client, "renders")):
            result = permanent_storage.list_renders()

        self.assertTrue(result["configured"])
        self.assertEqual(len(result["renders"]), 1)
        self.assertEqual(result["renders"][0]["videos"][0]["url"], "https://signed.example/fresh.mp4")
        self.assertTrue(result["renders"][0]["videos"][0]["permanent"])

    def test_legacy_listing_recovers_from_provider_no_such_key(self):
        client = MagicMock()
        client.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "key missing"}}, "ListObjectsV2")
        client.list_objects.return_value = {"Contents": [], "IsTruncated": False}
        with patch.object(permanent_storage, "_client", return_value=(client, "renders")):
            result = permanent_storage.list_renders()
        self.assertEqual(result["renders"], [])
        self.assertIsNone(result["cursor"])
        client.list_objects.assert_called_once_with(Bucket="renders",
            Prefix="redgraft/renders/", MaxKeys=300)


class AppWorkerTests(unittest.TestCase):
    def test_library_action_returns_archive_page(self):
        page = {"configured": True, "renders": [{"render_id": "r1"}], "cursor": None}
        with patch.object(app_worker, "list_renders", return_value=page):
            result = app_worker.handle_job({"input": {"action": "library", "max_keys": 300}})
        self.assertEqual(result["status"], "library")
        self.assertEqual(result["archive"], page)

    def test_successful_generation_archives_prompt_preset_and_runtime_settings(self):
        base_result = {
            "status": "success",
            "worker_version": "runpod-minimax-test",
            "videos": [{"filename": "render.mp4", "type": "url", "url": "https://temporary"}],
        }
        archived = {
            "permanent": True,
            "render_id": "r1",
            "videos": [{"filename": "render.mp4", "type": "url", "url": "https://signed"}],
        }
        job = {
            "id": "job-1",
            "input": {
                "image": "data:image/png;base64,AA==",
                "prompt": "Prompt text",
                "preset_name": "Anatomy Lock",
                "steps": 8,
                "enable_audio": False,
                "enable_gimm": True,
            },
        }
        with patch.object(app_worker, "base_handle_job", return_value=base_result) as base, patch.object(
            app_worker, "archive_generation_result", return_value=archived
        ) as archive:
            result = app_worker.handle_job(job)

        forwarded = base.call_args.args[0]
        self.assertNotIn("preset_name", forwarded["input"])
        archive_kwargs = archive.call_args.kwargs
        self.assertEqual(archive_kwargs["prompt"], "Prompt text")
        self.assertEqual(archive_kwargs["preset_name"], "Anatomy Lock")
        self.assertEqual(archive_kwargs["request_settings"]["steps"], 8)
        self.assertNotIn("image", archive_kwargs["request_settings"])
        self.assertNotIn("prompt", archive_kwargs["request_settings"])
        self.assertEqual(result["videos"][0]["url"], "https://signed")
        self.assertTrue(result["archive"]["permanent"])


if __name__ == "__main__":
    unittest.main()

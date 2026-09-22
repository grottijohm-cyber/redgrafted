# Permanent render storage (Cloudflare R2 / S3-compatible)

The phone client now has a permanent **Videos** library. Finished renders are stored in private object storage and the browser receives only short-lived signed playback/download URLs. The objects themselves do not expire unless you add a bucket lifecycle rule or delete them manually.

## RunPod endpoint environment variables

Set these on the deployed RunPod Serverless endpoint:

```text
BUCKET_ENDPOINT_URL=https://<cloudflare-account-id>.r2.cloudflarestorage.com
BUCKET_ACCESS_KEY_ID=<R2 access key id>
BUCKET_SECRET_ACCESS_KEY=<R2 secret access key>
BUCKET_NAME=<R2 bucket name>
BUCKET_REGION=auto
PERMANENT_ARCHIVE_PREFIX=redgraft/renders
PERMANENT_URL_SECONDS=604800
```

The first four variables are required. `BUCKET_REGION`, `PERMANENT_ARCHIVE_PREFIX`, and `PERMANENT_URL_SECONDS` are optional.

## R2 bucket settings

- Keep the bucket private. Public access is not required.
- Do **not** add an expiration/lifecycle rule if you want renders retained indefinitely.
- Create an R2 API token that can read/write objects in this bucket.
- The phone client never receives the R2 secret keys.

## How it works

1. The normal RunPod worker uploads each completed MP4 to the configured bucket.
2. `app_worker.py` writes a permanent metadata JSON object containing the prompt, preset, runtime settings, seed, job ID, workflow summary, and video object keys.
3. The hamburger menu's **Videos** tab requests `input.action=library` from the RunPod endpoint.
4. The worker lists metadata objects and creates fresh private signed URLs for playback/download.
5. Clearing Safari/local browser data does not delete the R2 videos. Re-entering the RunPod endpoint ID/API key and refreshing **Videos** can load the archive again.

The app intentionally has no permanent-video delete button. Queue × buttons only cancel/remove RunPod queue items; they do not delete completed archived renders.

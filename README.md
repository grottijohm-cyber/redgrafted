# LTX 2.5 adult image-to-video — RunPod Serverless

This is a fixed, queue-based RunPod worker. Each job accepts exactly two inputs:

- `image`: an HTTP(S) URL, raw base64, or a base64 data URI
- `prompt`: the motion/camera/audio instruction

The worker automatically expands the prompt with an adult-capable local Gemma
model, animates the supplied image with LTX 2.5, generates synchronized audio,
and returns an MP4.

The approximately 51 GB model set is downloaded once to the attached RunPod
network volume and reused by later workers after the endpoint scales to zero.

## What is included

- RunPod queue entrypoint detectable as `runpod.serverless.start(...)`
- ComfyUI 0.34 via `runpod/worker-comfyui:5.10.0-base`
- Native ComfyUI LTX 2.5 two-pass I2V graph; no custom-node packs
- REDGraft LTX 2.5 adult-tuned INT8 transformer from Civitai/Civit.red
- Uncensored Gemma 3 prompt enhancer with LTX projection
- Official LTX 2.5 main text encoder, VAEs, and latent upscaler
- Resumable, parallel downloads into `/runpod-volume/ltx25-models`
- URL/base64 input handling and optional S3-compatible output upload

The generation is fixed at 1152×640, 121 frames, 24 FPS (about 5 seconds), with
the official two-pass latent-upscale schedule. The main seed is randomized for
each job. Clients do not submit a workflow or node IDs.

## One-time prerequisites

1. Create a Hugging Face account.
2. Open [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), accept
   its access terms, and create a **read** token.
3. Keep your Civitai/Civit.red token available. The selected REDGraft file is
   currently public, so this token is optional unless the download endpoint
   later requires authentication.
4. Have a GitHub repository, a RunPod account, and the 70 GB network volume.

Do not commit either token to GitHub. They belong in RunPod endpoint environment
variables.

## Deploy from GitHub

### 1. Put this folder in GitHub

Create an empty repository, upload the contents of this folder to its root, and
commit them to `main`. `Dockerfile`, `handler.py`, and `api-workflow.json` must
be in the repository root.

### 2. Create the RunPod endpoint

In RunPod:

1. Open **Serverless** and create a new endpoint.
2. Select **Queue**.
3. Choose **Deploy from GitHub**, select the repository and branch `main`.
4. Set the Dockerfile path to `/Dockerfile`.
5. Under the endpoint's advanced/storage settings, attach the **70 GB network
   volume you created**. RunPod must mount it at `/runpod-volume`.
6. Continue even if GitHub indexing briefly shows an old scanner warning. This
   repository contains the required module-level call in `handler.py`, and the
   Dockerfile copies that file to `/handler.py`, which the inherited startup
   script executes.

This Docker build does not download any model weights, so it avoids the earlier
30-minute build timeout.

### 3. Worker settings

Use these as a starting point:

| Setting | Value |
| --- | --- |
| GPU | 48 GB VRAM: L40S, A40, or RTX A6000 |
| Network volume | Your 70 GB volume, attached |
| Container disk | 30 GB |
| Active workers | `0` |
| Max workers | `1` |
| Idle timeout | `5` seconds |
| Execution timeout | `7200` seconds |

A 24 GB GPU is not recommended for this exact graph because it uses a 22B video
model plus separate 12B main and enhancer text encoders. Active workers can stay
at `0`: scaling down stops GPU billing while the downloaded weights remain on
the network volume.

### 4. Environment variables

Add this required secret to the endpoint:

```text
HF_TOKEN=hf_your_read_token
```

Optional Civitai authentication:

```text
CIVITAI_TOKEN=your_civitai_token
```

For reliable video delivery, configure an S3-compatible bucket:

```text
BUCKET_ENDPOINT_URL=https://s3.YOUR_REGION.amazonaws.com
BUCKET_ACCESS_KEY_ID=...
BUCKET_SECRET_ACCESS_KEY=...
BUCKET_NAME=your-bucket
```

Without object storage, only MP4 files up to 8 MiB are returned inline as
base64. Larger files produce a clear configuration error rather than overflowing
the RunPod job response.

## First request

Use RunPod's asynchronous `/run` route because model preparation and video
generation are long-running:

```bash
curl --request POST \
  "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  --header "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "input": {
      "image": "https://example.com/your-first-frame.png",
      "prompt": "The adult subject turns slowly toward the camera while the camera makes a gentle push-in; quiet room ambience."
    }
  }'
```

For private images, send a base64 data URI instead of hosting the file:

```json
{
  "input": {
    "image": "data:image/png;base64,iVBORw0KGgo...",
    "prompt": "Your motion, camera, and sound instruction"
  }
}
```

Both fields are required, and extra job fields are rejected. The first job
downloads about 51 GB to `/runpod-volume/ltx25-models` before generation. Later
cold starts and jobs reuse those files without downloading them again.

Poll the job with:

```bash
curl \
  "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  --header "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

Successful output resembles:

```json
{
  "status": "success",
  "prompt_id": "...",
  "prompt_enhanced": true,
  "videos": [
    {
      "filename": "LTX25_i2v_00001_.mp4",
      "type": "url",
      "url": "https://your-bucket.example/...",
      "mime_type": "video/mp4"
    }
  ]
}
```

## How to write the prompt

The image already defines appearance and composition. Describe what happens
next:

- subject motion and timing
- camera motion or explicitly static camera
- environmental movement
- sound effects, ambience, music, or exact dialogue if wanted
- continuity constraints such as one continuous shot or no cuts

Short prompts are expanded automatically. Avoid re-describing the whole image;
conflicting appearance details can cause a visual jump.

## Downloaded model set

| Component | Approximate size | Source |
| --- | ---: | --- |
| REDGraft LTX 2.5 adult transformer | 17.0 GB | [Civitai version 3250230](https://civitai.com/models/1295569?modelVersionId=3250230) / [Civit.red mirror](https://civit.red/models/1295569?modelVersionId=3250230) |
| Official LTX 2.5 main text encoder | 15.4 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |
| Gemma 3 Heretic INT8 enhancer | 13.2 GB | [DreamFast/gemma-3-12b-it-heretic-v2](https://huggingface.co/DreamFast/gemma-3-12b-it-heretic-v2) |
| LTX 2.3 text projection | 2.3 GB | [ReubenF10/ComfyUI-Models](https://huggingface.co/ReubenF10/ComfyUI-Models) |
| LTX 2.5 video VAE | 1.45 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |
| LTX 2.5 audio VAE | 0.36 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |
| LTX 2.5 latent upscaler | 1.0 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |

REDGraft and the Heretic enhancer are community models, not official Lightricks
releases. Review each model page and license before commercial use. LTX 2.5 has
its own community license and gated access terms.

## Troubleshooting

### Build still exceeds 30 minutes

Confirm the GitHub repository contains this Dockerfile. It has no `wget`,
`curl`, Hugging Face CLI, model URL, `HF_TOKEN` build argument, or model `RUN`
step. If RunPod is building an older commit, select `main` again or wait for
GitHub indexing and redeploy.

### `HF_TOKEN is required`

Add `HF_TOKEN` to the endpoint environment, not the Dockerfile. Make sure the
same Hugging Face account accepted the LTX 2.5 access terms.

### Insufficient model-storage space

Verify that the 70 GB network volume is attached and mounted at
`/runpod-volume`. The worker calculates the remaining download size and requires
an additional 8 GiB safety margin. Seventy GB is enough for this fixed model set,
but it leaves limited room for additional checkpoints or large LoRAs.

### REDGraft download returns 401/403

Add `CIVITAI_TOKEN` to the endpoint environment. The code sends it as a bearer
token to the Civitai download API used by Civit.red.

### Every job downloads the models again

The network volume is not attached to this endpoint, or RunPod did not mount it
at `/runpod-volume`. Check the endpoint's storage/network-volume selection and
redeploy it.

### Output is too large to return inline

Add the four `BUCKET_*` environment variables shown above.

## Safety and rights

Use only clearly adult, consensual material. Do not use this worker for minors,
non-consensual sexual content, or sexual deepfakes of real people. Make sure you
have the rights and consent required for every input image.

## Local checks

The included checks validate the handler, workflow references, model mapping,
and request patching without downloading models or requiring a GPU:

```bash
python -m unittest discover -s tests -v
```

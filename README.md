# RunPod image-to-video worker

Use your existing `grottijohm-cyber/redgrafted` repository and Queue Serverless endpoint. Normal generation accepts **one first-frame image and one prepared prompt**, producing approximately **10 seconds** of video. The worker sends your prompt directly to the video text encoder.

Version 4 adds a selectable **10Eros 1.5** profile and an incremental upload to the existing private Hugging Face bundle. Set `MODEL_PROFILE=10eros` to use it; the default `redgraft` profile retains LTX 2.5. The new profile uses a complete LTX 2.3 checkpoint with its matching conditioning and VAEs. It is not a LoRA applied to the LTX 2.5 transformer.

## Add and activate 10Eros

Use the existing `cached-model-fix` branch, endpoint, and `grottijohm/redgraft-ltx25-runpod` Hugging Face repository.

1. Build/release the updated branch on the existing endpoint. Keep its current cached model selected so setup can reuse the Gemma 3 file from the older seven-file bundle.
2. Set `MODEL_PROFILE=10eros` and temporarily set `BOOTSTRAP_HF_REPO=1`. Keep the 100 GB container disk. Remove any explicit `WORKFLOW_PATH` override so the worker can select the matching workflow.
3. Set `HF_WRITE_TOKEN` to an existing token with write access to your private bundle. If the endpoint's existing `HF_TOKEN` already has write access, it can be reused without adding another token. A read-only token is rejected before model downloads.
4. Submit `{"input":{"action":"setup"}}` through Requests. This is a billable setup job; its downloads, integrity checks, and upload stay inside the tracked job.
5. After `setup_complete`, select **the new revision returned in `cached_model`** in RunPod's Model setting. The old selection is pinned to `d34af963480f3bc9fa9254d83ff10f3da0dca993` and will not gain the newly uploaded files automatically.
6. Remove `BOOTSTRAP_HF_REPO` and the temporary `HF_WRITE_TOKEN`, keep `MODEL_PROFILE=10eros`, and release the endpoint configuration. Status should report `runpod-model-profiles-4`, profile `10eros`, and `files_ready: true`.

The normal image-and-prompt request stays unchanged. Both sampling passes use `10Eros_v1.5_DMD_INT8_checkpoint.safetensors`; video and audio remain 241 frames at 24 fps, targeting 1152×640 after 2x upscaling. Gemma 3 is used only as the LTX 2.3 text encoder. Automatic prompt enhancement remains disabled.

| 10Eros component | Size | Source |
| --- | ---: | --- |
| Full INT8 checkpoint with DMD hybrid v2, projection, video/audio VAEs | 29.16 GB | [CornLogic/10EROS-INT8](https://huggingface.co/CornLogic/10EROS-INT8) |
| Gemma 3 12B INT8 text encoder | 13.22 GB | [DreamFast/gemma-3-12b-it-heretic-v2](https://huggingface.co/DreamFast/gemma-3-12b-it-heretic-v2) |
| LTX 2.3 spatial upscaler x2 1.1 | 1.00 GB | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |

The profile uses approximately **43.38 GB** of weights. When the existing Gemma 3 file is available, setup needs approximately **30.16 GB of new downloads**. Original bundle files remain present, so the combined repository grows by about 30.16 GB. RunPod can cache that entire repository even though a worker loads only one profile.

The [10Eros publisher](https://huggingface.co/TenStrip/LTX2.3-10Eros) links the selected INT8 release. Its [quantization notes](https://huggingface.co/CornLogic/10EROS-INT8/blob/main/README.md) specify that DMD hybrid v2 is merged at strength 1.0. Do not add that distillation LoRA again. The new files are pinned to upstream commit revisions and checked against upstream SHA-256 metadata. Sampling uses the [DMD publisher's first-pass and upscale schedule](https://huggingface.co/TenStrip/LTX2.3_DMD_Lora).

Upload preserves the digest records for all existing model files, adds the selected profile in one commit, and checks the previous repository revision to avoid overwriting a simultaneous update. It does not delete older weights. To switch back later, use `MODEL_PROFILE=redgraft` with a bundle revision containing the original files.

CPU tests validate the profile wiring, cross-version rejection, upstream hashes, reuse of existing files, and preservation of the prior bundle. Actual GPU loading, memory fit, and output quality require a live generation test.

## Existing REDGraft configuration

Version 3 removes the automatic prompt-enhancement nodes and changes video/audio length from 121 to 241 frames at 24 fps (approximately 10.04 seconds). It retains the existing REDGraft checkpoint, Gemma 4 video text encoder, VAEs, two sampling passes, and 2x latent upscaler. The five active model source URLs remain unchanged. The longer workflow still requires a real GPU test for speed, memory fit, and output quality.

## 1. Update the existing endpoint

Merge the repair into `cached-model-fix`, then build/release that branch on your existing RunPod endpoint using `/Dockerfile`. Confirm the new build is **Completed and active**. Endpoint “Ready” alone can refer to the previous release.

The image contains code and dependencies. It does not download model weights during Docker build. RunPod still enforces a 30-minute Docker build limit; idle timeout is a separate runtime setting.

To identify the worker that actually answers requests, send this in the endpoint's **Requests** tab:

```json
{"input":{"action":"status"}}
```

The output should show `worker_version: runpod-model-profiles-4`. `files_ready` checks available model containers; it is not a successful video-generation test. The `runtime` object also reports ComfyUI's process state and the container's RAM allowance. This status request starts a worker if one is needed and uses normal RunPod billing.


## Prepared prompts, duration, and checkpoint verification

Rebuild and release the current `cached-model-fix` branch. Keep your existing Cached Model selection; the completed seven-file bundle remains compatible. No new model download setup or Hugging Face upload is required when retaining the redgraft profile. Adding 10Eros requires the setup above.

The image input stays the same. Put your finished prompt in `input.prompt`; perform any enhancement in your own application before submitting the request. The worker only trims surrounding whitespace and encodes that text through node `364`. The Gemma 3 enhancer and its separate projection loader (nodes `380` and `393`) are removed. Gemma 4 remains because it supplies the conditioning representation required by LTX 2.5.

The video latent and audio latent both request **241 frames at 24 fps**, preserving the required `8*n+1` frame count. This increases clip length while keeping playback at normal speed. Longer clips require more generation work and can use more memory; the existing OOM diagnostics remain active.

Both sampler paths use `redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors` from loader `384`: first-pass guider `388` feeds sampler `344`, and second-pass guider `391` feeds sampler `368`. Status and successful job results report this requested configuration in `workflow`:

```json
{
  "checkpoint": "redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
  "frames": 241,
  "fps": 24.0,
  "duration_seconds": 10.042,
  "prompt_enhanced": false
}
```

These fields describe the workflow submitted by the answering worker. They are not an independent identification of the downloaded weights or a measurement of the final MP4. Model-container and manifest validation still apply.

The existing bundle may still contain the two retired enhancer files. Version 3 does not validate, link, or load those files. A new setup requires only five files, but this update does not delete anything from your existing Hugging Face repository or network volume. RunPod can still cache the full older repository, so reduced application model requirements do not by themselves shrink its host-side cache download.

The checkpoint's filename does not set output resolution or guarantee prompt adherence. The current workflow starts at 576x320 and applies 2x latent upscaling, targeting 1152x640. When the result exceeds the inline delivery allowance, it is re-encoded to fit and reports `compressed_for_delivery: true`. Ten-second clips share the same response-size limit, so delivery compression can affect fine detail. Existing object-storage delivery preserves the original encoded file when configured.

## 2. Prepare the cached bundle once, if needed

**Skip this section if your complete private bundle already exists and is selected as the endpoint's Cached Model.** An existing valid bundle remains supported even if it predates the new manifest.

If preparation has not finished, use these temporary settings on the same endpoint:

| Setting | Value |
| --- | --- |
| Cached Model | Leave empty for direct preparation of only the required files |
| Container disk | 100 GB |
| GPU | Your selected 48 GB class; GPU compatibility still needs a generation test |
| Active / maximum workers | 0 / 1 |
| Execution timeout | 7200 seconds |
| `BOOTSTRAP_HF_REPO` | `1` |
| `HF_BUNDLE_REPO` | `grottijohm/redgraft-ltx25-runpod` |
| `HF_TOKEN` | Existing Hugging Face read token with access to the required models |
| `HF_WRITE_TOKEN` | Token permitted to create/write your private bundle repository |
| `CIVITAI_TOKEN` | Existing token with access to the configured model download |

Set credentials in RunPod's environment/secret settings. They do not belong in GitHub files or Docker build arguments.

Submit this one job through **Requests**:

```json
{"input":{"action":"setup"}}
```

Wait for the RunPod job to complete with `output.status: setup_complete`. Worker logs show download and hash progress; upload progress also appears in the worker logs when provided by the Hub client. Preparation is now tracked as an active job, so its upload is not an untracked background process subject to idle shutdown. The platform execution timeout still applies.

The helper checks the existing model containers, records SHA-256 hashes, and publishes all files plus `bundle-manifest.json` in one Hugging Face commit. A hash manifest records the files obtained; it does not authenticate an upstream publisher or retroactively pin mutable upstream URLs. Original model access conditions and licenses continue to apply.

After successful setup, apply the normal settings below. Generation requests deliberately fail while `BOOTSTRAP_HF_REPO=1` to prevent accidentally downloading models on fresh generation workers.

## 3. Normal RunPod settings

| Setting | Value |
| --- | --- |
| Endpoint | Existing Queue Serverless endpoint |
| Cached Model | `grottijohm/redgraft-ltx25-runpod` |
| Model access token | Your Hugging Face read token, entered in RunPod's cached-model access configuration |
| Active / maximum workers | 0 / 1 |
| Idle timeout | 120 seconds; idle time is billable |
| Execution timeout | 7200 seconds |
| `JOB_TIMEOUT_SECONDS` | 6600, leaving time before the platform cutoff |
| `BOOTSTRAP_HF_REPO` | Remove, or set to `0` |
| `COMFY_MEMORY_PROFILE` | `conservative` (image default) |
| `HF_WRITE_TOKEN` | Remove after setup |
| `CIVITAI_TOKEN` | Not required when all files come from the complete cache |

An `HF_TOKEN` container environment variable alone does not configure RunPod's host-side access to a private cached model. Use the Model access-token setting too. Runtime read credentials are needed only during direct downloads.

This branch uses **RunPod Cached Models**. Your separate 70 GB network volume is a different storage resource; attaching it does not populate the model cache. A network volume is not required for this cached-bundle design. Do not delete any existing storage until you have checked whether it contains files you need.

Deploy these settings, then run the status request again. Normal readiness should report `cached_bundle_mounted`, `files_ready`, and `generation_configured` as true. On the first use of a bundle with a manifest, the worker verifies its recorded hashes; later jobs on the same worker reuse those validation results while files remain unchanged.

The image defaults to `COMFY_LOG_LEVEL=INFO` so routine backend dispatch messages do not bury progress and memory diagnostics. An existing endpoint environment override still takes precedence.

## If initialization or generation stalls

The September 13 logs establish two different stages:

| Last log line / symptom | Meaning and action |
| --- | --- |
| `image ready, initializing model files` | The Docker image has loaded. RunPod is preparing the selected cached model before starting application code. Changing the handler or idle timeout cannot unstick this host-side stage. |
| `BOOTSTRAP_COMPLETE` | The private model bundle was published successfully. Do not repeat setup merely because another worker is cold-starting. |
| `Using complete cached bundle ... no large runtime downloads needed` | The older worker found its seven required files; version 3 requires five. The supplied log reached this point successfully. |
| `triggered memory limits (OOM)`, followed by `Connection refused` | The earlier generation hit a container memory limit and ComfyUI became unavailable while loading the video model. This is not a missing `runpod.serverless.start` or a registry authentication error. |

The existing seven-file bundle is approximately **50.7 GB on disk**. A fresh host can need to fetch it before the worker starts. A previous host's cached files do not guarantee every subsequent host already has them. See [RunPod's cached-model lifecycle](https://docs.runpod.io/serverless/endpoints/model-caching).

For a worker with no model-initialization progress for an hour, first check its **System** logs for an explicit download/access error. Confirm the existing completed bundle is selected under **Manage → Edit endpoint → Model**, including host-side access to that private repository. If initialization remains stalled, cancel the pending test and replace the stuck worker once using RunPod's worker controls. If a fresh worker also stalls at the same host-side line, RunPod support needs the endpoint ID, worker ID, image tag, model repository, timestamps, and System logs. Rebuilding identical code or publishing the bundle again will not repair a host download failure. Do not put access tokens in shared logs.

Versions 2 and 3 read the ComfyUI PID recorded by the base image. A dead or restarted process now fails the job promptly and requests worker replacement. If the PID cannot be inspected, repeated connection failures after submission stop after 60 seconds (`COMFY_UNREACHABLE_TIMEOUT_SECONDS`). Slow responses alone do not trigger that connection-failure timer. The handler does not resubmit a failed generation.

The default memory profile disables pinned-memory caching, asynchronous offload, and node-result caching, and prefers disk-backed dynamic loading (`--disable-pinned-memory --disable-async-offload --cache-none --fast-disk`). Dynamic VRAM stays enabled. These flags reduce avoidable host-memory use; they can make generation slower and do not prove the full workflow fits a particular worker. Set `COMFY_MEMORY_PROFILE=default` only for an intentional comparison with upstream defaults. The launcher preserves the base image's GPU checks, PID tracking, and normal/locally served API modes.

Status results and generation logs include `runtime.memory.container_limit_bytes`, current/peak usage when available, and Linux OOM-kill counters. The GPU's advertised 48 GB is **VRAM**; it does not establish the container's available **system RAM**. A 70 GB network volume or larger container disk does not increase either kind of memory. If another OOM occurs, use the recorded container RAM limit and GPU memory error details to choose a worker with sufficient resources; increasing only disk size will not help. The supplied A40 run failed, so 48 GB GPU compatibility remains unverified even with the memory profile.

## 4. Use image + prompt

Download/extract the repository ZIP and double-click **`client/client.html`**. No Docker Desktop or local GPU setup is needed for this page.

1. Enter your existing RunPod endpoint ID and API key.
2. Choose an image, or paste a direct HTTPS image link.
3. Paste your prepared prompt and click **Generate video**.
4. Download the MP4 when the job finishes.

The key remains in page memory and is sent only to RunPod. The page saves only the endpoint/job ID so you can reconnect after closing it. Browser network/CORS compatibility and live endpoint access must be checked with your account; if the browser blocks the connection, the RunPod Requests tab below uses the same worker directly.

The page uses asynchronous `/run` requests, displays worker stages, supports cancellation, and resumes status checks without automatically resubmitting a generation. If submission times out before an ID arrives, check RunPod's Requests tab before trying again to avoid duplicate paid jobs.

For the RunPod Requests tab or an API client:

```json
{
  "input": {
    "image": "https://example.com/your-first-frame.png",
    "prompt": "Clouds drift slowly over the mountain as the camera gently moves forward."
  }
}
```

`image` also accepts a base64 data URI. Direct file uploads from the page are limited to 6 MB to leave room for base64 within RunPod's request limits. A URL must resolve to the image itself, not an HTML sharing page.

## Video delivery

By default, the worker returns MP4 bytes as base64, and the page makes them downloadable. The combined binary allowance is at most 6,000,000 bytes, with a final serialized-response check. Oversized MP4s are re-encoded with audio to fit; the response reports `compressed_for_delivery: true`. This changes encoding quality. If compression cannot fit the budget, the job returns a delivery error rather than a truncated video.

For original video files or longer videos, optionally configure all four object-storage variables:

- `BUCKET_ENDPOINT_URL`
- `BUCKET_ACCESS_KEY_ID`
- `BUCKET_SECRET_ACCESS_KEY`
- `BUCKET_NAME`

Complete storage configuration returns download URLs without re-encoding. Incomplete configuration fails before generation. A delivery failure preserves the original file on the current worker's temporary disk for diagnosis; it is **not durable storage**, and the file is lost when that worker is removed.

Retrieve `/run` results within 30 minutes of completion. Save downloaded videos locally if you want to keep them.

## Verification

Run the CPU checks with Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
bash -n start-worker.sh
node --test tests/test_client.cjs
```

The Python tests cover prepared-prompt preservation, synchronized ten-second video/audio length, the checkpoint connection to both samplers, compatibility with the older bundle manifest, input validation, the workflow mapping, complete/incomplete downloads, HTTP range recovery, cached-bundle hash mismatches, tracked setup, bundle publication, shared job deadlines, cancellation, and cleanup after delivery failure. Runtime checks cover cgroup v1/v2 limits, new versus historical OOM events, dead/restarted processes, bounded connection retries, recovery after temporary failures, and slow responses. A real child-process exit verifies fail-fast detection; a small stand-in application verifies that the ComfyUI launcher preserves its PID and arguments. Provider/ComfyUI calls are mocked. With FFmpeg installed, a real synthetic video is compressed and checked for size, duration, video, and audio. That media test is skipped if FFmpeg is absent; GitHub checks install it.

The client tests require Node.js 22 and exercise the actual page script with simulated page elements and RunPod responses. They check downloads, duplicate submission prevention, reconnection, and cancellation races. They are not a real browser/CORS test. Python, Node.js, and FFmpeg are needed only to run development checks, not to use the standalone page. GPU inference and external model access are not tested by this suite.

Before calling the deployment complete, run a neutral image/prompt through the real endpoint and confirm: model loading, prepared-prompt handling, approximately ten-second MP4 playback with expected audio, and download all succeed. Check logs for GPU memory errors. Build duration, cold-start time, generation speed, and 48 GB GPU fit have not been benchmarked here.

## Reference

- [RunPod GitHub builds](https://docs.runpod.io/serverless/workers/github-integration)
- [Cached models and their limitations](https://docs.runpod.io/serverless/endpoints/model-caching)
- [Endpoint configuration and billing-related timeouts](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
- [Asynchronous requests and result retention](https://docs.runpod.io/serverless/endpoints/send-requests)

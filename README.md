# REDGraft LTX 2.5 image-to-video — RunPod Serverless

This worker accepts exactly two job inputs: `image` and `prompt`. It automatically expands the prompt, runs REDGraft LTX 2.5 image-to-video, generates synchronized audio, and returns an MP4.

## Final deployment target

Use repository `grottijohm-cyber/redgrafted`, branch `cached-model-fix`, Dockerfile `/Dockerfile`.

The preferred final RunPod Cached Model is:

`grottijohm/redgraft-ltx25-runpod`

That private Hugging Face repo contains the exact REDGraft transformer, prompt enhancer, LTX text encoder, VAEs, projection, and latent upscaler needed by this worker. In final mode the worker only links files from RunPod Cached Models and performs no large model downloads at startup.

## One-time bootstrap

Because the complete bundle does not exist initially, do one bootstrap deployment first:

1. Set RunPod Cached Model to `Lightricks/LTX-2.5`.
2. Use one 48 GB GPU and at least 50 GB container disk.
3. Add `HF_TOKEN` (read token), `CIVITAI_TOKEN`, `HF_WRITE_TOKEN` (one-time Hugging Face write token), and `BOOTSTRAP_HF_REPO=1`.
4. Deploy and let the first worker start. It prepares the exact model set and uploads it to the private repo `grottijohm/redgraft-ltx25-runpod`.
5. Wait until the worker log contains `BOOTSTRAP_COMPLETE:grottijohm/redgraft-ltx25-runpod`.
6. Redeploy with Cached Model changed to `grottijohm/redgraft-ltx25-runpod`.
7. Remove `BOOTSTRAP_HF_REPO` and `HF_WRITE_TOKEN`. `CIVITAI_TOKEN` is no longer required for normal use.

The bootstrap is a one-time migration. Do not make the bundle public unless the upstream model licenses/terms permit redistribution.

## Normal RunPod settings

- Cached Model: `grottijohm/redgraft-ltx25-runpod`
- GPU: one 48 GB GPU
- Active workers: `0`
- Max workers: `1`
- Idle timeout: `120` seconds
- Execution timeout: `7200` seconds
- Network volume: none
- `HF_TOKEN`: keep your read token available so RunPod can access the private cached model

## Request format

```json
{
  "input": {
    "image": "https://example.com/image.jpg",
    "prompt": "The subject turns toward the camera while the camera slowly pushes in."
  }
}
```

The prompt is automatically passed through the bundled prompt enhancer before LTX conditioning. Clients do not submit ComfyUI workflows or node IDs.

## Model bundle

The bundle contains:

- REDGraft LTX 2.5 transformer
- Gemma Heretic prompt enhancer
- LTX text projection
- official LTX 2.5 text encoder
- LTX 2.5 video VAE
- LTX 2.5 audio VAE
- LTX 2.5 latent spatial upscaler

The original `main` branch remains untouched while this deployment path is being verified.

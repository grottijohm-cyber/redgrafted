# REDGraft LTX 2.5 image-to-video — RunPod Serverless

This is the REDGraft-based RunPod worker. Each job accepts exactly two inputs:

- `image`: an HTTP(S) URL, raw base64, or a base64 data URI
- `prompt`: the motion/camera/audio instruction

The worker automatically expands the prompt, animates the supplied image with
REDGraft LTX 2.5, generates synchronized audio, and returns an MP4.

## Deployment target

Use this repository and the `cached-model-fix` branch when deploying from GitHub.
The stock `Lightricks/LTX-2.5` model is only used as the source for the official
LTX 2.5 support files; the main diffusion transformer remains REDGraft.

## RunPod setup

1. RunPod Serverless -> Deploy from GitHub.
2. Repository: `grottijohm-cyber/redgrafted`.
3. Branch: `cached-model-fix`.
4. Dockerfile: `/Dockerfile`.
5. Cached Model: `Lightricks/LTX-2.5`.
6. GPU: one 48 GB GPU.
7. Active workers: `0`.
8. Max workers: `1`.
9. Idle timeout: `120` seconds.
10. Execution timeout: `7200` seconds.
11. Add `HF_TOKEN` after accepting the LTX 2.5 Hugging Face terms.
12. Add `CIVITAI_TOKEN` if the REDGraft download endpoint requires it.

No Docker Desktop is needed on your PC. No dedicated RunPod network volume is
required for the official LTX 2.5 cached model files.

## Request format

```json
{
  "input": {
    "image": "https://example.com/image.jpg",
    "prompt": "The subject turns toward the camera while the camera slowly pushes in."
  }
}
```

Clients only submit `image` and `prompt`.

## What is included

- RunPod Queue entrypoint in `handler.py`
- REDGraft LTX 2.5 adult-tuned transformer
- Official LTX 2.5 support files
- Automatic prompt expansion
- Fixed ComfyUI API workflow
- URL/base64 image input
- MP4 output

The original `main` branch has been left untouched while this deployment path is
being corrected and verified.

# RebelUI

A direct-to-library runner for ComfyUI's core. RebelUI skips the graph engine and calls ComfyUI's model loading, conditioning, sampling, memory management, and VAE code directly.

## What changed in this build

This revision supports **both GGUF and full-weight models/encoders** in the same UI.

- Diffusion-model filter: **All / Full / GGUF**
- Text-encoder filter: **All / Full / GGUF**
- **Full includes safetensors, FP16, FP8 and INT8** weights. INT8 safetensors are not treated as GGUF.
- Direct folder scanning makes `.gguf` files visible even when they are stored in `models/diffusion_models` or `models/text_encoders` and ComfyUI's normal extension filter hides them.
- GGUF text encoders can load even when the selected diffusion model is a normal safetensors model.
- Encoder types are read from the installed ComfyUI `CLIPType` enum instead of being hard-coded.
- Architecture browser is populated from the installed ComfyUI supported-model classes and includes a **Krea 2** preset.
- Sampler and scheduler lists are read from the installed ComfyUI build.
- Refreshed raised-button visual style, sharper typography, alignment, and RebelUI header logo.

## Krea 2 Turbo preset

For the current official Krea 2 Turbo local workflow, use:

- Architecture: `krea2`
- Diffusion model: `krea2_turbo_int8_convrot.safetensors` or a supported FP8 build
- Text encoder: `qwen3vl_4b_fp8_scaled.safetensors`
- CLIP type: `krea2`
- VAE: `qwen_image_vae.safetensors`
- Frames: `1`
- Typical Turbo start: `8` steps, CFG `1`, Euler + Simple

Selecting the `krea2` architecture preset attempts to select matching files automatically when those filenames are present.

> Architecture selection in RebelUI is a conditioning/UI preset. The diffusion-weight architecture itself is still detected by ComfyUI's loader. This avoids pretending RebelUI can safely force a model class that ComfyUI does not support.

## Folder layout

```text
rebelui/
├─ server.py
├─ README.md
└─ static/
   ├─ index.html
   └─ rebelui-logo.svg
```

Your ComfyUI models can remain in their normal locations:

```text
ComfyUI/models/
├─ diffusion_models/
├─ text_encoders/
└─ vae/
```

GGUF files may also be discovered from ComfyUI-GGUF registered folders.

## Requirements

- Current working ComfyUI install
- `aiohttp` and `pillow` in the Python environment
- ComfyUI-GGUF for `.gguf` models or `.gguf` text encoders
- `ffmpeg` on PATH only if you use video output

For a portable Windows build:

```bat
<COMFY>\python_embeded\python.exe -m pip install aiohttp pillow --no-deps
```

## Run

From the ComfyUI directory:

```bat
cd /d <COMFY>\ComfyUI
..\python_embeded\python.exe <path>\rebelui\server.py
```

Or pass the ComfyUI folder explicitly:

```bat
python server.py --comfy <COMFY>\ComfyUI
```

Then open:

```text
http://127.0.0.1:8199
```

## Important limitation

RebelUI's generic direct sampler handles ordinary text conditioning and a generic image/video latent. Model families requiring extra graph plumbing—reference-image encoders, control/reference latents, audio conditioning, specialized guider nodes, LoRA routing, or family-specific latent preparation—still need dedicated handling in `Runner.run`.


## Memory policy

RebelUI runs ComfyUI as a separate Python process, so it does not inherit flags
from your normal ComfyUI launcher. This build leaves ComfyUI dynamic VRAM enabled
and disables pinned host memory by default.

Use `--enable-pinned-memory` only if you explicitly want pinned memory in RebelUI.
The effective dynamic-VRAM and pinned-memory state is printed in the RebelUI log
at startup.

# RebelUI

> **ComfyUI without living in the node graph.**  
> A lightweight local interface for generating with the models already installed in ComfyUI — now with a built-in model Auto-Quantizer.

**No separate model library. No hosted backend. No telemetry. No nodes for ordinary generation. Just pick your models, prompt, and generate.**

---

## What is RebelUI?

RebelUI is a local web interface built directly on top of ComfyUI's Python backend.

It uses ComfyUI for model loading, conditioning, sampling, VRAM management, and VAE decoding, while presenting those capabilities through a simpler generation interface. Your existing ComfyUI model folders remain the source of truth.

RebelUI supports normal PyTorch/safetensors models and GGUF models in the same interface, and includes a dedicated native execution path for Krea 2 Turbo.

The project also includes an **experimental Auto-Quantizer** for creating lower-memory model variants directly from RebelUI.

---

## Features

### Generate

- Uses your existing ComfyUI installation
- Direct model discovery from ComfyUI model folders
- Safetensors / full-weight model support
- Native INT8 / FP8 model loading through ComfyUI
- GGUF diffusion-model support through ComfyUI-GGUF
- GGUF text-encoder support
- `ALL / FULL / GGUF` model filtering
- Dynamic CLIP-type discovery from the installed ComfyUI build
- Dynamic sampler and scheduler discovery
- Architecture presets
- Dedicated native Krea 2 Turbo execution path
- Prompt and negative-prompt controls
- Width, height, steps, CFG, seed, sampler and scheduler controls
- Batch/image output
- Video latent/output support for compatible model paths
- Live VRAM information
- Persistent UI settings
- Resizable panels and generation log
- Multiple UI themes
- Local output viewer

### Auto-Quantize — Beta

The Quantize tab can inspect a local model or supported public Hugging Face repository before conversion.

It can:

- Probe safetensors headers without loading the complete checkpoint
- Detect repository/model layout
- Fingerprint known model architectures
- Select architecture-specific quantization recipes
- Protect sensitive tensors from aggressive quantization
- Estimate output size and available disk space
- Require confirmation before conversion
- Stream conversion progress directly to the UI
- Process W4A8/INT8 source shards individually
- Build multiple GGUF quantization tiers from a compatible single-file source

Supported output targets:

| Output | Container |
|---|---|
| INT8 | Safetensors |
| W4A8 | Safetensors |
| Q8_0 | GGUF |
| Q6_K | GGUF |
| Q5_K_M | GGUF |
| Q4_K_M | GGUF |
| Q4_K_S | GGUF |
| Q3_K_M | GGUF |
| Q2_K | GGUF |

> **The Auto-Quantizer is beta software.**
>
> Quantization is architecture-sensitive. A converted model can load successfully while still showing quality changes such as color shifts, weaker prompt adherence, damaged conditioning, detail loss, or other behavioral differences. Test new quantizations against the original model before deleting source weights.

---

## How RebelUI Works

RebelUI does **not** replace ComfyUI.

It runs as its own Python process and imports ComfyUI's Python libraries directly.

```text
Your ComfyUI installation
        │
        ├── models/diffusion_models
        ├── models/text_encoders
        ├── models/vae
        ├── ComfyUI model loaders
        ├── samplers / schedulers
        └── VRAM management
                 │
                 ▼
              RebelUI
          ┌──────┴──────┐
          │             │
       Generate      Quantize
```

For ordinary generation, this means you do not need to maintain another copy of your model library just for RebelUI.

---

## Krea 2 Turbo

RebelUI contains a dedicated Krea 2 execution path that mirrors the native ComfyUI node sequence:

```text
UNETLoader
    ↓
CLIPLoader (krea2)
    ↓
CLIPTextEncode
    ↓
ConditioningZeroOut
    ↓
EmptyLatentImage
    ↓
KSampler
    ↓
VAELoader
    ↓
VAEDecode
```

A typical Krea 2 Turbo configuration is:

| Setting | Value |
|---|---|
| Architecture | `krea2` |
| Diffusion model | `krea2_turbo_int8_convrot.safetensors` or compatible FP8/full build |
| Text encoder | `qwen3vl_4b_fp8_scaled.safetensors` |
| CLIP type | `krea2` |
| VAE | `qwen_image_vae.safetensors` |
| Frames | `1` |
| Steps | `8` |
| CFG | `1` |
| Sampler | `euler` |
| Scheduler | `simple` |

Krea 2 FULL/INT8/FP8 generation uses the native ComfyUI node path. GGUF models use RebelUI's GGUF-capable generic path.

Architecture selection in the UI is a preset/routing choice. The actual diffusion architecture is still detected by ComfyUI's loader.

---

## Model Discovery

RebelUI scans the model locations registered with your ComfyUI installation.

Typical locations are:

```text
ComfyUI/
└── models/
    ├── diffusion_models/
    ├── text_encoders/
    └── vae/
```

GGUF files can also be discovered from folders registered by ComfyUI-GGUF.

RebelUI scans the actual registered directories in addition to ComfyUI's normal filename lists. This allows `.gguf` files to remain visible even when a ComfyUI category's extension filter does not normally expose them.

---

## Auto-Quantizer

### Workflow

```text
Model / Hugging Face repo
          ↓
Resolve source
          ↓
Probe safetensors headers
          ↓
Detect layout + architecture
          ↓
Select quantization recipe
          ↓
Protect sensitive tensors
          ↓
Estimate disk requirements
          ↓
Choose output tiers
          ↓
Explicit confirmation
          ↓
Convert
      ┌───┴───────────┐
      │               │
Safetensors          GGUF
INT8 / W4A8      Q2_K → Q8_0
```

The quantizer is designed to avoid blindly applying one rule to every architecture. Known model families use recipes containing architecture signatures, exclusion rules, group sizes, INT8 routing, high-precision tensor rules, and GGUF handling information.

Unknown architectures should be treated conservatively.

### Included quantization toolchain

RebelUI expects the following layout:

```text
RebelUI/
├── server.py
├── quantizer.py
├── AUTO_QUANTIZER.md
├── static/
│   ├── index.html
│   ├── rebelui-logo.svg
│   └── ...
└── tools/
    ├── w4a8_convert.py
    ├── st_to_gguf.py
    ├── gguf_swap_hiprec.py
    ├── gguf_set_config.py
    ├── gguf_fix_shapes.py
    └── llama/
        └── llama-quantize.exe
```

The runtime resolves these tools relative to RebelUI's own directory. Development/testing builds can override the locations with:

```text
REBELUI_QUANT_TOOLS
REBELUI_LLAMA_QUANTIZE
```

### Memory-conscious conversion

The quantizer is designed for systems where loading an entire large checkpoint into system RAM is undesirable.

Remote probing reads safetensors headers rather than loading complete model weights. W4A8/INT8 conversion processes source shards individually.

### Hugging Face privacy

For remote model inspection:

- No Hugging Face token UI is provided
- Hub access is anonymous
- Cached Hugging Face credentials are not intentionally used by RebelUI
- Gated/private repositories are refused
- RebelUI has no hosted quantization service
- No RebelUI telemetry or analytics service is used

For a gated/private model, download it yourself after accepting its license and point RebelUI at the local model.

### Remote sharded GGUF limitation

W4A8 and INT8 can process sharded safetensors sources shard-by-shard.

The current GGUF conversion path expects a single safetensors source. RebelUI therefore intentionally refuses a **remote sharded → GGUF** conversion instead of silently downloading and merging the entire checkpoint.

A local compatible single-file safetensors model can build the GGUF ladder.

---

## Architecture Recipes

The Auto-Quantizer contains recipes/fingerprints for multiple modern model families, including families such as:

- Flux
- Qwen Image
- MiniMax H3
- Wan
- LTX
- NextDiT / Z-Image / Lumina-style architectures
- MiniMax Music
- SenseNova unified architectures

Recipe support does **not** mean every checkpoint in a family has been exhaustively validated.

Different releases can introduce new tensor names, conditioning paths, shapes, or metadata requirements. Recipes will continue to be refined as models are tested.

### Why high-precision exclusions matter

Some tensors are disproportionately sensitive to quantization, including architecture-dependent:

```text
conditioning
embeddings
modulation
normalization
input projections
output/final projections
RoPE/frequency paths
adaptive normalization
```

RebelUI recipes can keep those paths at higher precision while quantizing the larger transformer weights.

---

## Requirements

### Generation

- Windows is the primary currently tested platform
- A working, reasonably current ComfyUI installation
- Python environment capable of importing that ComfyUI installation
- `aiohttp`
- `Pillow`
- Compatible PyTorch/CUDA stack for your ComfyUI installation
- ComfyUI-GGUF if you want to load GGUF diffusion models or GGUF text encoders
- `ffmpeg` on PATH if using RebelUI's video-output path

For ComfyUI portable on Windows, dependencies can be installed into the embedded Python environment if they are not already available:

```bat
D:\path\to\ComfyUI_windows_portable\python_embeded\python.exe -m pip install aiohttp pillow
```

Use the Python environment belonging to the ComfyUI installation RebelUI will run against.

---

## Installation

### 1. Clone RebelUI

```bat
git clone https://github.com/RealRebelAI/RebelUI.git
cd RebelUI
```

Or download the repository ZIP and extract it.

### 2. Make sure ComfyUI works normally

RebelUI depends on ComfyUI's backend. If your ComfyUI installation itself cannot load a model, RebelUI will not magically make that model compatible.

Update ComfyUI when using newly released architectures that require newer loader support.

### 3. Install required Python packages if necessary

Example for ComfyUI portable:

```bat
D:\AI_Tools\ComfyUI_windows_portable\python_embeded\python.exe -m pip install aiohttp pillow huggingface_hub gguf
```

### 4. Start RebelUI

Example:

```bat
D:\AI_Tools\ComfyUI_windows_portable\python_embeded\python.exe C:\path\to\RebelUI\server.py --comfy D:\AI_Tools\ComfyUI_windows_portable\ComfyUI
```

Then open:

```text
http://127.0.0.1:8199
```

---

## Optional Windows Launcher

You can create `start_REBELUI.bat`:

```bat
@echo off
title RebelUI

set "PYTHON=D:\AI_Tools\ComfyUI_windows_portable\python_embeded\python.exe"
set "REBELUI=C:\path\to\RebelUI"
set "COMFY=D:\AI_Tools\ComfyUI_windows_portable\ComfyUI"

cd /d "%REBELUI%"

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8199"

"%PYTHON%" "%REBELUI%\server.py" --comfy "%COMFY%" --port 8199

pause
```

Change the three paths to match your installation.

You can create a normal Windows shortcut to this BAT file for one-click startup.

---

## Command-Line Options

Basic launch:

```text
python server.py --comfy <ComfyUI directory>
```

Useful options include:

```text
--comfy <path>              Path to ComfyUI
--out <path>                RebelUI output directory
--port <number>             HTTP port (default: 8199)
--listen <address>          Listen address (default: 127.0.0.1)
--disable-pinned-memory     Disable pinned host memory
--enable-pinned-memory      Allow pinned host memory
```

By default RebelUI binds to localhost.

---

## Memory / VRAM Behavior

RebelUI is a separate Python process, so it does not automatically inherit every command-line flag used by your normal ComfyUI launcher.

The current server:

- Leaves ComfyUI dynamic VRAM behavior enabled
- Disables pinned host memory by default
- Runs generation under `torch.inference_mode()`
- Uses ComfyUI's model loading and device-management paths
- Lets ComfyUI manage VAE device placement/offloading

You can enable pinned memory explicitly with:

```text
--enable-pinned-memory
```

---

## GGUF Support

GGUF generation requires **ComfyUI-GGUF** in the ComfyUI installation RebelUI is using.

RebelUI can discover GGUF files in registered diffusion-model and text-encoder folders and initialize the ComfyUI-GGUF loader when needed.

If RebelUI reports that ComfyUI-GGUF is not installed, install/configure ComfyUI-GGUF in the underlying ComfyUI installation first.

The Auto-Quantizer's GGUF creation pipeline is separate from GGUF inference support.

---

## Current Limitations

RebelUI intentionally does not pretend every ComfyUI graph can be reduced to a few dropdowns.

The generic generation path currently focuses on ordinary text conditioning and generic image/video latent generation. Model families requiring additional graph-specific plumbing may need dedicated RebelUI execution paths.

Examples include workflows requiring:

- Reference-image encoders
- ControlNet/reference latents
- Specialized guider nodes
- Audio conditioning
- LoRA routing
- Family-specific latent preparation
- Complex multi-stage pipelines

Krea 2 is an example of a model family that already has a dedicated execution path.

### Quantizer limitations

- Auto-Quantizer is beta
- Architecture recipes are still being validated
- Unknown/new architectures may require recipe updates
- Remote sharded GGUF conversion is intentionally blocked
- Successful conversion does not guarantee identical model behavior
- Always keep the original model until the quantized version has been tested

---

## Reporting Quantization Problems

If a quantized model loads but behaves differently from the original, open a GitHub Issue.

Please include:

```text
Model:
Source repository / model filename:
Architecture:
Quantization: INT8 / W4A8 / Q4_K_M / etc.
GPU:
System RAM:
ComfyUI version:
RebelUI version / commit:

Prompt:
Seed:
Steps:
CFG:
Sampler:
Scheduler:
VAE:
Text encoder:

What changed:
```

When possible, attach comparison images generated with the **same prompt, seed, VAE, encoder, sampler, scheduler, steps, and CFG**.

Useful symptoms to report include:

- Color/tint shifts
- Prompt-adherence changes
- Broken text rendering
- Face/detail degradation
- Abnormal contrast
- Conditioning failures
- Unexpected artifacts
- GGUF load failures
- Shape/metadata errors
- Conversion failures

Those reports can be used to improve the architecture-specific recipe rather than weakening quantization rules globally.

---

## Project Structure

```text
RebelUI/
├── server.py                  # ComfyUI generation backend + HTTP API
├── quantizer.py               # Auto-Quantizer backend
├── README.md
├── AUTO_QUANTIZER.md
│
├── static/
│   ├── index.html             # RebelUI frontend
│   ├── rebelui-logo.svg
│   └── ...
│
├── tools/
│   ├── w4a8_convert.py
│   ├── st_to_gguf.py
│   ├── gguf_swap_hiprec.py
│   ├── gguf_set_config.py
│   ├── gguf_fix_shapes.py
│   └── llama/
│       └── llama-quantize.exe
│
└── output/                    # Created locally for generated output
```

---

## Troubleshooting

### RebelUI opens but models are missing

Make sure `--comfy` points to the correct `ComfyUI` directory.

RebelUI discovers models from the folders registered by that installation.

### A new architecture will not load

Update ComfyUI first. RebelUI relies on ComfyUI's underlying architecture detection and model loaders.

### GGUF model is missing or fails to load

Confirm that ComfyUI-GGUF is installed in the same ComfyUI installation RebelUI is using.

### Krea 2 fails to load

Check that:

- Architecture is `krea2`
- The selected encoder supports the Krea 2 workflow
- CLIP type is `krea2`
- The correct Qwen Image VAE is selected
- Your ComfyUI build contains Krea 2 support

### Quantizer says a tool is missing

Check:

```text
RebelUI/tools/
RebelUI/tools/llama/
```

The required conversion scripts and `llama-quantize.exe` must be present.

### Port 8199 is already in use

Start RebelUI on another port:

```bat
python server.py --comfy C:\path\to\ComfyUI --port 8200
```

Then open:

```text
http://127.0.0.1:8200
```

---

## Security & Privacy

RebelUI is designed as a local application.

By default the web server listens on:

```text
127.0.0.1
```

Generation happens through your local ComfyUI/PyTorch environment.

The Auto-Quantizer can contact Hugging Face when you explicitly provide a public Hugging Face model/repository for inspection or conversion. It does not provide a RebelUI-hosted model service or token-storage system.

Do not expose RebelUI to an untrusted network by changing `--listen` unless you understand the security implications.

---

## Development Status

RebelUI is under active development.

The generation interface is usable today, while support for additional architecture-specific workflows will continue to expand.

The Auto-Quantizer should currently be considered **beta** while its model-family recipes are tested against more checkpoints.

If something breaks, an Issue with reproducible settings is much more useful than simply reporting that a model "looks wrong."

---

## Contributing

Bug reports, architecture compatibility reports, quantization comparisons, and pull requests are welcome.

When contributing support for a new architecture, avoid globally changing quantization behavior when the issue is architecture-specific. Prefer a dedicated recipe or execution path.

For quantization changes, test against the original model using identical generation settings whenever possible.

---

## Credits

RebelUI is built around the ComfyUI ecosystem and depends on ComfyUI for its core model loading, sampling, memory management, and VAE functionality.

GGUF inference support relies on ComfyUI-GGUF when GGUF models or encoders are used.

The bundled/used quantization components have their own upstream implementations and licenses. Review and preserve the applicable upstream license and attribution files when redistributing third-party source code or compiled binaries.

---

## License

RebelUI's own license should be defined by the repository's `LICENSE` file.

**Before redistributing bundled third-party quantization scripts or compiled binaries, make sure the repository includes all licenses, copyright notices, and attribution required by their respective upstream projects.**

---

## About

**RebelUI — no nodes, no fluff, just generate.**

Built by [RealRebelAI](https://github.com/RealRebelAI).

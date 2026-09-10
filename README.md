# RebelUI

A direct-to-library runner for ComfyUI's core.

RebelUI skips the graph engine and calls ComfyUI's model loading,
conditioning, sampling, memory management, and VAE code directly.

**No nodes. No fluff. Just generate.**

RebelUI also includes a built-in **Auto-Quantizer** for creating INT8,
W4A8, and GGUF model variants.

> **Auto-Quantizer status:** Beta. Quantization behavior can vary by
> architecture. Always compare a quantized model against the original
> before distributing or relying on it.

------------------------------------------------------------------------

## Features

### Generation

-   Direct generation through ComfyUI's Python backend
-   Full-weight, FP8, INT8, and GGUF model discovery
-   Diffusion-model filter: All / Full / GGUF
-   Text-encoder filter: All / Full / GGUF
-   GGUF text encoders can be used with supported safetensors diffusion
    models
-   Encoder types are read from the installed ComfyUI `CLIPType` enum
-   Sampler and scheduler lists are read from the installed ComfyUI
    build
-   Architecture browser populated from supported ComfyUI model classes
-   Krea 2 preset and native generation path
-   Configurable model, encoder, VAE, resolution, steps, CFG, sampler,
    scheduler, and seed
-   Generated output preview
-   Dynamic VRAM support
-   Pinned host memory disabled by default

### Auto-Quantizer

Current output options include:

  Format   Output
  -------- ----------------
  INT8     `.safetensors`
  W4A8     `.safetensors`
  Q8_0     `.gguf`
  Q6_K     `.gguf`
  Q5_K_M   `.gguf`
  Q4_K_M   `.gguf`
  Q4_K_S   `.gguf`
  Q3_K_M   `.gguf`
  Q2_K     `.gguf`

The quantizer probes the source before conversion and can apply
architecture-aware recipes.

------------------------------------------------------------------------

# Requirements

RebelUI currently targets **ComfyUI Windows Portable**.

You need:

-   A current working ComfyUI Windows Portable installation
-   Git, or the ability to download the repository ZIP
-   `ffmpeg` on PATH only if you use video output
-   ComfyUI-GGUF if you want to **load/run** `.gguf` diffusion models or
    `.gguf` text encoders

RebelUI uses ComfyUI's existing embedded Python environment. Do **not**
install a separate PyTorch build just for RebelUI.

------------------------------------------------------------------------

# Installation

## 1. Verify ComfyUI first

Make sure your normal ComfyUI installation launches and generates
successfully before installing RebelUI.

A standard portable installation contains:

``` text
ComfyUI_windows_portable/
├── python_embeded/
└── ComfyUI/
    ├── main.py
    ├── models/
    └── ...
```

## 2. Open Command Prompt inside the `ComfyUI` folder

Open:

``` text
ComfyUI_windows_portable\ComfyUI
```

in File Explorer.

Click the File Explorer address bar, type:

``` text
cmd
```

and press **Enter**.

Your terminal should now be inside the actual ComfyUI directory.

## 3. Clone RebelUI directly into ComfyUI

Run:

``` bat
git clone https://github.com/RealRebelAI/RebelUI.git RebelUI
```

Your installation will now look like:

``` text
ComfyUI_windows_portable/
├── python_embeded/
└── ComfyUI/
    ├── main.py
    ├── models/
    └── RebelUI/
        ├── server.py
        ├── quantizer.py
        ├── requirements.txt
        ├── static/
        └── tools/
```

This location is intentional. It allows RebelUI to use universal
relative commands without requiring a drive letter, Windows username, or
custom path.

### No Git?

Download the repository ZIP from GitHub, extract it, rename the
extracted folder to:

``` text
RebelUI
```

and place it directly inside:

``` text
ComfyUI_windows_portable\ComfyUI
```

The final location must be:

``` text
ComfyUI_windows_portable\ComfyUI\RebelUI
```

------------------------------------------------------------------------

# Install Dependencies

Open Command Prompt **inside the RebelUI folder**.

An easy way is to open the `RebelUI` folder in File Explorer, click the
address bar, type `cmd`, and press Enter.

Then run:

``` bat
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

The repository's `requirements.txt` should contain the additional
RebelUI/quantizer dependencies:

``` text
aiohttp
Pillow
huggingface_hub
gguf
```

RebelUI uses the PyTorch and NumPy environment already provided by
ComfyUI.

Do not blindly reinstall PyTorch, because doing so can replace the
CUDA/PyTorch configuration already working with ComfyUI.

------------------------------------------------------------------------

# Start RebelUI

From Command Prompt **inside `ComfyUI\RebelUI`**, run:

``` bat
..\..\python_embeded\python.exe server.py --comfy ..
```

Then open:

``` text
http://127.0.0.1:8199
```

No placeholders are required.

From the RebelUI folder:

``` text
..                         = ComfyUI
..\..\python_embeded       = ComfyUI_windows_portable\python_embeded
```

Therefore the same commands work whether ComfyUI is installed on `C:`,
`D:`, `E:`, or another location.

------------------------------------------------------------------------

# Quick Start

If ComfyUI Windows Portable is already installed and working:

### Open Command Prompt inside:

``` text
ComfyUI_windows_portable\ComfyUI
```

### Clone RebelUI:

``` bat
git clone https://github.com/RealRebelAI/RebelUI.git RebelUI
```

### Enter RebelUI:

``` bat
cd RebelUI
```

### Install dependencies:

``` bat
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

### Start RebelUI:

``` bat
..\..\python_embeded\python.exe server.py --comfy ..
```

### Open:

``` text
http://127.0.0.1:8199
```

That's the complete Windows Portable setup.

------------------------------------------------------------------------

# Optional Windows Launcher

Create a file named:

``` text
start_REBELUI.bat
```

inside:

``` text
ComfyUI\RebelUI
```

Paste:

``` bat
@echo off
title RebelUI
setlocal

cd /d "%~dp0"

if not exist "..\..\python_embeded\python.exe" (
    echo.
    echo ERROR: ComfyUI embedded Python was not found.
    echo.
    echo RebelUI should be installed here:
    echo   ComfyUI_windows_portable\ComfyUI\RebelUI
    echo.
    pause
    exit /b 1
)

if not exist "..\main.py" (
    echo.
    echo ERROR: ComfyUI was not found one directory above RebelUI.
    echo.
    echo Expected:
    echo   ComfyUI_windows_portable\ComfyUI\RebelUI
    echo.
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8199"

"..\..\python_embeded\python.exe" server.py --comfy ".." --port 8199

echo.
echo RebelUI stopped.
pause

endlocal
```

Because the launcher uses `%~dp0`, it first switches to its own
directory. It can therefore be double-clicked without depending on the
directory from which Windows launched it.

------------------------------------------------------------------------

# Updating RebelUI

If RebelUI was installed with Git, open Command Prompt inside:

``` text
ComfyUI\RebelUI
```

and run:

``` bat
git pull
```

If `requirements.txt` changed in an update, run:

``` bat
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Then restart RebelUI.

------------------------------------------------------------------------

# Krea 2 Turbo

For the current Krea 2 Turbo local workflow, a typical configuration is:

``` text
Architecture: krea2
Diffusion model: krea2_turbo_int8_convrot.safetensors or supported FP8 build
Text encoder: qwen3vl_4b_fp8_scaled.safetensors
CLIP type: krea2
VAE: qwen_image_vae.safetensors
Frames: 1
Steps: 8
CFG: 1
Sampler: Euler
Scheduler: Simple
```

Selecting the `krea2` architecture preset attempts to select matching
files automatically when those filenames are present.

Architecture selection in RebelUI is a conditioning/UI preset. The
diffusion-weight architecture itself is still detected by ComfyUI's
loader.

------------------------------------------------------------------------

# Model Discovery

Your models stay in normal ComfyUI model locations, such as:

``` text
ComfyUI\models\
├── diffusion_models\
├── text_encoders\
└── vae\
```

RebelUI also scans registered ComfyUI model locations and can expose
`.gguf` files that may otherwise be hidden by normal extension
filtering.

Models do **not** need to be copied into RebelUI.

------------------------------------------------------------------------

# Using the Auto-Quantizer

Open RebelUI and select the **Quantize** tab.

A basic first-time workflow is:

1.  Select a supported local model or public Hugging Face source.
2.  Click **FETCH + PROBE**.
3.  Review the detected model and architecture information.
4.  Select one output tier. INT8 is a good first test.
5.  Select a separate output directory.
6.  Check the confirmation option.
7.  Start quantization.
8.  Load and compare the resulting model against the original.

Do not overwrite your only copy of a source model.

------------------------------------------------------------------------

# Testing a Quantized Model

Use the same settings for the original and quantized model:

-   prompt
-   seed
-   text encoder
-   VAE
-   resolution
-   steps
-   CFG
-   sampler
-   scheduler

A conversion completing successfully does **not** guarantee that the
model's visual behavior survived quantization correctly.

If colors, composition, details, conditioning, or output quality are
obviously wrong, report the architecture and model.

------------------------------------------------------------------------

# INT8

INT8 produces an 8-bit `.safetensors` model using RebelUI's conversion
pipeline.

Conceptually:

``` text
Original model
      ↓
Fetch + Probe
      ↓
Architecture detection
      ↓
INT8 conversion
      ↓
INT8 .safetensors
      ↓
Test against original
```

Architecture-specific recipes can protect sensitive tensors at higher
precision.

------------------------------------------------------------------------

# W4A8

W4A8 produces a lower-precision `.safetensors` model.

It can reduce model size more aggressively than INT8 but may also be
more sensitive to architecture-specific settings.

Always validate the resulting model.

------------------------------------------------------------------------

# GGUF

Current GGUF output tiers:

``` text
Q8_0
Q6_K
Q5_K_M
Q4_K_M
Q4_K_S
Q3_K_M
Q2_K
```

RebelUI's bundled quantization tools live under:

``` text
RebelUI\tools\
```

The bundled llama quantizer is expected under:

``` text
RebelUI\tools\llama\llama-quantize.exe
```

The GGUF conversion path is approximately:

``` text
safetensors
      ↓
temporary F16 GGUF
      ↓
llama-quantize
      ↓
selected GGUF tier
      ↓
metadata / high-precision / shape processing
```

## Current GGUF limitation

GGUF creation currently expects a supported single-file safetensors
source.

Remote sharded Hugging Face repositories should not be assumed to
support direct GGUF conversion.

Creating a GGUF file and **running** a GGUF model are separate
operations. To run GGUF models, ComfyUI must have compatible GGUF loader
support.

------------------------------------------------------------------------

# Architecture-Aware Quantization

Different model architectures do not necessarily tolerate identical
quantization rules.

RebelUI recipes can control:

-   architecture signature detection
-   exclusion patterns
-   INT8 tensor selection
-   W4A8 eligibility
-   group size
-   GGUF architecture metadata
-   high-precision tensors
-   configuration metadata
-   known shape handling

Unknown architectures should be treated as experimental.

------------------------------------------------------------------------

# Hugging Face

RebelUI can inspect/download supported public Hugging Face model
repositories.

Remote access uses `huggingface_hub`.

Public repositories do not require the user to enter a Hugging Face
token into RebelUI.

Private or gated repositories may require users to accept the
repository's terms and obtain the model files themselves.

RebelUI is not intended to bypass gated repository access.

------------------------------------------------------------------------

# Memory and Storage

Quantizing large models can require substantial:

-   system RAM
-   disk space
-   temporary storage
-   processing time

RebelUI performs probe/preflight work before conversion.

INT8/W4A8 processing is designed to avoid unnecessarily loading an
entire sharded checkpoint at once where supported.

GGUF conversion can require significant temporary storage because an
intermediate GGUF may be produced before the final quantized tier.

------------------------------------------------------------------------

# Memory Policy

RebelUI runs ComfyUI through its own Python process, so it does not
automatically inherit every flag from your normal ComfyUI launcher.

The current build leaves ComfyUI dynamic VRAM enabled and disables
pinned host memory by default.

Use:

``` text
--enable-pinned-memory
```

only if you explicitly want pinned memory in RebelUI.

The effective dynamic-VRAM and pinned-memory state is printed in the
RebelUI startup log.

------------------------------------------------------------------------

# Important Generation Limitation

RebelUI's generic direct sampler handles ordinary text conditioning and
a generic image/video latent.

Model families requiring additional graph plumbing---such as
reference-image encoders, control/reference latents, audio conditioning,
specialized guider nodes, LoRA routing, or family-specific latent
preparation---still require dedicated handling in RebelUI.

------------------------------------------------------------------------

# Troubleshooting

## `ModuleNotFoundError: No module named 'comfy'`

Do not launch the current Windows Portable setup using:

``` bat
python server.py
```

That can use your system Python instead of ComfyUI's environment.

From inside `ComfyUI\RebelUI`, use:

``` bat
..\..\python_embeded\python.exe server.py --comfy ..
```

------------------------------------------------------------------------

## `The system cannot find the path specified`

Verify the exact layout:

``` text
ComfyUI_windows_portable/
├── python_embeded/
└── ComfyUI/
    └── RebelUI/
```

Then open Command Prompt inside `RebelUI` before running the
launch/dependency commands.

------------------------------------------------------------------------

## Missing RebelUI dependency

From inside `ComfyUI\RebelUI`, run:

``` bat
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Then restart RebelUI.

------------------------------------------------------------------------

## Model does not appear

Verify that:

1.  Normal ComfyUI can see the model.
2.  The model is in a registered ComfyUI model location.
3.  Required custom loader support is installed.
4.  RebelUI has been restarted or its model list refreshed.

------------------------------------------------------------------------

## GGUF model does not load

The Auto-Quantizer can create GGUF files, but inference depends on
compatible ComfyUI GGUF loader support.

Verify that the required GGUF support is installed and that the model
architecture itself is supported.

------------------------------------------------------------------------

## Quantized model looks wrong

Compare against the original using identical settings.

For a useful issue report, include:

``` text
Model:
Architecture:
Source format:
Quantization tier:
Prompt:
Seed:
Resolution:
Steps:
CFG:
Sampler:
Scheduler:
Text encoder:
VAE:
Observed problem:
```

Original-vs-quantized comparison images are especially helpful.

------------------------------------------------------------------------

# Project Layout

After installation:

``` text
ComfyUI_windows_portable/
├── python_embeded/
└── ComfyUI/
    ├── main.py
    ├── models/
    └── RebelUI/
        ├── server.py
        ├── quantizer.py
        ├── requirements.txt
        ├── README.md
        ├── AUTO_QUANTIZER.md
        ├── static/
        └── tools/
            ├── w4a8_convert.py
            ├── st_to_gguf.py
            ├── gguf_swap_hiprec.py
            ├── gguf_set_config.py
            ├── gguf_fix_shapes.py
            └── llama/
                └── llama-quantize.exe
```

------------------------------------------------------------------------

# Privacy

RebelUI is intended to run locally.

Local generation and local model quantization remain on the user's
machine.

When a public Hugging Face repository is supplied, RebelUI communicates
with Hugging Face as necessary to inspect or download the requested
model files.

------------------------------------------------------------------------

# Development Status

RebelUI is actively developed.

The Auto-Quantizer should currently be considered **beta**.

Architecture-specific testing and issue reports are welcome.

------------------------------------------------------------------------

# Contributing

Issues and pull requests are welcome.

For quantization issues, include the model, architecture, quantization
tier, logs, and an original-vs-quantized comparison whenever possible.

------------------------------------------------------------------------

# Third-Party Components

RebelUI uses and/or interoperates with third-party software.

Third-party libraries, scripts, binaries, model formats, and models
remain subject to their respective licenses and terms.

Preserve required upstream licenses and notices when redistributing
bundled third-party components.

------------------------------------------------------------------------

# License

RebelUI's own source code is governed by the license included in this
repository.

ComfyUI, third-party components, model files, and generated
quantizations may have separate licenses or terms. Review those licenses
before redistribution.

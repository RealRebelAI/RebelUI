# RebelUI

**RebelUI** is a lightweight, direct-generation frontend for
[ComfyUI](https://github.com/Comfy-Org/ComfyUI). It is designed for
users who want to use their existing ComfyUI installation, models,
loaders, samplers, VRAM management, and output pipeline without building
or navigating a node graph for every generation.

RebelUI provides a focused desktop-style interface for selecting models,
entering prompts, configuring sampling, monitoring GPU memory, viewing
outputs, and restoring your previous session.

> **Project status:** Beta / active development. RebelUI is usable for
> supported direct-generation workflows, but it is not intended to
> reproduce every possible ComfyUI graph.

------------------------------------------------------------------------

## Table of Contents

-   [Features](#features)
-   [How RebelUI Works](#how-rebelui-works)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Running RebelUI](#running-rebelui)
-   [Windows Launcher](#windows-launcher)
-   [Model Folders](#model-folders)
-   [Model Formats](#model-formats)
-   [Krea 2 / Krea 2 Turbo](#krea-2--krea-2-turbo)
-   [Interface](#interface)
-   [Live VRAM Monitor](#live-vram-monitor)
-   [Persistent Session Memory](#persistent-session-memory)
-   [Themes and Settings](#themes-and-settings)
-   [Outputs](#outputs)
-   [GGUF Support](#gguf-support)
-   [Architecture Presets](#architecture-presets)
-   [Troubleshooting](#troubleshooting)
-   [Known Limitations](#known-limitations)
-   [Contributing](#contributing)
-   [License](#license)
-   [Credits](#credits)

------------------------------------------------------------------------

## Features

-   Direct integration with an existing ComfyUI installation
-   Uses your existing ComfyUI model directories
-   Automatic discovery of diffusion models, text encoders, and VAEs
-   FULL and GGUF format filters
-   Safetensors, FP8, INT8, and other non-GGUF weights treated as FULL
-   GGUF discovery inside standard ComfyUI model directories
-   Dynamic sampler and scheduler discovery from the installed ComfyUI
    version
-   Architecture / conditioning presets
-   Dedicated native ComfyUI-node execution path for Krea 2
-   Krea 2 Turbo INT8 / FP8 workflow support
-   Live CUDA GPU-memory monitoring
-   ComfyUI dynamic-VRAM integration
-   Persistent model and generation settings
-   Persistent UI layout settings
-   Resizable left and right sidebars
-   Independently scrollable left and right control rails
-   Resizable positive and negative prompt fields
-   Resizable Run Log
-   Hide/show model and control sidebars
-   Contained, aspect-ratio-correct output viewer
-   Output image saving and automatic browser display
-   Multiple UI themes
-   Comfortable and compact density modes
-   Windows launcher support
-   No separate model database or model-copying step

------------------------------------------------------------------------

## How RebelUI Works

RebelUI is **not another node editor**.

ComfyUI remains the generation engine. RebelUI provides a simpler
frontend for direct workflows.

The general relationship is:

``` text
RebelUI
   │
   ├── User interface
   ├── Session persistence
   ├── Model selection
   ├── Prompt controls
   ├── Sampling controls
   ├── Live VRAM display
   └── Output viewer
          │
          ▼
       ComfyUI
          │
          ├── Model loading
          ├── Architecture detection
          ├── Text encoding
          ├── Sampling
          ├── Dynamic VRAM
          ├── VAE decoding
          └── CUDA / PyTorch runtime
```

RebelUI intentionally relies on ComfyUI for model compatibility rather
than attempting to replace ComfyUI's underlying model system.

------------------------------------------------------------------------

## Requirements

You need a working ComfyUI installation.

### Required

-   Python environment compatible with your ComfyUI installation
-   ComfyUI
-   PyTorch / CUDA configured through ComfyUI
-   `aiohttp`
-   Pillow

### Optional

-   ComfyUI-GGUF for `.gguf` model support
-   FFmpeg for workflows or outputs that require video handling

A Windows ComfyUI Portable installation can use its included embedded
Python.

------------------------------------------------------------------------

## Installation

### 1. Install and verify ComfyUI

Make sure normal ComfyUI already launches and can generate successfully
before troubleshooting RebelUI.

A typical Windows Portable installation might look like:

``` text
D:\AI_Tools\ComfyUI_windows_portable\
├── ComfyUI\
└── python_embeded\
```

### 2. Install RebelUI

Place the RebelUI directory wherever you want.

Example:

``` text
D:\rebelui\
```

Typical project structure:

``` text
rebelui\
├── server.py
├── README.md
└── static\
    ├── index.html
    ├── rebelui-logo-generated.png
    └── rebelui-logo.svg
```

### 3. Install Python dependencies

For Windows ComfyUI Portable:

``` bat
cd /d D:\AI_Tools\ComfyUI_windows_portable\ComfyUI
..\python_embeded\python.exe -m pip install aiohttp pillow
```

If those packages are already installed in the ComfyUI Python
environment, no additional installation is necessary.

------------------------------------------------------------------------

## Running RebelUI

RebelUI should be launched with the Python environment used by ComfyUI
and from the ComfyUI directory so ComfyUI's Python modules can be
resolved correctly.

Example:

``` bat
cd /d D:\AI_Tools\ComfyUI_windows_portable\ComfyUI
..\python_embeded\python.exe D:\rebelui\server.py
```

Then open:

``` text
http://127.0.0.1:8199
```

Keep the server console open while using RebelUI. Console messages can
provide additional information if a model fails to load or generation
fails.

------------------------------------------------------------------------

## Windows Launcher

You can create a `Start RebelUI.bat` file to launch the server and
automatically open RebelUI in your default browser.

Example:

``` bat
@echo off
title RebelUI Server

cd /d D:\AI_Tools\ComfyUI_windows_portable\ComfyUI

start "" cmd /k "..\python_embeded\python.exe D:\rebelui\server.py"

timeout /t 15 /nobreak >nul

start "" http://127.0.0.1:8199

exit
```

The delay gives ComfyUI and RebelUI time to initialize before the
browser connects.

Change the paths to match your installation.

If your machine initializes RebelUI more slowly, increase:

``` bat
timeout /t 15 /nobreak >nul
```

to a larger value.

------------------------------------------------------------------------

## Model Folders

RebelUI discovers models from the directories registered with ComfyUI.

Common locations include:

``` text
ComfyUI\models\diffusion_models\
ComfyUI\models\text_encoders\
ComfyUI\models\vae\
```

After adding or removing model files, restart RebelUI so the server can
rescan the model directories.

### Diffusion Models

Examples:

``` text
ComfyUI\models\diffusion_models\model.safetensors
ComfyUI\models\diffusion_models\model.gguf
```

### Text Encoders

Examples:

``` text
ComfyUI\models\text_encoders\encoder.safetensors
ComfyUI\models\text_encoders\encoder.gguf
```

### VAEs

Example:

``` text
ComfyUI\models\vae\vae.safetensors
```

RebelUI directly scans registered model directories so GGUF files placed
in normal ComfyUI diffusion-model or text-encoder directories can still
be discovered.

------------------------------------------------------------------------

## Model Formats

RebelUI provides format filters for model selection.

### ALL

Displays all supported files discovered for that model category.

### FULL

`FULL` means a supported **non-GGUF** weight file.

Examples include:

``` text
.safetensors
.sft
.pt
.pth
.ckpt
.bin
```

Quantized safetensors are still considered FULL.

For example:

``` text
krea2_turbo_int8_convrot.safetensors
```

is a FULL model even though its weights are INT8.

Likewise, FP8 `.safetensors` files are FULL models.

### GGUF

Files ending in:

``` text
.gguf
```

are treated as GGUF weights and require a compatible ComfyUI-GGUF
installation.

------------------------------------------------------------------------

## Krea 2 / Krea 2 Turbo

RebelUI includes a dedicated Krea 2 preset and a native ComfyUI-node
execution path for supported FULL Krea 2 weights.

A typical Krea 2 Turbo setup uses:

### Diffusion model

``` text
ComfyUI\models\diffusion_models\krea2_turbo_int8_convrot.safetensors
```

### Text encoder

``` text
ComfyUI\models\text_encoders\qwen3vl_4b_fp8_scaled.safetensors
```

### VAE

``` text
ComfyUI\models\vae\qwen_image_vae.safetensors
```

### Suggested starting settings

``` text
Architecture: krea2
Steps: 8
CFG: 1
Sampler: euler
Scheduler: simple
Frames: 1
Batch: 1
```

### Native Krea 2 execution path

For supported non-GGUF Krea 2 configurations, RebelUI uses ComfyUI's own
node implementations rather than manually recreating the complete Krea
execution lifecycle.

The path is conceptually:

``` text
UNETLoader
    │
    ▼
CLIPLoader (krea2)
    │
    ▼
CLIPTextEncode
    │
    ▼
ConditioningZeroOut
    │
    ▼
EmptyLatentImage
    │
    ▼
KSampler
    │
    ▼
VAELoader
    │
    ▼
VAEDecode
```

The VAE is loaded after sampling for this path, following the dependency
order of the direct Krea workflow more closely.

### Negative prompts

The Krea 2 native path uses zeroed conditioning for the unconditional
branch. If the UI contains a negative prompt while using this path,
RebelUI may report that the negative prompt is ignored for that
workflow.

------------------------------------------------------------------------

## Interface

RebelUI is divided into three main areas.

### Model Library

The left sidebar contains model and architecture controls.

Depending on the selected tab, it can display:

-   Architecture / conditioning preset
-   Diffusion model
-   Diffusion-model format filter
-   GGUF quantization shortcuts where applicable
-   Text encoder
-   Text-encoder format filter
-   ComfyUI CLIP type
-   VAE
-   Additional setup controls

Available tabs include:

``` text
ALL
MODELS
ENCODERS
SETUP
```

The sidebar has its own scrollbar, so lower controls such as the VAE
remain accessible without scrolling the entire page.

The sidebar can also be hidden with **Hide Models**.

### Output Viewer

The center area displays the generated output.

Generated media is constrained to the viewer using
aspect-ratio-preserving scaling. Images and videos should not overflow
into the generation controls or Run Log.

The center area also includes:

-   Generate button
-   Current status
-   Clear Output
-   Clear Log
-   Run Log
-   Copy Settings

### Run Controls

The right sidebar contains generation controls.

Tabs include:

``` text
ALL
PROMPTS
SAMPLING
OUTPUT
```

Controls include:

-   Positive prompt
-   Negative prompt
-   Steps
-   CFG
-   Sampler
-   Scheduler
-   Seed
-   Seed randomization
-   Width
-   Height
-   Batch and other output options where available

The right sidebar scrolls independently from the rest of the interface.

It can be hidden with **Hide Controls**.

------------------------------------------------------------------------

## Resizing the Interface

### Left and Right Sidebars

Drag the divider between a sidebar and the center viewer to adjust the
sidebar width.

The sidebar widths can be persisted between sessions.

If one sidebar is hidden, the remaining sidebar can still be resized.

When a sidebar is hidden, its grid space is removed so the output viewer
can expand into the available area.

### Prompt Fields

The positive and negative prompt fields can be resized vertically.

Their sizes can be remembered between sessions.

### Run Log

The Run Log can be resized vertically.

Grab the **RUN LOG** header and drag:

``` text
Drag upward   → make the log taller
Drag downward → make the log shorter
```

The selected log height is stored locally and restored when RebelUI is
reopened.

------------------------------------------------------------------------

## Live VRAM Monitor

The header contains a live CUDA-memory monitor.

It reports actual GPU-memory information rather than estimating VRAM
from model file sizes.

Information can include:

-   Used GPU memory
-   Free GPU memory
-   Total GPU memory
-   PyTorch allocated memory
-   PyTorch reserved memory

Example:

``` text
LIVE GPU MEMORY
4.83 GB used / 8.00 GB
3.17 GB free
```

The displayed total uses GiB-style binary units so an 8 GB RTX 3070 is
represented consistently as approximately:

``` text
8.00 GB
```

rather than mixing decimal and binary units.

------------------------------------------------------------------------

## Dynamic VRAM

RebelUI uses ComfyUI's model-management system for supported execution
paths.

At startup, the Run Log reports detected memory behavior.

Example:

``` text
memory: dynamic VRAM enabled · pinned memory disabled
```

RebelUI is designed to coexist with ComfyUI's dynamic model loading and
offloading rather than reserving all model weights permanently on the
GPU.

------------------------------------------------------------------------

## Persistent Session Memory

RebelUI stores interface and generation preferences in browser local
storage.

Depending on enabled settings, RebelUI can remember:

-   Diffusion model
-   Diffusion format filter
-   Text encoder
-   Encoder format filter
-   VAE
-   Architecture preset
-   CLIP type
-   Positive prompt
-   Negative prompt
-   Sampler
-   Scheduler
-   Steps
-   CFG
-   Seed
-   Width
-   Height
-   Frames
-   FPS
-   Batch size
-   Theme
-   Interface density
-   Sidebar widths
-   Sidebar visibility
-   Active left/right tabs
-   Prompt field sizes
-   Run Log height

When a previous generation session is restored, the log may display:

``` text
restored last RebelUI session
```

This memory is local to the browser profile being used to access
RebelUI.

------------------------------------------------------------------------

## Themes and Settings

Open the **Settings** menu from the header.

Available appearance options can include:

-   Ember Dark
-   Slate Blue
-   Matrix Green
-   Light Studio

Density options include:

-   Comfortable
-   Compact

Settings can also control behavior such as:

-   Remember last generation settings
-   Remember sidebar sizes and tab state
-   Show/hide Run Log
-   Reset layout
-   Reset saved session
-   Restore sidebars
-   Reset prompt sizes

------------------------------------------------------------------------

## Outputs

When generation succeeds, RebelUI saves the output to its configured
output directory.

The backend then sends the resulting output path to the browser.

A successful Run Log sequence can look similar to:

``` text
sampling complete
loading vae
decoding VAE
VAE decode finished
saved /out/rebel-20260831-021313.png
displayed /out/rebel-20260831-021313.png
```

The distinction between **saved** and **displayed** is useful for
troubleshooting.

### Saved but not displayed

If the PNG exists in the output directory but does not appear in
RebelUI, check the Run Log for a browser display error.

The frontend cache-busts generated output URLs so a newly generated
image should not be hidden by an old browser cache entry.

------------------------------------------------------------------------

## GGUF Support

GGUF support requires ComfyUI-GGUF or another compatible GGUF loader
expected by the installed ComfyUI environment.

RebelUI can discover `.gguf` files from registered model directories
such as:

``` text
ComfyUI\models\diffusion_models\
ComfyUI\models\text_encoders\
```

The model and text-encoder selectors have independent:

``` text
ALL
FULL
GGUF
```

filters.

This allows combinations such as:

-   FULL diffusion model + FULL encoder
-   FULL diffusion model + GGUF encoder
-   GGUF diffusion model + FULL encoder
-   GGUF diffusion model + GGUF encoder

Actual compatibility still depends on the model architecture and the
installed ComfyUI / custom-node environment.

The dedicated native Krea 2 path currently targets supported
FULL/INT8/FP8 weights. GGUF Krea configurations may use a different
loader path.

------------------------------------------------------------------------

## Architecture Presets

RebelUI can expose architecture / conditioning presets based on the
installed ComfyUI environment.

Examples may include:

-   Auto
-   Krea 2
-   Flux
-   Flux 2
-   Qwen Image
-   Qwen Image Edit
-   SD3
-   SDXL
-   SD 1.5
-   WAN
-   LTX Video
-   LTX 2
-   Hunyuan Video
-   Hunyuan Image
-   HiDream
-   Cosmos
-   Mochi
-   Lumina 2
-   PixArt

The architecture selector is primarily a RebelUI workflow/conditioning
preset.

**RebelUI does not blindly force a diffusion architecture onto arbitrary
weights.**

The actual model architecture is still detected by ComfyUI's loader.

Available architectures and CLIP types can vary with the installed
ComfyUI version.

------------------------------------------------------------------------

## Run Log

The Run Log is intended to make the direct pipeline visible without
requiring a node graph.

A typical generation can report stages such as:

``` text
RebelUI ready.
device: NVIDIA GeForce RTX 3070 · 8.00 GB vram
memory: dynamic VRAM enabled · pinned memory disabled
found 12 diffusion · 34 encoders · 30 vae
restored last RebelUI session

generate · krea2 · 8 steps · cfg 1 · seed 42
Krea 2 execution mode · native ComfyUI nodes
loading diffusion model
loading text encoder
encoding prompt
native latent
sampling
sampling complete
loading vae
decoding VAE
VAE decode finished
saved /out/...
displayed /out/...
```

When reporting a bug, include the complete Run Log and any console
traceback.

------------------------------------------------------------------------

## Troubleshooting

### RebelUI says the backend is not reachable

Make sure `server.py` is still running.

Verify that the server console did not close because of a Python
exception.

Then reload:

``` text
http://127.0.0.1:8199
```

### Browser opens before RebelUI is ready

Increase the launcher delay:

``` bat
timeout /t 20 /nobreak >nul
```

### Models are missing

1.  Verify the file is in a ComfyUI model directory.
2.  Verify normal ComfyUI can see/load the model.
3.  Restart RebelUI after adding the file.
4.  Check the `ALL`, `FULL`, and `GGUF` filters.

### VAE is not visible

Use the independent scrollbar on the left Model Library panel to scroll
to the VAE section.

Also verify the VAE is stored in:

``` text
ComfyUI\models\vae\
```

or another VAE directory registered with ComfyUI.

### Krea 2 does not appear as a CLIP type

Update ComfyUI to a version containing Krea 2 support.

RebelUI derives supported behavior from the installed ComfyUI
environment.

### Krea 2 model loads but generation fails

Verify the complete stack.

Example:

``` text
Diffusion:
krea2_turbo_int8_convrot.safetensors

Text encoder:
qwen3vl_4b_fp8_scaled.safetensors

VAE:
qwen_image_vae.safetensors

CLIP type:
krea2
```

Also confirm that the same model files work in normal ComfyUI.

### Generation completes but the image does not display

Check whether the file exists in the output folder.

Then inspect the Run Log.

Expected:

``` text
saved /out/filename.png
displayed /out/filename.png
```

If the file is saved but the browser cannot load it, RebelUI should
report a display error separately.

### Output image is too large in the viewer

Current builds constrain generated media to the center viewer with:

-   proportional scaling
-   maximum width/height containment
-   centered positioning

If an older build allows an image to overlap the controls, update
`static/index.html`.

### VRAM number looks different from another monitoring application

Different programs may display decimal GB or binary GiB.

RebelUI uses binary GPU-memory calculations internally for consistency
with CUDA memory reporting.

### GGUF model is missing

Make sure:

1.  ComfyUI-GGUF is installed.
2.  The `.gguf` file is inside a registered model directory.
3.  RebelUI has been restarted.
4.  The appropriate selector is set to `ALL` or `GGUF`.

------------------------------------------------------------------------

## Known Limitations

RebelUI is intentionally not a complete replacement for ComfyUI's graph
system.

Complex workflows may still require normal ComfyUI.

Examples include:

-   ControlNet
-   IPAdapter
-   LoRA stacking and advanced LoRA routing
-   Multi-model conditioning
-   Complex image-to-image workflows
-   Inpainting / masking pipelines
-   Reference-image workflows
-   Multi-stage upscaling
-   Advanced video pipelines
-   Audio pipelines
-   Custom-node-specific graphs
-   Specialized conditioning nodes
-   Workflows requiring arbitrary graph topology

Model-family support can also vary depending on your ComfyUI version and
installed custom nodes.

------------------------------------------------------------------------

## Development Philosophy

RebelUI is built around a simple rule:

> **RebelUI should simplify ComfyUI, not reimplement ComfyUI.**

Where possible, the backend should use ComfyUI's native:

-   model loaders
-   text encoders
-   samplers
-   schedulers
-   latent implementations
-   VAE loaders
-   VAE decoders
-   CUDA management
-   dynamic VRAM system

The frontend should focus on usability:

-   fast model selection
-   clear prompting
-   predictable controls
-   readable logs
-   live hardware information
-   persistent settings
-   clean output presentation

------------------------------------------------------------------------

## Reporting Bugs

When opening an issue, include as much of the following as possible:

``` text
Operating system:
GPU:
VRAM:
NVIDIA driver:
ComfyUI version / commit:
PyTorch version:
Python version:

Architecture preset:
Diffusion model:
Text encoder:
VAE:
CLIP type:
Sampler:
Scheduler:
Steps:
CFG:
Width:
Height:
Batch:

RebelUI Run Log:
[Paste log here]

Server console traceback:
[Paste traceback here]
```

Also mention whether the same model stack works in normal ComfyUI.

That comparison is especially useful because RebelUI depends directly on
the installed ComfyUI runtime.

------------------------------------------------------------------------

## Contributing

Issues, compatibility reports, UI feedback, documentation improvements,
and code contributions are welcome.

Before submitting a code change:

1.  Verify normal ComfyUI works with the affected model/workflow.
2.  Test RebelUI with a clean browser refresh.
3.  Check both restored-session and fresh-session behavior.
4.  Test hiding/showing both sidebars.
5.  Test sidebar resizing.
6.  Test prompt and Run Log resizing.
7.  Verify generated output is both saved and displayed.
8.  Include relevant console/log output with the pull request.

------------------------------------------------------------------------

## Security / Network Access

RebelUI is primarily intended as a local interface.

The default URL is:

``` text
http://127.0.0.1:8199
```

Do not expose a development server directly to the public internet
without understanding the security implications and adding appropriate
authentication, reverse-proxy configuration, and network protections.

------------------------------------------------------------------------

## Updating

RebelUI is currently a small project, so updating generally consists of
replacing the application files with the newer release.

Before updating, you may want to back up any files you have manually
modified.

Your browser-side session preferences are stored separately in local
storage and may survive application-file updates unless the storage
schema changes.

------------------------------------------------------------------------

## FAQ

### Does RebelUI replace ComfyUI?

No.

RebelUI is an alternate direct-generation frontend that uses ComfyUI as
its engine.

### Do I need to copy my models?

No. RebelUI reads the model directories registered with your existing
ComfyUI installation.

### Is INT8 considered GGUF?

No.

An INT8 `.safetensors` model is categorized as FULL.

### Is FP8 considered GGUF?

No.

An FP8 `.safetensors` model is categorized as FULL.

### Can RebelUI load GGUF?

Yes, when the required GGUF support is installed in the ComfyUI
environment.

### Does RebelUI remember my models?

Yes. Session memory can restore the previous model, encoder, VAE,
prompts, sampler settings, dimensions, and UI state.

### Why does RebelUI need to run from the ComfyUI directory?

The current direct-library architecture imports ComfyUI's Python modules
directly. Running with the ComfyUI Python environment and working
directory ensures those imports and model paths resolve correctly.

### Where are generated images stored?

They are written to RebelUI's configured output location and served back
to the browser through the local RebelUI server.

### Can I still use normal ComfyUI?

Yes. RebelUI does not replace your ComfyUI installation.

------------------------------------------------------------------------

## Roadmap

Potential future improvements include:

-   Saved generation presets
-   Import/exportable RebelUI profiles
-   Searchable model selectors
-   Favorite models and encoders
-   Generation history
-   Queue management
-   Additional native model-family execution paths
-   LoRA support
-   Image-to-image
-   Inpainting
-   Reference-image workflows
-   Improved video support
-   Better model metadata
-   Additional diagnostics
-   Packaging / installer improvements
-   Broader cross-platform testing

The roadmap is not a guarantee of implementation order or release dates.

------------------------------------------------------------------------

## Disclaimer

RebelUI is an independent project built to work with ComfyUI.

It is not affiliated with, sponsored by, or endorsed by the ComfyUI
project or by model creators whose models may be used through the
application.

Users are responsible for complying with the licenses and terms
associated with ComfyUI, custom nodes, model weights, and other
third-party software they install.

------------------------------------------------------------------------

## License

**Choose and add a license before publishing the project for public
reuse.**

Common open-source options include:

-   MIT
-   Apache License 2.0
-   GPL-3.0

Until a license is explicitly included in the repository, do not assume
that public availability automatically grants permission to copy,
modify, or redistribute the code.

Once you choose a license, add a `LICENSE` file to the repository and
update this section.

Example:

``` text
Licensed under the MIT License. See LICENSE for details.
```

------------------------------------------------------------------------

## Credits

RebelUI is built around the ComfyUI ecosystem and uses technologies
including:

-   ComfyUI
-   PyTorch
-   CUDA
-   aiohttp
-   Pillow
-   ComfyUI-GGUF where applicable

Special thanks to the developers and contributors building the
open-source ComfyUI ecosystem.

------------------------------------------------------------------------

## RebelUI

**ComfyUI underneath. A focused generation interface on top.**

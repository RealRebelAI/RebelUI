# RebelUI

A direct-to-library runner for ComfyUI's core. No graph engine, no node
system, no frontend bundle — it imports `comfy` the same way ComfyUI's own
nodes do and calls it directly.

Built for people running quantized models on 8-12 GB cards, where the question
is never "what shape should this graph be" but "does this fit, and how fast."

## What it is

ComfyUI's `comfy/` directory is a standalone library: model detection, the ldm
implementations, samplers, the memory manager, quant ops. The graph engine and
web UI sit on top of it. RebelUI skips those and calls the library.

- **The VRAM gauge is the point.** Model + encoder + VAE + working memory,
  segmented against your card's ceiling, updating as you change files. On an
  8 GB card that number decides everything.
- GGUF loads route through ComfyUI-GGUF, so quantized weights dequant on the
  fly exactly as they do in ComfyUI.
- Components stay loaded between runs — change the seed, hit generate, no
  reload.
- Streams progress over NDJSON: per-step timing, log lines, the finished file.

## Requirements

- A working ComfyUI portable install (RebelUI borrows its library and models)
- `aiohttp`, `pillow` in the embedded python
- ComfyUI-GGUF for `.gguf` models
- `ffmpeg` on PATH for video output

```
<COMFY>\python_embeded\python.exe -m pip install aiohttp pillow --no-deps
```

## Run

```
cd /d <COMFY>\ComfyUI
..\python_embeded\python.exe <path>\rebelui\server.py
```

Then open **http://127.0.0.1:8199**

`--comfy <path>` if you'd rather not run from the ComfyUI directory,
`--port`, `--listen 0.0.0.0` to reach it from another machine,
`--out <dir>` for where renders land.

## Status

The generation path calls `comfy.sample.sample` directly, which covers plain
text-to-image and text-to-video. Models needing extra conditioning wiring
(reference latents, audio VAEs, image encoders) need that step added to
`Runner.run` — the hooks are there, the plumbing per model family isn't.

# RebelUI Auto-Quantizer

The Quantize tab performs anonymous HuggingFace metadata discovery, safetensors
header range probes, architecture fingerprinting, recipe selection, disk/RAM
planning, explicit confirmation, and NDJSON-streamed conversion progress.

## Security / privacy

- No HuggingFace token UI exists.
- HuggingFace Hub calls explicitly use `token=False`; cached credentials are not used.
- Gated/private repositories are refused.
- No RebelUI API, hosted service, telemetry, or analytics are used.

## Memory rule

The quantizer never calls `safetensors.torch.load_file()`. Remote probes read
only safetensors headers. W4A8/INT8 conversion delegates to the existing
row-chunked `D:\minimax\w4a8_convert.py` and downloads one source shard at a
time.

GGUF generation delegates to `st_to_gguf.py`, `llama-quantize.exe`,
`gguf_swap_hiprec.py`, `gguf_set_config.py`, and `gguf_fix_shapes.py`.

### Sharded remote GGUF safety guard

The current `st_to_gguf.py` accepts a single safetensors source. RebelUI will
therefore **refuse a GGUF ladder from a sharded remote repository** rather than
silently downloading/merging the entire checkpoint and violating the
one-shard peak-disk principle. W4A8 and INT8 remain available shard-by-shard.
A local single-file safetensors source can build the GGUF ladder.

A future streaming multi-shard GGUF adapter should remove this guard without
materializing a merged source checkpoint.

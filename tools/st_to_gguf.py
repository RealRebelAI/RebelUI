r"""Stream-convert MiniMax-H3 bf16 safetensors -> F16 GGUF (low RAM).

Usage:
  python_embeded\python.exe D:\minimax\mmh3_convert.py
  python_embeded\python.exe D:\minimax\mmh3_convert.py <src.safetensors> <dst.gguf>

Never loads the whole model: reads each tensor by byte offset, converts,
writes, frees. Peak RAM ~2 GB regardless of model size.

Type policy (mirrors city96 convert.py):
  - 1-D tensors            -> F32
  - <= 1024 elements       -> F32
  - conditioning stack     -> F32   (HIPREC below)
  - everything else        -> F16
"""
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(r"D:\minimax\minimax_h3_fl2va_bf16.safetensors")
DST = Path(r"D:\minimax\minimax_h3_fl2va-F16.gguf")
ARCH = "wan"   # override with --arch (qwen3vl for text encoders)

# tensors that must never be quantized (native loaders build these outside
# the GGUF dequant path -- the WanDancer music-tensor lesson)
HIPREC = (
    "audio_patch_proj",
    "time_embedder",
    "condition_proj",
    "final_layer",
    "rope.",
)

QUANT_THRESHOLD = 1024
ALIGNMENT = 32
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# ggml types
T_F32, T_F16 = 0, 1
# gguf value types
V_UINT32, V_STRING, V_ARRAY, V_INT32 = 4, 8, 9, 5

DT_SIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
           "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
           "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hlen).decode("utf-8"))
    meta.pop("__metadata__", None)
    return meta, 8 + hlen


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def out_type(name, shape, dtype):
    if dtype not in ("BF16", "F16", "F32"):
        return T_F32
    if len(shape) <= 1:
        return T_F32
    if numel(shape) <= QUANT_THRESHOLD:
        return T_F32
    if any(h in name for h in HIPREC):
        return T_F32
    return T_F16


def convert_bytes(raw, dtype, out_t):
    """raw bytes -> numpy array of the target ggml type."""
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        f32 = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unsupported source dtype {dtype}")
    return f32 if out_t == T_F32 else f32.astype(np.float16)


def w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def pad_to(f, alignment):
    pos = f.tell()
    rem = pos % alignment
    if rem:
        f.write(b"\0" * (alignment - rem))


def parse_cli():
    """positional src dst, plus --arch X --hiprec a,b,c --no-reshape"""
    args = [a for a in sys.argv[1:]]
    arch, hiprec, reshape = ARCH, list(HIPREC), True
    pos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--arch" and i + 1 < len(args):
            arch = args[i + 1]
            i += 2
        elif a == "--hiprec" and i + 1 < len(args):
            hiprec = [x for x in args[i + 1].split(",") if x]
            i += 2
        elif a == "--no-hiprec":
            hiprec = []
            i += 1
        elif a == "--no-reshape":
            reshape = False
            i += 1
        else:
            pos.append(a)
            i += 1
    return pos, arch, hiprec, reshape


def main():
    pos, arch_name, hiprec, do_reshape = parse_cli()
    global HIPREC, ARCH
    HIPREC, ARCH = tuple(hiprec), arch_name
    src = Path(pos[0]) if len(pos) > 0 else SRC
    dst = Path(pos[1]) if len(pos) > 1 else DST
    print(f"[convert] arch='{ARCH}'  reshape={do_reshape}")
    if not src.exists():
        print(f"[convert] source not found: {src}")
        sys.exit(1)
    if dst.exists():
        print(f"[convert] destination already exists: {dst}")
        print("[convert] delete it or pass a different name. Aborting.")
        sys.exit(1)

    meta, data_start = read_header(src)
    names = list(meta.keys())
    print(f"[convert] {src.name}: {len(names)} tensors, "
          f"{human(src.stat().st_size)}")

    # ---- plan every tensor first (no data read) ----
    plan = []
    n_f32 = n_f16 = 0
    for name in names:
        v = meta[name]
        shape = list(v["shape"])
        ot = out_type(name, shape, v["dtype"])
        if len(name.encode("utf-8")) > 127:
            print(f"[convert] FATAL name too long for ggml: {name}")
            sys.exit(1)
        # ggml supports at most 4 dims. Merge the middle dims of >4-D
        # tensors (row-major, so no data movement) and record orig_shape.
        resh = None
        # ggml drops trailing singleton dims when it rewrites a file
        # (llama-quantize does this), which loses e.g. [1, 4096] -> [4096].
        # Record orig_shape so the loader restores it.
        ggml_dims = list(reversed(shape))
        while len(ggml_dims) > 1 and ggml_dims[-1] == 1:
            ggml_dims.pop()
        if len(ggml_dims) != len(shape):
            resh = shape
        if len(shape) > 4:
            mid = 1
            for d in shape[1:-2]:
                mid *= d
            resh = [shape[0], mid, shape[-2], shape[-1]]
            print(f"[convert] {len(shape)}-D flattened for ggml: {name} "
                  f"{shape} -> {resh}")
        nb = numel(shape) * (4 if ot == T_F32 else 2)
        # K-quants need row length divisible by 256. If the row isn't but the
        # total is, reshape to (n/256, 256) and record the original shape so
        # the loader can restore it. Row-major data is unchanged on disk.
        if (do_reshape and resh is None and ot == T_F16 and len(shape) == 2
                and numel(shape) % 256 == 0 and shape[-1] % 256 != 0):
            resh = [numel(shape) // 256, 256]
        plan.append({"name": name, "shape": shape, "resh": resh,
                     "src_dtype": v["dtype"],
                     "src_off": v["data_offsets"], "out_t": ot, "nbytes": nb})
        if ot == T_F32:
            n_f32 += 1
        else:
            n_f16 += 1
    reshaped = [p for p in plan if p["resh"]]
    total_out = sum(p["nbytes"] for p in plan)
    print(f"[convert] output plan: {n_f16} F16, {n_f32} F32, "
          f"~{human(total_out)} of tensor data")
    print(f"[convert] hiprec/F32 kept for: "
          f"{', '.join(HIPREC) if HIPREC else '(none)'}")
    print(f"[convert] reshaped for K-quant compatibility: {len(reshaped)} "
          f"tensors")
    for p in reshaped[:3]:
        print(f"           {p['name']}  {p['shape']} -> {p['resh']}")
    nd = [p for p in plan if len(p["shape"]) > 4]
    if nd:
        print(f"[convert] >4-D tensors flattened: {len(nd)} "
              f"(these stay F16 at quantize time, which is correct for "
              f"conv patch embeddings)")
    bad = [p["name"] for p in plan
           if p["out_t"] == T_F16 and len(p["resh"] or p["shape"]) == 2
           and (p["resh"] or p["shape"])[-1] % 256 != 0]
    print(f"[convert] F16 tensors still not 256-divisible (would fall back "
          f"to F16 at quantize time): {len(bad)}")
    for n in bad[:5]:
        print(f"           {n}")

    kv = [("general.architecture", V_STRING, ARCH),
          ("general.quantization_version", V_UINT32, 2),
          ("general.alignment", V_UINT32, ALIGNMENT),
          ("general.file_type", V_UINT32, 1)]
    for p in reshaped:
        kv.append((f"comfy.gguf.orig_shape.{p['name']}", V_ARRAY,
                   [int(d) for d in p["shape"]]))

    with open(dst, "wb") as f:
        # ---- header ----
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION,
                            len(plan), len(kv)))
        for key, vt, val in kv:
            w_str(f, key)
            f.write(struct.pack("<I", vt))
            if vt == V_STRING:
                w_str(f, val)
            elif vt == V_ARRAY:
                f.write(struct.pack("<I", V_INT32))
                f.write(struct.pack("<Q", len(val)))
                for d in val:
                    f.write(struct.pack("<i", d))
            else:
                f.write(struct.pack("<I", val))

        # ---- tensor info block (offsets relative to data section) ----
        off = 0
        offsets = []
        for p in plan:
            w_str(f, p["name"])
            dims = list(reversed(p["resh"] or p["shape"]))   # ggml order
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", p["out_t"]))
            f.write(struct.pack("<Q", off))
            offsets.append(off)
            step = p["nbytes"]
            rem = step % ALIGNMENT
            off += step + ((ALIGNMENT - rem) if rem else 0)

        pad_to(f, ALIGNMENT)
        data_base = f.tell()
        print(f"[convert] data section starts at {data_base}")

        # ---- stream tensor data ----
        t0 = time.time()
        with open(src, "rb") as sf:
            for i, p in enumerate(plan, 1):
                a, b = p["src_off"]
                sf.seek(data_start + a)
                raw = sf.read(b - a)
                arr = convert_bytes(raw, p["src_dtype"], p["out_t"])
                del raw
                exp = numel(p["shape"])
                if arr.size != exp:
                    print(f"[convert] FATAL element mismatch {p['name']}: "
                          f"{arr.size} vs {exp}")
                    sys.exit(1)
                f.seek(data_base + offsets[i - 1])
                f.write(arr.tobytes())
                del arr
                if i % 25 == 0 or i == len(plan):
                    el = time.time() - t0
                    print(f"[convert] {i}/{len(plan)}  {p['name'][:48]:<48} "
                          f"{el:6.1f}s", flush=True)
        # pad final tensor
        f.seek(data_base + off)
        f.truncate()

    print(f"[convert] wrote {dst.name}  ({human(dst.stat().st_size)})")

    # ---- verify by reading it back ----
    try:
        from gguf import GGUFReader
        from gguf.constants import GGMLQuantizationType as Q
        r = GGUFReader(str(dst))
        counts = {}
        for t in r.tensors:
            counts[Q(t.tensor_type).name] = counts.get(
                Q(t.tensor_type).name, 0) + 1
        print(f"[convert] VERIFY: {len(r.tensors)} tensors readable, "
              f"qtypes {counts}")
        if len(r.tensors) != len(plan):
            print("[convert] WARNING tensor count mismatch!")
    except Exception as e:
        print(f"[convert] readback check skipped/failed: {e}")


if __name__ == "__main__":
    main()

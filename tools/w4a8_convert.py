r"""W4A8 / W3A8 / W2A8 codebook quantizer for diffusion transformers.

Reimplements Kijai's AsymW4A8Int8Layout quantization (comfy-kitchen PR #90)
as a streaming converter, plus 3-bit and 2-bit variants.

STREAMING: output offsets are computed before anything is quantized, the
header is written first, then each tensor is produced and written to its own
offset. Large tensors are processed in row chunks. Peak RAM is one chunk,
not one model.

Usage:
  python_embeded\python.exe w4a8_convert.py <src.safetensors> <out.safetensors>
      [--bits 4|3|2] [--group-size 16] [--convrot 256] [--exclude "regex"]
      [--chunk-rows 4096] [--verify N] [--min-numel 4096] [--measure]

Per quantized weight:
  <n>.qdata      packed codes       uint8   4b: K/2, 3b: 3K/8, 2b: K/4 per row
  <n>.scale      per-group scale    F8_E4M3 [N, K/group_size]
  <n>.s_channel  per-channel scale  F32     [N]
  <n>.codebook   Lloyd-Max levels   F32     [2**bits]
  <n>.w4a8       json config        uint8   blob

bits/elem = bits + 8/group_size
  4b gs16 = 4.50   3b gs16 = 3.50   2b gs16 = 2.50
  4b gs32 = 4.25   3b gs32 = 3.25   2b gs32 = 2.25
"""
import json
import math
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

HAD = {}
HAD4 = {}


def hadamard_ck(size):
    """comfy-kitchen _build_hadamard: power-of-4 Kronecker H4, /sqrt(size)."""
    if size in HAD4:
        return HAD4[size]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"W4A4 Hadamard size must be a power of 4: {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                       [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h, cur = h4, 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    HAD4[size] = h / (size ** 0.5)
    return HAD4[size]


def best_cr4(k, cap=256):
    """Largest power-of-FOUR <= cap dividing k, also requiring k %% 64 == 0."""
    g = cap
    while g >= 4:
        if math.log(g, 4) % 1 == 0 and k % g == 0:
            return g
        g //= 2
    return 0


def quant_rows_w4a4(rows, cr):
    """ConvRot rotate (H.T) + signed int4 row-wise + nibble pack.
    Mirrors comfy_kitchen quantize_convrot_w4a4_weight exactly."""
    n, k = rows.shape
    h = hadamard_ck(cr)
    rot = torch.matmul(rows.float().reshape(n, k // cr, cr), h.T).reshape(n, k)
    absmax = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / 7.0
    q = (rot / scales).round().clamp(-7, 7).to(torch.int8)
    lo = q[:, 0::2].to(torch.int32) & 0x0F
    hi = q[:, 1::2].to(torch.int32) & 0x0F
    packed = (lo | (hi << 4)).to(torch.int8)
    return packed, scales.reshape(n).float()


def dequant_rows_w4a4(packed, scales, cr):
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1).float()
    w = q * scales.reshape(-1, 1).float()
    n, k = w.shape
    h = hadamard_ck(cr)
    return torch.matmul(w.reshape(n, k // cr, cr), h).reshape(n, k)


def hadamard(size):
    if size in HAD:
        return HAD[size]
    if size < 2 or (size & (size - 1)):
        raise ValueError(f"hadamard size must be power of two: {size}")
    h = torch.ones(1, 1)
    while h.shape[0] < size:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / math.sqrt(size)
    HAD[size] = h
    return h


def best_cr(k, cap):
    g = min(cap, 1 << (k.bit_length() - 1))
    while g >= 16:
        if k % g == 0:
            return g
        g //= 2
    return 0


def rotate(w, g):
    """comfy-kitchen _rotate_weight: W_rot = W @ H_block^T with the REGULAR
    (power-of-4 Kronecker) Hadamard. Must match exactly."""
    n, k = w.shape
    h = hadamard_ck(g)
    return (w.view(n, k // g, g) @ h.T).view(n, k)


def unrotate(w, g):
    n, k = w.shape
    h = hadamard_ck(g)
    return (w.view(n, k // g, g) @ h).view(n, k)


def fit_codebook(samples, levels, iters=25):
    s = samples.float().flatten()
    cb = torch.quantile(s, torch.linspace(0, 1, levels))
    for _ in range(iters):
        a = (s.unsqueeze(-1) - cb).abs().argmin(-1)
        new = cb.clone()
        for i in range(levels):
            m = a == i
            if m.any():
                new[i] = s[m].mean()
        cb = new
    return cb.contiguous()


def assign(x, cb):
    """Exact nearest-codebook index via midpoint bucketing.

    Indices are produced by torch.bucketize against the midpoints of the
    sorted levels, so they are always in [0, levels-1] by construction.
    (The previous loop-and-torch.where version could emit out-of-range
    indices through dtype promotion.)"""
    cbs, order = torch.sort(cb.float())
    bounds = ((cbs[1:] + cbs[:-1]) * 0.5).contiguous()
    xi = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    idx = torch.bucketize(xi.contiguous(), bounds)
    out = order.to(idx.device)[idx].to(torch.int64)
    if int(out.min()) < 0 or int(out.max()) >= cb.numel():
        raise RuntimeError(
            f"internal: codebook index out of range "
            f"[{int(out.min())}, {int(out.max())}] for {cb.numel()} levels")
    return out


def packed_row_bytes(k, bits):
    if bits == 4:
        return k // 2
    if bits == 3:
        return (k // 8) * 3
    if bits == 2:
        return k // 4
    raise ValueError("bits must be 2, 3 or 4")


def pack(q, bits):
    n, k = q.shape
    u = q.to(torch.int32)
    if bits == 4:
        return ((u[:, 0::2] & 0xF) | ((u[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    if bits == 2:
        return ((u[:, 0::4] & 3) | ((u[:, 1::4] & 3) << 2)
                | ((u[:, 2::4] & 3) << 4) | ((u[:, 3::4] & 3) << 6)
                ).to(torch.uint8)
    v = u.view(n, k // 8, 8) & 7
    acc = torch.zeros(n, k // 8, dtype=torch.int32)
    for i in range(8):
        acc |= v[:, :, i] << (3 * i)
    return torch.stack([acc & 0xFF, (acc >> 8) & 0xFF,
                        (acc >> 16) & 0xFF], -1).reshape(
        n, (k // 8) * 3).to(torch.uint8)


def unpack(p, k, bits):
    n = p.shape[0]
    u = p.to(torch.int32)
    if bits == 4:
        out = torch.empty(n, k, dtype=torch.int32)
        out[:, 0::2] = u & 0xF
        out[:, 1::2] = (u >> 4) & 0xF
        return out
    if bits == 2:
        out = torch.empty(n, k, dtype=torch.int32)
        out[:, 0::4] = u & 3
        out[:, 1::4] = (u >> 2) & 3
        out[:, 2::4] = (u >> 4) & 3
        out[:, 3::4] = (u >> 6) & 3
        return out
    g = u.view(n, k // 8, 3)
    acc = g[:, :, 0] | (g[:, :, 1] << 8) | (g[:, :, 2] << 16)
    return torch.stack([(acc >> (3 * i)) & 7 for i in range(8)],
                       -1).reshape(n, k)


def quant_rows(rows, cb, group_size, cr, levels, rounds=3):
    """Quantize a row chunk. Returns packed codes, s_rel(fp8), s_channel."""
    n, k = rows.shape
    rot = rotate(rows.float(), cr)
    groups = k // group_size
    gw = rot.view(n, groups, group_size)
    gs = gw.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    q = assign(gw / gs, cb)
    for _ in range(rounds):
        qv = cb[q]
        gs = ((gw * qv).sum(-1, keepdim=True)
              / (qv * qv).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        q = assign(gw / gs, cb)
    shifted = cb[q] * gs
    s_ch = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (gs.squeeze(-1) / s_ch.unsqueeze(1)).float()
    s_rel = s_rel.to(torch.float8_e4m3fn)
    grid = (cb.view(1, 1, -1) * s_rel.float().unsqueeze(-1)).round().clamp(-127, 127)
    tgt = (gw / s_ch.view(n, 1, 1))
    d = (tgt.unsqueeze(-1) - grid.unsqueeze(2)).abs()
    q = d.argmin(-1).to(torch.int32).view(n, k)
    return pack(q, int(math.log2(cb.numel()))), s_rel, s_ch.float()


def dequant_rows(packed, s_rel, s_ch, cb, k, bits, group_size, cr):
    q = unpack(packed, k, bits)
    n = q.shape[0]
    groups = k // group_size
    grid = (cb.view(1, 1, -1) * s_rel.float().unsqueeze(-1)).round().clamp(-127, 127)
    picked = torch.gather(grid, 2, q.view(n, groups, group_size).long())
    rot = (picked * s_ch.view(n, 1, 1)).view(n, k)
    return unrotate(rot, cr)


# ---------------------------------------------------------------- safetensors
DT_NP = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
         "I64": np.int64, "I32": np.int32, "I16": np.int16,
         "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
DT_SZ = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n).decode("utf-8"))
    meta.pop("__metadata__", None)
    return meta, 8 + n


def read_rows(path, entry, data_start, r0, r1):
    """Read rows [r0, r1) of a 2-D tensor as float32."""
    shape = entry["shape"]
    k = shape[1]
    esz = DT_SZ[entry["dtype"]]
    base = data_start + entry["data_offsets"][0] + r0 * k * esz
    nbytes = (r1 - r0) * k * esz
    with open(path, "rb") as f:
        f.seek(base)
        raw = f.read(nbytes)
    dt = entry["dtype"]
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=DT_NP[dt]).astype(np.float32)
    return torch.from_numpy(arr.reshape(r1 - r0, k).copy())


def copy_bytes(src, entry, data_start, out, chunk=32 << 20):
    a, b = entry["data_offsets"]
    remaining = b - a
    with open(src, "rb") as f:
        f.seek(data_start + a)
        while remaining > 0:
            buf = f.read(min(chunk, remaining))
            out.write(buf)
            remaining -= len(buf)


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print(__doc__)
        sys.exit(1)
    src = Path(args[0])
    dst = Path(args[1]) if len(args) > 1 and not args[1].startswith("-") else None
    bits, gsz, cr_cap = 4, 16, 256
    act_bits = 8
    fmt = "asym_w4a8_int8"
    fallback = "int8"
    model_type = None
    copy_meta_from = None
    only_prefix = None
    int8_keys = []
    exclude, chunk_rows, verify_n, min_numel = None, 4096, 8, 4096
    measure = dst is None
    i = 2 if dst else 1
    while i < len(args):
        a = args[i]
        if a == "--bits":
            bits = int(args[i + 1]); i += 2
        elif a == "--format":
            fmt = args[i + 1]; i += 2
        elif a == "--only-prefix":
            only_prefix = args[i + 1]; i += 2
        elif a == "--int8-keys":
            int8_keys = [k for k in args[i + 1].split(",") if k]; i += 2
        elif a == "--model-type":
            model_type = args[i + 1]; i += 2
        elif a == "--metadata-from":
            copy_meta_from = args[i + 1]; i += 2
        elif a == "--fallback":
            fallback = args[i + 1]; i += 2
        elif a == "--act-bits":
            act_bits = int(args[i + 1]); i += 2
        elif a == "--group-size":
            gsz = int(args[i + 1]); i += 2
        elif a == "--convrot":
            cr_cap = int(args[i + 1]); i += 2
        elif a == "--exclude":
            exclude = re.compile(args[i + 1]); i += 2
        elif a == "--chunk-rows":
            chunk_rows = int(args[i + 1]); i += 2
        elif a == "--verify":
            verify_n = int(args[i + 1]); i += 2
        elif a == "--min-numel":
            min_numel = int(args[i + 1]); i += 2
        elif a == "--measure":
            measure = True; i += 1
        else:
            print(f"unknown option {a}"); sys.exit(1)
    levels = 1 << bits
    bpe = bits + 8.0 / gsz
    with open(src, "rb") as _f:
        _n = struct.unpack("<Q", _f.read(8))[0]
        src_meta = json.loads(_f.read(_n).decode("utf-8")).get("__metadata__")
    meta, data_start = read_header(src)
    print(f"[w{bits}a8] {src.name}: {len(meta)} tensors")
    print(f"[w{bits}a8] levels={levels} group={gsz} convrot<={cr_cap} "
          f"chunk_rows={chunk_rows}")
    print(f"[w{bits}a{act_bits}] storage {bpe:.2f} bits/elem ({bpe/8:.4f} B/elem)")

    # ---- plan ----
    plan = []
    q_el = p_el = 0
    if only_prefix:
        kept = {k: v for k, v in meta.items() if k.startswith(only_prefix)}
        print(f"[convert] --only-prefix '{only_prefix}': keeping "
              f"{len(kept)} of {len(meta)} tensors, stripping the prefix")
        if not kept:
            print("[convert] no tensors matched that prefix"); sys.exit(1)
        meta = {k[len(only_prefix):]: v for k, v in kept.items()}
    for name, e in meta.items():
        shape = e["shape"]
        numel = int(np.prod(shape)) if shape else 0
        ok = (len(shape) == 2 and numel >= min_numel
              and name.endswith("weight")
              and not (exclude and exclude.search(name)))
        kk = shape[1] if ok else 0
        mode, cr = None, 0
        if ok and int8_keys and any(t in name for t in int8_keys):
            # forced to int8_tensorwise: ~1% error, no convrot constraint
            mode, cr = "int8", 0
        elif ok:
            # fused w4a8 CUDA kernel accepts convrot group 256 ONLY
            if (kk % 256 == 0 and kk % gsz == 0 and kk % 16 == 0
                    and not (bits == 3 and kk % 8)
                    and not (bits == 2 and kk % 4)):
                mode, cr = "w4a8", 256
            elif fallback == "int8":
                mode, cr = "int8", 0
            elif bits == 4 and kk % 64 == 0 and best_cr4(kk):
                # fallback: convrot_w4a4 kernel accepts power-of-4 groups
                mode, cr = "w4a4", best_cr4(kk)
        if mode:
            plan.append({"name": name, "e": e, "quant": True, "cr": cr,
                         "mode": mode})
            q_el += numel
        else:
            plan.append({"name": name, "e": e, "quant": False})
            p_el += numel
            p_el += numel
    est = q_el * bpe / 8 + p_el * 2
    print(f"[w{bits}a8] quantize {sum(1 for p in plan if p['quant'])} tensors "
          f"({q_el/1e9:.2f} B params), copy {p_el/1e9:.2f} B params")
    print(f"[w{bits}a8] estimated output {est/1e9:.2f} GB")
    if measure:
        print(f"[w{bits}a8] measure-only; no file written")
        if not dst:
            return

    # ---- output header (offsets known before any quantization) ----
    entries, off = {}, 0

    def add(nm, dtype, shape):
        nonlocal off
        nb = int(np.prod(shape)) * DT_SZ[dtype] if shape else 0
        entries[nm] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [off, off + nb]}
        off += nb

    for p in plan:
        nm, e = p["name"], p["e"]
        if not p["quant"]:
            add(nm, e["dtype"], e["shape"])
            continue
        n, k = e["shape"]
        base = nm[:-len(".weight")] if nm.endswith(".weight") else nm
        p["cq_key"] = f"{base}.comfy_quant"
        md = p.get("mode")
        if md == "w4a4":
            p["cfg"] = json.dumps({"format": "convrot_w4a4",
                                   "convrot_groupsize": p["cr"],
                                   "quant_group_size": 64,
                                   "linear_dtype": "int4",
                                   "orig_shape": [n, k]}).encode()
            add(nm, "I8", [n, k // 2])
            add(f"{nm}_scale", "F32", [n])
        elif md == "int8":
            p["cfg"] = json.dumps({"format": "int8_tensorwise",
                                   "orig_shape": [n, k]}).encode()
            add(nm, "I8", [n, k])
            add(f"{nm}_scale", "F32", [1])
        else:
            p["cfg"] = json.dumps({"format": "asym_w4a8_int8",
                                   "bits": bits, "act_bits": act_bits,
                                   "group_size": gsz,
                                   "convrot_groupsize": p["cr"],
                                   "orig_shape": [n, k]}).encode()
            add(nm, "I8", [n, packed_row_bytes(k, bits)])
            add(f"{nm}_s_rel", "F8_E4M3", [n, k // gsz])
            add(f"{nm}_s_channel", "F32", [n])
            add(f"{nm}_codebook", "F32", [levels])
        add(p["cq_key"], "U8", [len(p["cfg"])])
    hdr = dict(entries)
    out_meta = {"format": "pt",
                "quantization": f"w{bits}a{act_bits}_int8_codebook"}
    if copy_meta_from:
        with open(copy_meta_from, "rb") as _f:
            _n2 = struct.unpack("<Q", _f.read(8))[0]
            ref_meta = json.loads(_f.read(_n2).decode("utf-8")).get(
                "__metadata__") or {}
        out_meta.update(ref_meta)
        print(f"[convert] copied metadata: {ref_meta}")
    elif src_meta:
        out_meta.update(src_meta)
        print(f"[convert] preserved source metadata: {src_meta}")
    if model_type:
        cfg_blob = {}
        if "config" in out_meta:
            try:
                cfg_blob = json.loads(out_meta["config"])
            except Exception:
                cfg_blob = {}
        cfg_blob.setdefault("transformer", {})["model_type"] = model_type
        out_meta["config"] = json.dumps(cfg_blob)
        print(f"[convert] set model_type={model_type}")
    hdr["__metadata__"] = out_meta
    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    base = 8 + len(hj)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.truncate(base + off)

    # ---- stream ----
    errs = []
    done = 0
    n_quant = sum(1 for x in plan if x["quant"])
    run_start = time.time()
    with open(dst, "r+b") as out:
        for p in plan:
            nm, e = p["name"], p["e"]
            if not p["quant"]:
                out.seek(base + entries[nm]["data_offsets"][0])
                copy_bytes(src, e, data_start, out)
                continue
            n, k = e["shape"]
            cr = p["cr"]
            t_start = time.time()
            print(f"[w{bits}a8] {done+1}/{n_quant} {nm[:52]} {n}x{k} "
                  f"cr={cr} ...", end="", flush=True)
            md = p.get("mode")
            if md in ("w4a4", "int8"):
                qo = base + entries[nm]["data_offsets"][0]
                so = base + entries[f"{nm}_scale"]["data_offsets"][0]
                if md == "int8":
                    amax = 0.0
                    for r0 in range(0, n, chunk_rows):
                        r1 = min(r0 + chunk_rows, n)
                        rows = read_rows(src, e, data_start, r0, r1)
                        amax = max(amax, float(rows.abs().max()))
                        del rows
                    sc = max(amax, 1e-10) / 127.0
                    for r0 in range(0, n, chunk_rows):
                        r1 = min(r0 + chunk_rows, n)
                        rows = read_rows(src, e, data_start, r0, r1)
                        q = (rows / sc).round().clamp(-127, 127).to(torch.int8)
                        out.seek(qo + r0 * k)
                        out.write(q.numpy().tobytes())
                        if r0 == 0 and len(errs) < verify_n:
                            errs.append((nm + " [int8]",
                                         ((q.float() * sc - rows).norm()
                                          / rows.norm().clamp(min=1e-12)
                                          ).item(), 0))
                        del rows, q
                    out.seek(so)
                    out.write(np.array([sc], dtype=np.float32).tobytes())
                else:
                    for r0 in range(0, n, chunk_rows):
                        r1 = min(r0 + chunk_rows, n)
                        rows = read_rows(src, e, data_start, r0, r1)
                        pk, sc4 = quant_rows_w4a4(rows, cr)
                        out.seek(qo + r0 * (k // 2))
                        out.write(pk.numpy().tobytes())
                        out.seek(so + r0 * 4)
                        out.write(sc4.numpy().tobytes())
                        if r0 == 0 and len(errs) < verify_n:
                            d = dequant_rows_w4a4(pk, sc4, cr)
                            errs.append((nm + " [w4a4]",
                                         ((d - rows).norm()
                                          / rows.norm().clamp(min=1e-12)
                                          ).item(), cr))
                        del rows, pk, sc4
                out.seek(base + entries[p["cq_key"]]["data_offsets"][0])
                out.write(p["cfg"])
                done += 1
                rate = done / max(time.time() - run_start, 1e-9)
                print(f" {time.time()-t_start:6.1f}s [{md}]   eta "
                      f"{(n_quant-done)/rate/60:6.1f} min", flush=True)
                continue
            # pass A: sample for codebook
            samp = []
            for r0 in range(0, n, chunk_rows):
                r1 = min(r0 + chunk_rows, n)
                rows = read_rows(src, e, data_start, r0, r1)
                rot = rotate(rows, cr).view(r1 - r0, k // gsz, gsz)
                gs = rot.abs().amax(-1, keepdim=True).clamp(min=1e-8)
                v = (rot / gs).flatten()
                if v.numel() > 40000:
                    g = torch.Generator().manual_seed(0)
                    v = v[torch.randint(0, v.numel(), (40000,), generator=g)]
                samp.append(v)
                if len(samp) * 40000 > 300000:
                    break
            cb = fit_codebook(torch.cat(samp), levels)
            del samp
            # pass B: quantize and write
            qo = base + entries[nm]["data_offsets"][0]
            so = base + entries[f"{nm}_s_rel"]["data_offsets"][0]
            co = base + entries[f"{nm}_s_channel"]["data_offsets"][0]
            prb = packed_row_bytes(k, bits)
            first = None
            for r0 in range(0, n, chunk_rows):
                r1 = min(r0 + chunk_rows, n)
                rows = read_rows(src, e, data_start, r0, r1)
                pk, s_rel, s_ch = quant_rows(rows, cb, gsz, cr, levels)
                out.seek(qo + r0 * prb)
                out.write(pk.numpy().tobytes())
                out.seek(so + r0 * (k // gsz))
                out.write(s_rel.view(torch.uint8).numpy().tobytes())
                out.seek(co + r0 * 4)
                out.write(s_ch.numpy().tobytes())
                if first is None and len(errs) < verify_n:
                    deq = dequant_rows(pk, s_rel, s_ch, cb, k, bits, gsz, cr)
                    rel = ((deq - rows).norm() / rows.norm().clamp(min=1e-12)).item()
                    errs.append((nm, rel, cr))
                    first = rel
                del rows, pk, s_rel, s_ch
            out.seek(base + entries[f"{nm}_codebook"]["data_offsets"][0])
            out.write(cb.float().numpy().tobytes())
            out.seek(base + entries[p["cq_key"]]["data_offsets"][0])
            out.write(p["cfg"])
            done += 1
            el = time.time() - t_start
            rate = done / max(time.time() - run_start, 1e-9)
            eta = (n_quant - done) / rate if rate > 0 else 0
            print(f" {el:6.1f}s   eta {eta/60:6.1f} min", flush=True)

    print(f"\n[w{bits}a8] wrote {dst.name} "
          f"({dst.stat().st_size/1e9:.2f} GB, source "
          f"{src.stat().st_size/1e9:.2f} GB)")
    if errs:
        e_ = np.array([x[1] for x in errs])
        print(f"[w{bits}a8] sample relL2: mean {e_.mean()*100:.3f}%  "
              f"worst {e_.max()*100:.3f}%")
        for nm, rel, cr in errs:
            print(f"    {rel*100:7.3f}%  cr={cr:<4} {nm[:58]}")


if __name__ == "__main__":
    main()

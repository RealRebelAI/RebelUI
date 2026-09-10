r"""Restore selected tensors in a quantized GGUF to F16 from an F16 reference.

Usage:
  python_embeded\python.exe D:\minimax\gguf_swap_hiprec.py <quant.gguf> <f16_ref.gguf> --keys audio_patch_proj,time_embedder,condition_proj,final_layer,video_patch_proj,rope.

Writes <quant>-fixed.gguf next to the input. Original untouched.
Any tensor whose name contains one of the --keys substrings and is stored
quantized gets replaced by the F16 reference copy. Everything else, and all
metadata (including comfy.gguf.orig_shape.*), is copied unchanged.
"""
import sys
from pathlib import Path

import numpy as np


def main():
    args = sys.argv[1:]
    keys = []
    mix_keys = []
    mix_path = None
    pos = []
    i = 0
    while i < len(args):
        if args[i] == "--keys" and i + 1 < len(args):
            keys = [k for k in args[i + 1].split(",") if k]
            i += 2
        elif args[i] == "--mix" and i + 1 < len(args):
            mix_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--mix-keys" and i + 1 < len(args):
            mix_keys = [k for k in args[i + 1].split(",") if k]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    if len(pos) < 2 or not (keys or mix_keys):
        print("usage: gguf_swap_hiprec.py <quant.gguf> <f16_ref.gguf> "
              "--keys a,b,c [--mix <donor.gguf> --mix-keys d,e]")
        sys.exit(1)
    if mix_keys and (mix_path is None or not mix_path.exists()):
        print(f"[swap] --mix donor file missing: {mix_path}")
        sys.exit(1)
    qpath, rpath = Path(pos[0]), Path(pos[1])
    out = qpath.with_name(qpath.stem + "-fixed.gguf")

    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.constants import GGMLQuantizationType as Q

    print(f"[swap] reading {qpath.name}")
    reader = GGUFReader(str(qpath))
    print(f"[swap] reading reference {rpath.name}")
    ref = {t.name: t for t in GGUFReader(str(rpath)).tensors}
    mix = {}
    if mix_keys:
        print(f"[swap] reading mix donor {mix_path.name}")
        mix = {t.name: t for t in GGUFReader(str(mix_path)).tensors}

    fields = {k: v for k, v in reader.fields.items()
              if not k.startswith("GGUF.")}

    def field_value(field):
        vt = field.types[0]
        if vt == GGUFValueType.ARRAY:
            sub = field.types[1]
            if sub == GGUFValueType.STRING:
                return [str(bytes(field.parts[i]), "utf-8")
                        for i in field.data]
            return [field.parts[i].tolist()[0] for i in field.data]
        if vt == GGUFValueType.STRING:
            return str(bytes(field.parts[field.data[0]]), "utf-8")
        return field.parts[field.data[0]].tolist()[0]

    arch = field_value(fields["general.architecture"])
    writer = GGUFWriter(str(out), arch)
    copied = 0
    for name, field in fields.items():
        if name == "general.architecture":
            continue
        vt = field.types[0]
        val = field_value(field)
        if vt == GGUFValueType.ARRAY:
            writer.add_array(name, val)
        else:
            writer.add_key_value(name, val, vt)
        copied += 1
    print(f"[swap] metadata: {copied} keys copied (arch '{arch}')")

    def is_target(name):
        return any(k in name for k in keys)

    def is_mix(name):
        return any(k in name for k in mix_keys)

    swapped = kept = mixed = 0
    swapped_names = []
    for t in reader.tensors:
        qt = Q(t.tensor_type)
        if is_mix(t.name) and t.name in mix:
            m = mix[t.name]
            mq = Q(m.tensor_type)
            if tuple(m.shape.tolist()) != tuple(t.shape.tolist()):
                print(f"[swap]  WARNING shape mismatch vs donor for "
                      f"{t.name}; kept original")
                writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=qt)
                kept += 1
                continue
            writer.add_tensor(t.name, np.asarray(m.data), raw_dtype=mq)
            mixed += 1
            continue
        if is_target(t.name) and qt not in (Q.F32, Q.F16, Q.BF16):
            r = ref.get(t.name)
            if r is None:
                print(f"[swap]  WARNING no reference for {t.name}; kept")
                writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=qt)
                kept += 1
                continue
            rq = Q(r.tensor_type)
            shape = tuple(reversed(r.shape.tolist()))
            arr = np.asarray(r.data)
            if rq == Q.BF16:
                u = arr.view(np.uint16).astype(np.uint32) << 16
                f = u.view(np.float32).reshape(shape).astype(np.float16)
            elif rq == Q.F32:
                f = arr.reshape(shape).astype(np.float16)
            else:
                f = arr.reshape(shape).astype(np.float16)
            writer.add_tensor(t.name, f)
            swapped += 1
            swapped_names.append(t.name)
        else:
            writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=qt)
            kept += 1
    print(f"[swap] tensors: {swapped} restored to F16, {mixed} taken from "
          f"donor, {kept} unchanged")
    for n in swapped_names[:20]:
        print(f"         {n}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"[swap] wrote {out.name}")

    check = GGUFReader(str(out))
    bad = [t.name for t in check.tensors
           if is_target(t.name) and not is_mix(t.name)
           and Q(t.tensor_type) not in (Q.F32, Q.F16)]
    from collections import Counter
    mtypes = Counter(Q(t.tensor_type).name for t in check.tensors
                     if is_mix(t.name))
    if mix_keys:
        print(f"[swap] mix tensors now: {dict(mtypes)}")
    print(f"[swap] VERIFY: {len(check.tensors)} tensors; "
          f"{'ALL target tensors F16/F32 - OK' if not bad else 'STILL QUANTIZED: ' + ', '.join(bad[:5])}")


if __name__ == "__main__":
    main()

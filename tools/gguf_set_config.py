r"""Write a model config blob into a GGUF's KV metadata.

ComfyUI-GGUF's get_gguf_metadata() exposes every simple scalar KV field by
name, and ComfyUI's model_detection does:

    dit_config.update(json.loads(metadata["config"]).get("transformer", {}))

So a GGUF KV field named "config" holding the JSON from the original
safetensors __metadata__ makes architectures like LTX-2.5 (whose dimensions
cannot be derived from tensor names) load with the STOCK GGUF loader.

Usage:
  # copy the config out of the original safetensors
  python_embeded\python.exe gguf_set_config.py <file.gguf> --from <src.safetensors>

  # or from a text file / literal JSON
  python_embeded\python.exe gguf_set_config.py <file.gguf> --config-file <cfg.txt>
  python_embeded\python.exe gguf_set_config.py <file.gguf> --config "{...}"

  # inspect
  python_embeded\python.exe gguf_set_config.py <file.gguf> --show

Adding a KV grows the header, so this streams the file to a new copy and
replaces it. Tensor data is copied byte-for-byte.
"""
import json
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
V_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
           4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
           10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
V_STRING, V_ARRAY = 8, 9
CHUNK = 64 << 20


def w_str(parts, s):
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    parts.append(struct.pack("<Q", len(b)))
    parts.append(b)


def field_value(field):
    import gguf as _g
    VT = _g.GGUFValueType
    vt = field.types[0]
    if vt == VT.ARRAY:
        sub = field.types[1]
        if sub == VT.STRING:
            return [str(bytes(field.parts[i]), "utf-8") for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    if vt == VT.STRING:
        return str(bytes(field.parts[field.data[0]]), "utf-8")
    return field.parts[field.data[0]].tolist()[0]


def encode_kv(parts, key, vtypes, value):
    w_str(parts, key)
    vt = int(vtypes[0])
    parts.append(struct.pack("<I", vt))
    if vt == V_ARRAY:
        sub = int(vtypes[1])
        parts.append(struct.pack("<I", sub))
        parts.append(struct.pack("<Q", len(value)))
        if sub == V_STRING:
            for item in value:
                w_str(parts, item)
        else:
            fmt, _ = V_FIXED[sub]
            for item in value:
                parts.append(struct.pack(fmt, item))
    elif vt == V_STRING:
        w_str(parts, value)
    else:
        fmt, _ = V_FIXED[vt]
        parts.append(struct.pack(fmt, value))



def _release_and_replace(tmp, path, tries=20):
    """Windows keeps the mmap alive until every numpy view is dropped.
    Force collection, then retry the swap with backoff."""
    import gc
    import os
    import time
    gc.collect()
    last = None
    for i in range(tries):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError as e:
            last = e
            gc.collect()
            time.sleep(0.25 * (i + 1))
    raise RuntimeError(
        f"could not replace {path.name} after {tries} tries: {last}. "
        f"The new file is at {tmp.name} -- close anything using the original "
        f"(ComfyUI, an editor) and rename it manually.")

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    path = Path(args[0])
    src, cfg_file, cfg_text, show = None, None, None, False
    i = 1
    while i < len(args):
        if args[i] == "--from":
            src = Path(args[i + 1]); i += 2
        elif args[i] == "--config-file":
            cfg_file = Path(args[i + 1]); i += 2
        elif args[i] == "--config":
            cfg_text = args[i + 1]; i += 2
        elif args[i] == "--show":
            show = True; i += 1
        else:
            print(f"unknown option {args[i]}"); sys.exit(1)
    if not path.exists():
        print(f"[config] not found: {path}"); sys.exit(1)

    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType as Q

    reader = GGUFReader(str(path))
    fields = {k: v for k, v in reader.fields.items()
              if not k.startswith("GGUF.")}
    cur = fields.get("config")
    print(f"[config] {path.name}: existing 'config' KV: "
          f"{'yes (' + str(len(field_value(cur))) + ' chars)' if cur else 'no'}")
    if show:
        if cur:
            print(field_value(cur)[:2000])
        return

    if src is not None:
        with open(src, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n).decode("utf-8")).get("__metadata__") or {}
        blob = meta.get("config")
        if not blob:
            print(f"[config] {src.name} has no __metadata__['config']")
            sys.exit(1)
    elif cfg_file is not None:
        blob = cfg_file.read_text(encoding="utf-8").strip()
    elif cfg_text is not None:
        blob = cfg_text
    else:
        print("[config] need --from, --config-file, --config or --show")
        sys.exit(1)

    try:
        parsed = json.loads(blob)
    except Exception as e:
        print(f"[config] not valid JSON: {e}"); sys.exit(1)
    tkeys = len(parsed.get("transformer", {}))
    print(f"[config] writing config ({len(blob)} chars, "
          f"{tkeys} transformer keys)")

    alignment = 32
    if "general.alignment" in fields:
        alignment = int(field_value(fields["general.alignment"]))
    old_base = min(int(t.data_offset) for t in reader.tensors)

    n_kv = len(fields) + (0 if cur else 1)
    parts = [struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION,
                         len(reader.tensors), n_kv)]
    for name, field in fields.items():
        if name == "config":
            w_str(parts, "config")
            parts.append(struct.pack("<I", V_STRING))
            w_str(parts, blob)
        else:
            encode_kv(parts, name, field.types, field_value(field))
    if not cur:
        w_str(parts, "config")
        parts.append(struct.pack("<I", V_STRING))
        w_str(parts, blob)

    rel0 = None
    for t in reader.tensors:
        if rel0 is None:
            rel0 = int(t.data_offset)
        w_str(parts, t.name)
        dims = [int(d) for d in t.shape.tolist()]
        parts.append(struct.pack("<I", len(dims)))
        for d in dims:
            parts.append(struct.pack("<Q", d))
        parts.append(struct.pack("<I", int(t.tensor_type)))
        parts.append(struct.pack("<Q", int(t.data_offset) - rel0))
    hdr = b"".join(parts)
    rem = len(hdr) % alignment
    new_base = len(hdr) + ((alignment - rem) if rem else 0)
    # every ReaderField holds numpy views backed by the file's mmap;
    # all of them must go before Windows will let us replace the file
    import gc
    del reader
    fields = None
    cur = None
    gc.collect()

    if new_base == old_base:
        with open(path, "r+b") as f:
            f.write(hdr + b"\0" * (old_base - len(hdr)))
        print(f"[config] in-place header write; data untouched")
    else:
        # Grow the header IN PLACE: extend the file, shift the tensor data
        # forward from the end, then write the new header. No temp file and
        # no rename, so Defender / Controlled Folder Access can't block it,
        # and it needs no extra free space.
        import os
        delta = new_base - old_base
        size = path.stat().st_size
        data_len = size - old_base
        print(f"[config] header grew {delta} B -> shifting {data_len/1e9:.2f} "
              f"GB in place (no temp file)")
        with open(path, "r+b") as f:
            f.truncate(size + delta)
            pos = data_len
            done = 0
            while pos > 0:
                n = min(CHUNK, pos)
                pos -= n
                f.seek(old_base + pos)
                buf = f.read(n)
                if len(buf) != n:
                    raise RuntimeError("short read during in-place shift")
                f.seek(new_base + pos)
                f.write(buf)
                done += n
                if done % (CHUNK * 8) == 0:
                    print(f"[config]   shifted {done/1e9:.1f}/"
                          f"{data_len/1e9:.1f} GB", flush=True)
                del buf
            f.seek(0)
            f.write(hdr + b"\0" * (new_base - len(hdr)))
            f.flush()
            os.fsync(f.fileno())
        print(f"[config] wrote new header ({new_base} B)")

    r2 = GGUFReader(str(path))
    got = field_value(r2.fields["config"])
    counts = {}
    for t in r2.tensors:
        counts[Q(t.tensor_type).name] = counts.get(Q(t.tensor_type).name, 0) + 1
    ok = json.loads(got).get("transformer", {})
    print(f"[config] VERIFY: {len(r2.tensors)} tensors, qtypes {counts}")
    print(f"[config] VERIFY: config readable, {len(ok)} transformer keys, "
          f"cross_attention_dim={ok.get('cross_attention_dim')}")


if __name__ == "__main__":
    main()

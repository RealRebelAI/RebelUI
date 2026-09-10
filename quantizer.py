"""RebelUI Auto-Quantizer.

Metadata/probe operations never load model weights. Remote safetensors headers are
read with HTTP Range requests. Conversion delegates to the existing low-RAM tools
in D:\\minimax and downloads at most one source shard at a time for safetensors
w4a8/int8 output.

No tokens, auth storage, telemetry, or hosted RebelUI service are implemented.
"""
from __future__ import annotations
import json, os, re, shutil, struct, subprocess, sys, tempfile, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HF_HOST = "https://huggingface.co"

# Keep the runtime toolchain inside RebelUI so the quantizer is portable.
# Environment variables still override these paths for development/testing.
HERE = Path(__file__).resolve().parent
TOOLS = Path(os.environ.get("REBELUI_QUANT_TOOLS", str(HERE / "tools")))
LLAMA_QUANTIZE = Path(os.environ.get(
    "REBELUI_LLAMA_QUANTIZE",
    str(TOOLS / "llama" / "llama-quantize.exe")
))
SAFE_LICENSES = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
    "cc0-1.0", "unlicense", "isc", "mpl-2.0"
}
TIER_BPE = {
    "Q2_K": .34, "Q3_K_M": .44, "Q4_K_S": .50, "Q4_K_M": .56,
    "Q5_K_M": .69, "Q6_K": .81, "Q8_0": 1.06, "w4a8": .5625, "int8": 1.0
}
HEURISTIC = r"(embed|text_|caption_|time_|patch_|proj_in|proj_out|final_layer|norm_out|rope|freqs|adaln|modulation)"

RECIPES = [
    dict(name="flux", signatures=("double_blocks.", "single_blocks.", "img_attn"),
         # Flux/Klein-safe INT8/W4A8 recipe: keep conditioning, modulation,
         # normalization, embedding and final/output paths at source precision.
         exclude=r"(img_in|txt_in|time_in|vector_in|guidance_in|final_layer|pe_embedder|rope|freqs|img_mod|txt_mod|modulation|adaln|ada_ln|scale_shift|norm|embed)",
         int8_keys="", group_size=16, gguf_arch="flux",
         hiprec="img_in,txt_in,time_in,vector_in,guidance_in,final_layer,pe_embedder,rope,freqs,img_mod,txt_mod,modulation,adaln,ada_ln,scale_shift,norm,embed",
         notes="Flux/Klein-safe: protect conditioning, modulation, normalization, embedding and final/output paths."),
    dict(name="qwen_image", signatures=("transformer_blocks.", ".img_mod", ".txt_mod"),
         exclude=r"(^img_in|^proj_out|^norm_out|^txt_in|time_text_embed|^txt_norm|norm_q|norm_k|norm_added)",
         int8_keys="weight", group_size=16, gguf_arch="qwen_image",
         hiprec="img_in,proj_out,norm_out,txt_in,time_text_embed,txt_norm,norm_q,norm_k,norm_added",
         notes="Qwen-Image family defaults to int8; 4-bit quality is not recommended."),
    dict(name="minimax_h3", signatures=("blocks.", "adaln_proj.linear", "audio_patch_proj"),
         exclude=r"(audio_patch_proj|time_embedder|condition_proj|final_layer|video_patch_proj|rope\.)",
         int8_keys="", group_size=16, gguf_arch="wan",
         hiprec="audio_patch_proj,time_embedder,condition_proj,final_layer,video_patch_proj,rope.",
         notes="adaln_proj K=2688-class layers take automatic int8 fallback when not /256."),
    dict(name="wan", signatures=("blocks.", "self_attn.q", "text_embedding.0", "patch_embedding"),
         exclude=r"(patch_embedding|time_embedding|time_projection|text_embedding|img_emb|head\.|rope\.|modulation)",
         int8_keys="", group_size=16, gguf_arch="wan",
         hiprec="patch_embedding,time_embedding,time_projection,text_embedding,img_emb,head,rope.",
         notes="text_embedding.0/2 is the text path; keep it high precision."),
    dict(name="ltx25", signatures=("transformer_blocks.", "scale_shift_table", "audio_to_video_attn"),
         exclude=r"(patchify_proj|adaln|caption|text|time|proj_in|proj_out|norm_out|rope|to_gate_logits|audio_embeddings_connector)",
         int8_keys="", group_size=16, gguf_arch="ltxv",
         hiprec="patchify_proj,adaln,caption,text,time,proj_in,proj_out,norm_out,rope,to_gate_logits,audio_embeddings_connector",
         notes="Metadata config must be preserved; cross-modal gate tensors are protected."),
    dict(name="nextdit_zimage_lumina", signatures=("cap_embedder", "context_refiner", "x_embedder"),
         exclude=r"(x_embedder|cap_embedder|cap_pad_token|t_embedder|time_|final_layer|rope|freqs|adaLN)",
         int8_keys="weight", group_size=32, gguf_arch="lumina",
         hiprec="x_embedder,cap_embedder,cap_pad_token,t_embedder,time_,final_layer,rope,freqs,adaLN",
         notes="Small NextDiT/Z-Image-family models default to int8; known Z-Image group size is 32."),
    dict(name="minimax_music", signatures=("diffusion_transformer.preprocess_conv", "latent_conditioners"),
         exclude=r"(preprocess_conv|latent_conditioners|embed|time_|text_|caption_|proj_in|proj_out|final_layer|norm_out|rope|adaln)",
         int8_keys="weight", group_size=16, gguf_arch="wan",
         hiprec="preprocess_conv,latent_conditioners,embed,time_,text_,caption_,proj_in,proj_out,final_layer,norm_out,rope,adaln",
         notes="Audio paths default to int8; conditioning stays high precision."),
    dict(name="sensenova_unified", signatures=("language_model.", "vision_model.", "fm_modules."),
         exclude=r"(language_model\.embed|vision_model|fm_modules\..*(embed|proj_in|proj_out|norm_out|rope|adaln)|text_|time_|patch_)",
         int8_keys="weight", group_size=16, gguf_arch="wan",
         hiprec="language_model.embed,vision_model,proj_in,proj_out,norm_out,rope,adaln,text_,time_,patch_",
         notes="Unified AR+diffusion architecture; recipe is conservative."),
]

def parse_source(value: str):
    value = value.strip().rstrip("/")
    if value.startswith("https://huggingface.co/"):
        bits = value.split("huggingface.co/", 1)[1].split("/")
        if len(bits) < 2: raise ValueError("Expected a HuggingFace org/repo URL.")
        return "hf", f"{bits[0]}/{bits[1]}"
    p = Path(value).expanduser()
    if p.exists(): return "local", str(p.resolve())
    if re.fullmatch(r"[^/\s]+/[^/\s]+", value):
        return "hf", value
    raise ValueError("Enter https://huggingface.co/<org>/<repo>, <org>/<repo>, or an existing local path.")

def _hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except Exception as e:
        raise RuntimeError("huggingface_hub is required for remote repositories.") from e
    return HfApi, hf_hub_download, GatedRepoError, HfHubHTTPError

def list_files(kind, source):
    if kind == "local":
        p=Path(source)
        if p.is_file():
            return [p.name]
        base=p
        return [str(x.relative_to(base)).replace("\\","/") for x in base.rglob("*") if x.is_file()]
    HfApi, _, GatedRepoError, HfHubHTTPError = _hf()
    try:
        return HfApi().list_repo_files(source, repo_type="model", token=False)
    except GatedRepoError:
        raise PermissionError("GATED: This repository requires license access. Accept the license on HuggingFace, then use a local downloaded file/folder. RebelUI does not accept or store HuggingFace tokens.")
    except HfHubHTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) in (401,403):
            raise PermissionError("GATED/PRIVATE: RebelUI does not implement HuggingFace authentication. Accept the license and use a local file/folder.")
        raise

def fetch_small(kind, source, rel):
    if kind=="local":
        p=Path(source)
        base=p if p.is_dir() else p.parent
        f=base/rel
        return f.read_bytes() if f.exists() else None
    url=f"{HF_HOST}/{source}/resolve/main/{rel}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r: return r.read()
    except Exception: return None

def hf_range(repo, rel, end):
    url=f"{HF_HOST}/{repo}/resolve/main/{rel}"
    req=urllib.request.Request(url, headers={"Range":f"bytes=0-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def remote_header(repo, rel):
    buf=hf_range(repo, rel, 2*1024*1024-1)
    if len(buf)<8: raise RuntimeError(f"Short safetensors header response: {rel}")
    n=struct.unpack("<Q",buf[:8])[0]
    need=8+n
    if len(buf)<need:
        buf=hf_range(repo, rel, need-1)
    if len(buf)<need: raise RuntimeError(f"Could not retrieve complete safetensors header: {rel}")
    return json.loads(buf[8:need].decode("utf-8")), need

def local_header(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8")),8+n

def discover(files):
    indexes=[f for f in files if f.endswith(".safetensors.index.json")]
    safes=[f for f in files if f.endswith(".safetensors")]
    layout="single-folder"
    if "model_index.json" in files or any(f.startswith(("transformer/","text_encoder/","vae/")) for f in files):
        layout="diffusers"
    elif any(f.startswith("split_files/") for f in files):
        layout="comfy-org-repack"
    comps={}
    for f in indexes+safes:
        part=f.split("/")
        comp=part[-2] if len(part)>1 else "root"
        if "diffusion_models" in part: comp="diffusion_model"
        comps.setdefault(comp,[]).append(f)
    return layout, comps, indexes, safes

def shard_files(kind, source, files, indexes, safes):
    out=[]
    for idx in indexes:
        raw=fetch_small(kind,source,idx)
        if not raw: continue
        try: data=json.loads(raw)
        except Exception: continue
        prefix=idx.rsplit("/",1)[0]+"/" if "/" in idx else ""
        for shard in dict.fromkeys(data.get("weight_map",{}).values()):
            rel=shard if "/" in shard else prefix+shard
            if rel in files and rel not in out: out.append(rel)
    for f in safes:
        if f not in out: out.append(f)
    return out

def probe(kind, source, shards):
    names=set(); dtypes={}; total_params=0; total_bytes=0
    bad32=[]; bad256=[]; rescuable=[]; nonblock=[]; metadata_configs=[]
    per_shard=[]
    for rel in shards:
        if kind=="hf": h,data_start=remote_header(source,rel)
        else:
            base=Path(source) if Path(source).is_dir() else Path(source).parent
            h,data_start=local_header(base/rel)
        md=h.get("__metadata__") or {}
        if isinstance(md,dict) and md.get("config"): metadata_configs.append({"file":rel,"config":md.get("config")})
        count=params=0
        for name,e in h.items():
            if name=="__metadata__" or not isinstance(e,dict) or "shape" not in e: continue
            shape=e.get("shape") or []; dtype=e.get("dtype","?")
            n=1
            for d in shape:n*=int(d)
            count+=1; params+=n; total_params+=n
            dtypes[dtype]=dtypes.get(dtype,0)+n
            names.add(name)
            offs=e.get("data_offsets",[0,0]); total_bytes += max(0,int(offs[1])-int(offs[0]))
            if len(shape)==2:
                k=int(shape[1])
                if k%32: bad32.append({"name":name,"shape":shape})
                if k%256:
                    bad256.append({"name":name,"shape":shape,"reason":f"K={k} is not divisible by 256; w4a8 fused ConvRot kernel requires /256, so this layer takes int8 fallback."})
                    if n%256==0: rescuable.append({"name":name,"shape":shape})
            if not re.search(r"(?:^|\.)(?:blocks?|transformer_blocks|double_blocks|single_blocks)\.\d+\.",name):
                nonblock.append({"name":name,"dtype":dtype,"shape":shape})
        per_shard.append({"file":rel,"tensors":count,"params":params})
    return dict(names=sorted(names),tensor_count=len(names),params=total_params,bytes=total_bytes,
                dtypes=dtypes,bad32=bad32,bad256=bad256,rescuable=rescuable,
                nonblock=nonblock[:500],metadata_configs=metadata_configs,per_shard=per_shard)

def fingerprint(names):
    joined="\n".join(names)
    for r in RECIPES:
        if all(s in joined for s in r["signatures"]):
            return dict(r, matched=True)
    return dict(name="unknown",matched=False,exclude=HEURISTIC,int8_keys="",group_size=16,
                gguf_arch="wan",hiprec="embed,text_,caption_,time_,patch_,proj_in,proj_out,final_layer,norm_out,rope,freqs,adaln,modulation",
                notes="Unknown architecture: heuristic sensitive-layer recipe. Review before starting.")

def license_info(kind,source):
    if kind=="local": return {"license":"local/unknown","warning":"Local source: license could not be verified automatically."}
    HfApi,_,_,_= _hf()
    try:
        info=HfApi().model_info(source, token=False)
        card=getattr(info,"cardData",None) or {}
        lic=(card.get("license") if hasattr(card,"get") else None) or "unknown"
    except Exception: lic="unknown"
    warn=None
    if str(lic).lower() not in SAFE_LICENSES:
        warn=f"License '{lic}' is non-permissive or unknown. Confirm you have the right to create/use derivative quantized weights."
    return {"license":lic,"warning":warn}

def resolve_plan(value):
    kind,source=parse_source(value)
    files=list_files(kind,source)  # gated check first
    layout,comps,indexes,safes=discover(files)
    shards=shard_files(kind,source,files,indexes,safes)
    if not shards: raise RuntimeError("No safetensors weights were found.")
    p=probe(kind,source,shards)
    recipe=fingerprint(p["names"])
    lic=license_info(kind,source)
    free=shutil.disk_usage(Path(source) if kind=="local" else Path.cwd()).free
    warnings=[]
    if lic["warning"]: warnings.append({"type":"license","message":lic["warning"]})
    if not recipe["matched"]: warnings.append({"type":"architecture","message":"Architecture is not in the recipe registry. Heuristic exclude list will be used and must be reviewed."})
    for x in p["bad256"]:
        warnings.append({"type":"int8_fallback","tensor":x["name"],"message":x["reason"]})
    download=p["bytes"]
    estimates={k:int(p["params"]*v) for k,v in TIER_BPE.items()}
    # rough load-time RAM estimate: file size + conservative runtime overhead, not build RAM.
    ramfits={k: estimates[k] < 14*(1024**3) for k in estimates}
    return {
      "kind":kind,"source":source,"files":files,"layout":layout,"components":comps,
      "shards":shards,"probe":{k:v for k,v in p.items() if k!="names"},
      "recipe":recipe,"license":lic,"warnings":warnings,
      "download_bytes":download,"free_disk":free,"estimates":estimates,"ramfits":ramfits,
    }

def _run(cmd, emit, cwd=None):
    emit("$ "+" ".join(map(str,cmd)))
    p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,encoding="utf-8",errors="replace",bufsize=1)
    for line in p.stdout:
        emit(line.rstrip())
    rc=p.wait()
    if rc: raise RuntimeError(f"Command failed ({rc}): {cmd[0]}")

def _download_one(repo, rel, scratch, emit):
    _,hf_hub_download,_,_=_hf()
    emit(f"download shard · {rel}")
    return Path(hf_hub_download(repo_id=repo,filename=rel,local_dir=str(scratch),token=False))

def _tool(name):
    p=TOOLS/name
    if not p.exists(): raise RuntimeError(f"Required quantization tool not found: {p}")
    return p

def run_quantization(plan, request, emit):
    """Run confirmed plan. Source shards are processed one at a time for ST tiers.
    GGUF tiers currently require a single-file source; sharded GGUF is refused
    rather than silently merging the entire source and violating the disk rule.
    """
    tiers=request.get("tiers") or []
    if not tiers: raise RuntimeError("Select at least one tier.")
    exclude=request.get("exclude",plan["recipe"]["exclude"])
    int8_keys=request.get("int8_keys",plan["recipe"].get("int8_keys",""))
    group=int(request.get("group_size",plan["recipe"].get("group_size",16)))
    outdir=Path(request.get("output_dir") or (Path.cwd()/"RebelUI-Quants"))
    outdir.mkdir(parents=True,exist_ok=True)
    kind,source=plan["kind"],plan["source"]
    shards=plan["shards"]
    ggufs=[t for t in tiers if t.startswith("Q")]
    sttiers=[t for t in tiers if t in ("w4a8","int8")]
    if ggufs and len(shards)>1:
        raise RuntimeError("GGUF ladder for a sharded remote source would require a full merged source with the current st_to_gguf.py. RebelUI refuses to do that automatically because it violates the one-shard peak-disk rule. Select w4a8/int8, or provide a local single-file safetensors source for GGUF.")
    scratch=Path(tempfile.mkdtemp(prefix="rebelui-quant-"))
    outputs=[]
    try:
        # Native safetensors tiers: quantize each source shard independently.
        for tier in sttiers:
            tierdir=outdir/tier
            tierdir.mkdir(exist_ok=True)
            for i,rel in enumerate(shards,1):
                emit(f"[{tier}] shard {i}/{len(shards)} · {rel}")
                if kind=="hf": src=_download_one(source,rel,scratch,emit)
                else:
                    base=Path(source) if Path(source).is_dir() else Path(source).parent
                    src=base/rel
                dst=tierdir/Path(rel).name
                cmd=[sys.executable,_tool("w4a8_convert.py"),src,dst,"--bits","4","--act-bits","8",
                     "--group-size",str(group),"--fallback","int8","--exclude",exclude,
                     "--chunk-rows","2048","--verify","8"]
                if tier=="int8": cmd += ["--int8-keys","weight"]
                elif int8_keys: cmd += ["--int8-keys",int8_keys]
                _run(cmd,emit)
                outputs.append(str(dst))
                if kind=="hf":
                    try: src.unlink()
                    except Exception: pass

        # GGUF path: only single safetensors source, so no source merge is needed.
        if ggufs:
            rel=shards[0]
            if kind=="hf": src=_download_one(source,rel,scratch,emit)
            else:
                base=Path(source) if Path(source).is_dir() else Path(source).parent
                src=base/rel
            f16=outdir/(Path(rel).stem+"-F16.gguf")
            _run([sys.executable,_tool("st_to_gguf.py"),src,f16,"--arch",plan["recipe"]["gguf_arch"],
                  "--hiprec",plan["recipe"]["hiprec"]],emit)
            for tier in ggufs:
                raw=outdir/(Path(rel).stem+f"-{tier}.gguf")
                _run([LLAMA_QUANTIZE,f16,raw,tier],emit)
                # mandatory hiprec restoration
                _run([sys.executable,_tool("gguf_swap_hiprec.py"),raw,f16,"--keys",plan["recipe"]["hiprec"]],emit)
                fixed=raw.with_name(raw.stem+"-fixed.gguf")
                target=fixed if fixed.exists() else raw
                if plan["probe"].get("metadata_configs"):
                    cfg=plan["probe"]["metadata_configs"][0]["config"]
                    _run([sys.executable,_tool("gguf_set_config.py"),target,"--config",cfg],emit)
                _run([sys.executable,_tool("gguf_fix_shapes.py"),target,"--from",src],emit)
                outputs.append(str(target))
            if kind=="hf":
                try: src.unlink()
                except Exception: pass

        readme=outdir/"README.md"
        readme.write_text(
            "---\nbase_model: "+source+"\n---\n\n# RebelUI quantization\n\n"
            f"- Architecture recipe: `{plan['recipe']['name']}`\n"
            f"- Exclude: `{exclude}`\n- Group size: `{group}`\n"
            f"- INT8 keys: `{int8_keys}`\n- High precision: `{plan['recipe']['hiprec']}`\n\n"
            "Generated locally with RebelUI Auto-Quantizer. Review converter verification output "
            "for copy-param count, format histogram and measured relL2 before publishing.\n",
            encoding="utf-8")
        emit(f"complete · {len(outputs)} output file(s)")
        return {"outputs":outputs,"readme":str(readme),"output_dir":str(outdir)}
    finally:
        shutil.rmtree(scratch,ignore_errors=True)

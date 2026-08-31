"""RebelUI — direct-to-library ComfyUI runner.

RebelUI imports ComfyUI's Python library directly. It supports both normal
PyTorch/safetensors weights and GGUF files, including GGUF files placed in
ComfyUI's normal diffusion_models / text_encoders folders.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STATIC = HERE / "static"

FULL_EXTS = {".safetensors", ".sft", ".pt", ".pth", ".ckpt", ".bin"}
GGUF_EXTS = {".gguf"}
ALL_MODEL_EXTS = FULL_EXTS | GGUF_EXTS


def bootstrap_comfy(comfy_dir, disable_pinned_memory=True):
    if comfy_dir:
        sys.path.insert(0, str(comfy_dir))

    # RebelUI is its own Python process; it does NOT inherit the command-line
    # flags used by the user's normal ComfyUI launcher. Apply memory-policy
    # flags before importing model_management because that module initializes
    # its memory behavior from comfy.cli_args at import time.
    import comfy.cli_args
    comfy.cli_args.args.disable_pinned_memory = bool(disable_pinned_memory)

    # Do not disable dynamic VRAM. Leave ComfyUI's normal dynamic/offload policy
    # intact; only mirror the user's no-pinned-memory configuration.
    if hasattr(comfy.cli_args.args, "disable_dynamic_vram"):
        comfy.cli_args.args.disable_dynamic_vram = False

    import folder_paths
    import comfy
    import comfy.sd
    import comfy.samplers
    import comfy.utils
    import comfy.model_management
    return folder_paths, comfy


# ── model discovery ──────────────────────────────────────────────────────
def _folder_roots(folder_paths, key):
    """Return every registered filesystem root for a ComfyUI model category."""
    roots = []
    try:
        roots.extend(folder_paths.get_folder_paths(key) or [])
    except Exception:
        pass

    # Some custom nodes register entries in folder_names_and_paths but older
    # Comfy builds/custom-node combinations do not expose get_folder_paths well.
    try:
        entry = folder_paths.folder_names_and_paths.get(key)
        if entry:
            raw = entry[0]
            if isinstance(raw, (str, Path)):
                raw = [raw]
            roots.extend(raw or [])
    except Exception:
        pass

    out, seen = [], set()
    for p in roots:
        p = str(Path(p))
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _scan(folder_paths, keys, extensions):
    """Scan registered folders directly, bypassing Comfy extension filters."""
    out, seen = [], set()

    # First take anything Comfy already knows about.
    for key in keys:
        try:
            for name in folder_paths.get_filename_list(key):
                if Path(name).suffix.lower() in extensions and name not in seen:
                    seen.add(name)
                    out.append(name)
        except Exception:
            pass

    # Then scan the actual directories so GGUF in diffusion_models/text_encoders
    # is still visible even when that category's extension whitelist excludes it.
    for key in keys:
        for root in _folder_roots(folder_paths, key):
            rp = Path(root)
            if not rp.exists():
                continue
            try:
                for f in rp.rglob("*"):
                    if not f.is_file() or f.suffix.lower() not in extensions:
                        continue
                    name = f.relative_to(rp).as_posix()
                    if name not in seen:
                        seen.add(name)
                        out.append(name)
            except OSError:
                pass

    return sorted(out, key=str.lower)


def _model_format(name):
    return "gguf" if str(name).lower().endswith(".gguf") else "full"


def _clip_types(comfy):
    """Expose the CLIPType values supported by the installed ComfyUI."""
    values = []
    try:
        enum = comfy.sd.CLIPType
        for item in enum:
            value = getattr(item, "name", str(item)).lower()
            if value not in values:
                values.append(value)
    except Exception:
        pass

    # Krea 2 is new enough that keeping this visible is useful even if the local
    # ComfyUI install is slightly behind; generation will give a clear error.
    for required in ("krea2", "qwen_image", "flux", "sd3", "wan", "ltxv"):
        if required not in values:
            values.append(required)
    return values


def _architectures(comfy):
    """Build an architecture browser from the installed ComfyUI model classes."""
    names = {"auto", "krea2"}
    try:
        import inspect
        import comfy.supported_models as sm
        for name, obj in vars(sm).items():
            if inspect.isclass(obj) and obj.__module__ == sm.__name__:
                if not name.startswith("_") and name.lower() not in {"base", "base_model"}:
                    names.add(name)
    except Exception:
        pass

    # Human-friendly aliases/presets that remain useful even when class names
    # differ between ComfyUI releases.
    names.update({
        "flux", "flux2", "qwen_image", "qwen_image_edit", "sd3", "sdxl",
        "sd15", "wan", "ltxv", "ltx2", "hunyuan_video", "hunyuan_image",
        "hidream", "cosmos", "mochi", "lumina2", "pixart"
    })
    return sorted(names, key=lambda x: (x != "auto", x.lower()))


def list_models(folder_paths, comfy):
    unet = _scan(folder_paths, ("unet_gguf", "unet", "diffusion_models"), ALL_MODEL_EXTS)
    clip = _scan(folder_paths, ("clip_gguf", "text_encoders", "clip"), ALL_MODEL_EXTS)
    vae = _scan(folder_paths, ("vae",), FULL_EXTS)
    return {
        "unet": [{"name": n, "format": _model_format(n)} for n in unet],
        "clip": [{"name": n, "format": _model_format(n)} for n in clip],
        "vae": vae,
        "clip_types": _clip_types(comfy),
        "architectures": _architectures(comfy),
        "tiers": ["Q2_K", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    }


def resolve(folder_paths, name, *keys):
    if not name:
        return None

    # Normal Comfy resolver first.
    for key in keys:
        try:
            p = folder_paths.get_full_path(key, name)
            if p:
                return p
        except Exception:
            pass

    # Direct filesystem fallback, important for .gguf living in a folder whose
    # Comfy category does not whitelist .gguf.
    normalized = str(name).replace("\\", "/")
    for key in keys:
        for root in _folder_roots(folder_paths, key):
            p = Path(root) / Path(normalized)
            if p.is_file():
                return str(p)

            # Last resort: match basename inside nested folders.
            try:
                for candidate in Path(root).rglob(Path(normalized).name):
                    if candidate.is_file():
                        return str(candidate)
            except OSError:
                pass
    return None


def gb(path):
    try:
        return round(os.path.getsize(path) / 1e9, 2)
    except Exception:
        return 0.0


# ── generation ───────────────────────────────────────────────────────────
class Runner:
    """Holds loaded components so consecutive runs don't reload."""

    def __init__(self, folder_paths, comfy):
        self.fp = folder_paths
        self.comfy = comfy
        self._cache = {}

    def _gguf_modules(self):
        """Initialize ComfyUI-GGUF once for either UNet OR encoder loading."""
        import glob
        import importlib
        import importlib.util
        import importlib.machinery

        base = Path(self.fp.base_path) / "custom_nodes"
        candidates = []
        for p in glob.glob(str(base / "*")):
            stem = Path(p).name.lower().replace("_", "-")
            if stem in ("comfyui-gguf", "comfyui-gguf-main"):
                candidates.append(p)
        if not candidates:
            raise RuntimeError(
                "ComfyUI-GGUF is not installed. It is required for .gguf "
                "diffusion models and .gguf text encoders."
            )

        pkg = "_rebelui_gguf"
        if pkg not in sys.modules:
            spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
            mod = importlib.util.module_from_spec(spec)
            mod.__path__ = [candidates[0]]
            sys.modules[pkg] = mod

        loader = importlib.import_module(f"{pkg}.loader")
        ops_mod = importlib.import_module(f"{pkg}.ops")
        return loader, ops_mod

    def _load_unet(self, name, emit):
        key = ("unet", name)
        if key in self._cache:
            return self._cache[key]

        path = resolve(self.fp, name, "unet_gguf", "unet", "diffusion_models")
        if not path:
            raise FileNotFoundError(f"diffusion model not found: {name}")

        emit(f"loading diffusion model  {name}")
        if name.lower().endswith(".gguf"):
            model = self._load_gguf_unet(path, emit)
        else:
            # IMPORTANT: use the same high-level loader as ComfyUI's UNETLoader.
            # load_diffusion_model() preserves current metadata, mmap/quantization,
            # model-options and native INT8/FP8 handling. Loading the state dict
            # ourselves bypasses parts of that path and can break models such as
            # Krea 2 Turbo INT8/FP8 before the first sampling step.
            import comfy.sd
            model = comfy.sd.load_diffusion_model(path, model_options={})
            if model is None:
                raise RuntimeError(
                    "ComfyUI could not detect this model architecture. "
                    "Update ComfyUI if this is a newly-supported architecture."
                )

        # Keep only one diffusion model cached to avoid accidental VRAM pile-up.
        for k in [k for k in self._cache if k[0] == "unet" and k != key]:
            self._cache.pop(k, None)
        self._cache[key] = model
        return model

    def _load_gguf_unet(self, path, emit):
        loader, ops_mod = self._gguf_modules()
        res = loader.gguf_sd_loader(path)
        sd, extra = (res if isinstance(res, tuple) else (res, {}))
        meta = (extra or {}).get("metadata", None)
        emit(f"gguf tensors: {len(sd)}")

        kwargs = {"model_options": {"custom_operations": ops_mod.GGMLOps()}}
        import inspect
        if meta and "metadata" in inspect.signature(
                self.comfy.sd.load_diffusion_model_state_dict).parameters:
            kwargs["metadata"] = meta

        model = self.comfy.sd.load_diffusion_model_state_dict(sd, **kwargs)
        if model is None:
            raise RuntimeError(
                "architecture not detected from GGUF. The model may require "
                "newer ComfyUI support or GGUF metadata/config keys."
            )
        return model

    def _load_clip(self, name, clip_type, emit):
        key = ("clip", name, clip_type)
        if key in self._cache:
            return self._cache[key]

        path = resolve(self.fp, name, "clip_gguf", "text_encoders", "clip")
        if not path:
            raise FileNotFoundError(f"text encoder not found: {name}")

        emit(f"loading text encoder  {name}  ({clip_type})")
        import comfy.sd

        clip_type_key = str(clip_type or "stable_diffusion").upper()
        if not hasattr(comfy.sd.CLIPType, clip_type_key):
            available = ", ".join(_clip_types(self.comfy))
            raise RuntimeError(
                f"CLIP type '{clip_type}' is not supported by this ComfyUI build. "
                f"Available: {available}. Update ComfyUI for newer encoder types."
            )
        ct = getattr(comfy.sd.CLIPType, clip_type_key)

        if name.lower().endswith(".gguf"):
            loader, _ = self._gguf_modules()
            clip_sd = loader.gguf_clip_loader(path)
            clip = comfy.sd.load_text_encoder_state_dicts(
                [clip_sd], embedding_directory=None, clip_type=ct
            )
        else:
            clip = comfy.sd.load_clip(ckpt_paths=[path], clip_type=ct)

        # Encoder cache can coexist with the active diffusion model.
        for k in [k for k in self._cache if k[0] == "clip" and k != key]:
            self._cache.pop(k, None)
        self._cache[key] = clip
        return clip

    def _load_vae(self, name, emit):
        key = ("vae", name)
        if key in self._cache:
            return self._cache[key]
        path = resolve(self.fp, name, "vae")
        if not path:
            raise FileNotFoundError(f"VAE not found: {name}")
        emit(f"loading vae  {name}")
        import comfy.sd

        # Match ComfyUI's native VAELoader: safetensors metadata is part of
        # VAE construction and must not be discarded. Some modern VAEs use it
        # for architecture/configuration details.
        sd, metadata = self.comfy.utils.load_torch_file(
            path, return_metadata=True
        )
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        if hasattr(vae, "throw_exception_if_invalid"):
            vae.throw_exception_if_invalid()

        # Current ComfyUI VAELoader registers a reload factory on the patcher.
        # This is important for CoreModelPatcher / dynamic-VRAM transitions:
        # when the VAE needs to be reloaded or cloned for a device transition,
        # the patcher must know how to reconstruct it from the original file.
        if hasattr(vae, "patcher") and hasattr(comfy.sd, "load_vae_patcher"):
            vae.patcher.cached_patcher_init = (
                comfy.sd.load_vae_patcher,
                (path, metadata, None),
            )

        for k in [k for k in self._cache if k[0] == "vae" and k != key]:
            self._cache.pop(k, None)
        self._cache[key] = vae
        emit(
            "vae patcher reload factory: "
            + ("ready" if getattr(getattr(vae, "patcher", None), "cached_patcher_init", None) else "missing")
        )
        return vae

    def run(self, req, emit):
        import torch

        # ComfyUI's normal execution engine runs node execution under
        # torch.inference_mode(). RebelUI must do the same, especially for
        # quantized models whose parameters are moved/offloaded dynamically.
        with torch.inference_mode():
            return self._run_inference(req, emit)

    def _run_krea2_native(self, req, emit):
        """Run Krea 2 using ComfyUI's built-in node implementations.

        This deliberately mirrors the official Krea 2 graph order:
        UNETLoader -> CLIPLoader -> CLIPTextEncode -> ConditioningZeroOut ->
        EmptyLatentImage -> KSampler -> VAELoader -> VAEDecode.

        The VAE is not constructed until sampling has finished.
        """
        import time
        import torch
        import nodes
        import comfy.model_management as mm

        if str(req["unet"]).lower().endswith(".gguf") or str(req["clip"]).lower().endswith(".gguf"):
            raise RuntimeError(
                "Native Krea 2 execution currently requires FULL/INT8/FP8 weights. "
                "Use the generic GGUF path for GGUF models."
            )

        emit("Krea 2 execution mode · native ComfyUI nodes")

        # Match the official loader nodes instead of manually constructing patchers.
        emit(f"loading diffusion model  {req['unet']}  (UNETLoader)")
        model = nodes.UNETLoader().load_unet(req["unet"], "default")[0]

        emit(f"loading text encoder  {req['clip']}  (CLIPLoader:krea2)")
        clip = nodes.CLIPLoader().load_clip(
            req["clip"], type="krea2", device="default"
        )[0]

        emit("encoding prompt  (CLIPTextEncode)")
        positive = nodes.CLIPTextEncode().encode(
            clip, req.get("positive", "")
        )[0]

        # The official Krea 2 Turbo workflow zeros the positive conditioning
        # to create the unconditional branch instead of encoding a negative prompt.
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        if req.get("negative", "").strip():
            emit("Krea 2 note · negative prompt ignored; official workflow uses zero conditioning")

        w = int(req["width"])
        h = int(req["height"])
        batch = max(1, int(req.get("batch", 1)))

        latent = nodes.EmptyLatentImage().generate(w, h, batch)[0]
        emit(f"native latent {tuple(latent['samples'].shape)}")

        emit(
            f"sampling with native KSampler · {int(req['steps'])} steps · "
            f"{req['sampler']} / {req['scheduler']}"
        )
        sample_started = time.time()
        sampled = nodes.KSampler().sample(
            model=model,
            seed=int(req["seed"]),
            steps=int(req["steps"]),
            cfg=float(req["cfg"]),
            sampler_name=req["sampler"],
            scheduler=req["scheduler"],
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )[0]
        emit(f"sampling complete in {time.time() - sample_started:.1f}s")

        # Drop conditioning/encoder references before VAE loader executes.
        # This mirrors the node boundary in a real Comfy graph more closely.
        del positive, negative, clip, latent

        try:
            render_dev = mm.get_torch_device()
            if torch.cuda.is_available() and getattr(render_dev, "type", None) == "cuda":
                idx = render_dev.index if render_dev.index is not None else torch.cuda.current_device()
                free_b, total_b = torch.cuda.mem_get_info(idx)
                emit(
                    f"VRAM after sampler · free={free_b/1024**3:.2f} GiB · "
                    f"used={(total_b-free_b)/1024**3:.2f} GiB"
                )
        except Exception:
            pass

        # IMPORTANT: only instantiate/load the VAE after KSampler has completed,
        # matching the dependency order of the official graph.
        emit(f"loading vae  {req['vae']}  (VAELoader after sampler)")
        vae = nodes.VAELoader().load_vae(req["vae"])[0]

        emit("decoding VAE · native VAEDecode node")
        decode_started = time.time()
        images = nodes.VAEDecode().decode(
            samples=sampled,
            vae=vae,
        )[0]
        emit(f"VAE decode finished in {time.time() - decode_started:.2f}s")
        emit(f"decoded image tensor {tuple(images.shape)}")

        return images, req

    def _run_inference(self, req, emit):
        # Krea 2 FULL/INT8/FP8 now uses ComfyUI's own node classes and official
        # graph order rather than RebelUI's hand-assembled sampler/VAE lifecycle.
        if str(req.get("architecture", "")).lower() == "krea2":
            unet_name = str(req.get("unet", ""))
            clip_name = str(req.get("clip", ""))
            if not unet_name.lower().endswith(".gguf") and not clip_name.lower().endswith(".gguf"):
                return self._run_krea2_native(req, emit)

        import torch
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import comfy.model_management as mm
        import latent_preview  # noqa: F401

        arch = req.get("architecture", "auto")
        emit(f"architecture preset  {arch}  (weight architecture is auto-detected by ComfyUI)")

        model = self._load_unet(req["unet"], emit)
        clip = self._load_clip(req["clip"], req.get("clip_type", "stable_diffusion"), emit)
        vae = self._load_vae(req["vae"], emit)

        emit("encoding prompt")
        pos = clip.encode_from_tokens_scheduled(
            clip.tokenize(req.get("positive", ""))
        )
        neg = clip.encode_from_tokens_scheduled(
            clip.tokenize(req.get("negative", ""))
        )

        w, h = int(req["width"]), int(req["height"])
        frames = max(1, int(req.get("frames", 1)))
        batch = max(1, int(req.get("batch", 1)))
        ch = getattr(model.model, "latent_format", None)
        latent_ch = getattr(ch, "latent_channels", 16) if ch else 16

        if frames > 1:
            lat = torch.zeros([
                batch, latent_ch, ((frames - 1) // 4) + 1, h // 8, w // 8
            ])
        else:
            lat = torch.zeros([batch, latent_ch, h // 8, w // 8])
        emit(f"latent {tuple(lat.shape)}")

        steps = int(req["steps"])
        t0 = [time.time()]
        emit_json = emit.json

        def cb(i, denoised, x, total):
            now = time.time()
            emit_json({"step": i + 1, "total": total, "its": now - t0[0]})
            t0[0] = now

        noise = comfy.sample.prepare_noise(lat, int(req["seed"]))

        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            float(req["cfg"]),
            req["sampler"],
            req["scheduler"],
            pos,
            neg,
            lat,
            denoise=1.0,
            callback=cb,
            seed=int(req["seed"]),
        )

        # Hand the sampler result directly to ComfyUI's managed VAE path.
        # comfy.sd.VAE.decode() performs its own model loading, device placement,
        # memory reservation/offloading and tiled fallback. Do not manually move
        # the latent or unload models around this call.
        emit("sampling complete · handing latent to managed VAE")

        try:
            render_dev = mm.get_torch_device()
            vae_dev = mm.vae_device()
            offload_dev = mm.vae_offload_device()
            emit(
                f"devices · render={render_dev} · vae={vae_dev} · "
                f"vae_offload={offload_dev} · latent={samples.device}"
            )
            if torch.cuda.is_available():
                idx = render_dev.index if getattr(render_dev, "index", None) is not None else torch.cuda.current_device()
                free_b, total_b = torch.cuda.mem_get_info(idx)
                emit(
                    f"VRAM before decode · free={free_b/1024**3:.2f} GB · "
                    f"total={total_b/1024**3:.2f} GB"
                )
        except Exception as diag_err:
            emit(f"decode diagnostics unavailable: {diag_err}")

        # Match ComfyUI's native VAEDecode node: pass the sampled latent
        # directly into VAE.decode(). Do not pre-load or manually re-manage the
        # VAE here; comfy.sd.VAE.decode() owns its memory/device lifecycle.
        emit("decoding VAE · native ComfyUI path")
        decode_started = time.time()
        imgs = vae.decode(samples)
        emit(f"VAE decode finished in {time.time() - decode_started:.1f}s")

        if imgs is None:
            raise RuntimeError("VAE decode returned no image tensor.")
        if not hasattr(imgs, "shape"):
            raise RuntimeError(f"VAE decode returned unexpected type: {type(imgs)!r}")

        emit(f"decoded image tensor {tuple(imgs.shape)}")
        return imgs, req


# ── http ─────────────────────────────────────────────────────────────────
def make_app(runner, folder_paths, comfy, out_dir):
    from aiohttp import web

    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def env(_):
        import torch
        import comfy.model_management as mm
        total = 0
        dev = mm.get_torch_device()
        name = str(dev)
        device_error = None
        try:
            if torch.cuda.is_available() and getattr(dev, "type", None) == "cuda":
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                total = props.total_memory / (1024 ** 3)
                name = props.name
            else:
                name = str(dev)
        except Exception as e:
            device_error = str(e)
            name = str(dev)
        try:
            import comfy.cli_args
            pinned_disabled = bool(comfy.cli_args.args.disable_pinned_memory)
            dynamic_disabled = bool(getattr(comfy.cli_args.args, "disable_dynamic_vram", False))
        except Exception:
            pinned_disabled = None
            dynamic_disabled = None
        return web.json_response({
            "device": name,
            "vram_gb": round(total, 1),
            "torch": torch.__version__,
            "device_error": device_error,
            "pinned_memory": "disabled" if pinned_disabled else "enabled",
            "dynamic_vram": "disabled" if dynamic_disabled else "enabled",
        })

    async def vram(_):
        import torch
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        result = {
            "device": str(dev),
            "free_gib": 0.0,
            "used_gib": 0.0,
            "total_gib": 0.0,
            "torch_allocated_gib": 0.0,
            "torch_reserved_gib": 0.0,
        }
        try:
            if torch.cuda.is_available() and getattr(dev, "type", None) == "cuda":
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
                free_b, total_b = torch.cuda.mem_get_info(idx)
                result.update({
                    "free_gib": free_b / (1024 ** 3),
                    "used_gib": (total_b - free_b) / (1024 ** 3),
                    "total_gib": total_b / (1024 ** 3),
                    "torch_allocated_gib": torch.cuda.memory_allocated(idx) / (1024 ** 3),
                    "torch_reserved_gib": torch.cuda.memory_reserved(idx) / (1024 ** 3),
                })
        except Exception as e:
            result["error"] = str(e)
        return web.json_response(result)

    async def models(_):
        return web.json_response(list_models(folder_paths, comfy))

    async def sampling(_):
        try:
            samplers = list(comfy.samplers.KSampler.SAMPLERS)
            schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        except Exception:
            samplers = ["euler", "euler_ancestral", "dpmpp_2m"]
            schedulers = ["simple", "normal", "sgm_uniform", "beta", "karras"]
        return web.json_response({"samplers": samplers, "schedulers": schedulers})

    async def size(request):
        q = request.query
        u = resolve(folder_paths, q.get("unet", ""), "unet_gguf", "unet", "diffusion_models")
        c = resolve(folder_paths, q.get("clip", ""), "clip_gguf", "text_encoders", "clip")
        v = resolve(folder_paths, q.get("vae", ""), "vae")
        return web.json_response({
            "unet": gb(u),
            "clip": gb(c),
            "vae": gb(v),
            "work": 1.1,
        })

    async def generate(request):
        req = await request.json()
        resp = web.StreamResponse(headers={"content-type": "application/x-ndjson"})
        await resp.prepare(request)
        loop = asyncio.get_running_loop()

        def send(obj):
            asyncio.run_coroutine_threadsafe(
                resp.write((json.dumps(obj) + "\n").encode()), loop
            )

        def emit(msg, level=None):
            send({"log": msg, "level": level})
        emit.json = send

        def work():
            return runner.run(req, emit)

        try:
            imgs, r = await loop.run_in_executor(None, work)
            path = save_output(imgs, r, out_dir)

            # IMPORTANT: the final output event must be awaited directly.
            # Using the fire-and-forget thread-safe helper here created a race
            # with write_eof(), so the PNG could save successfully while the
            # browser never received its /out/... path.
            output_url = "/out/" + Path(path).name
            await resp.write(
                (json.dumps({"output": output_url}) + "\n").encode()
            )
            await resp.drain()
        except Exception:
            import traceback
            err = traceback.format_exc()
            await resp.write(
                (json.dumps({"log": err, "level": "er"}) + "\n").encode()
            )

        await resp.write_eof()
        return resp

    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_get("/", index)
    app.router.add_get("/api/env", env)
    app.router.add_get("/api/vram", vram)
    app.router.add_get("/api/models", models)
    app.router.add_get("/api/sampling", sampling)
    app.router.add_get("/api/size", size)
    app.router.add_post("/api/generate", generate)
    app.router.add_static("/out/", out_dir)
    app.router.add_static("/static/", STATIC)
    return app


def save_output(imgs, req, out_dir):
    import numpy as np
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    arr = imgs
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype("uint8")

    if arr.ndim == 4 and arr.shape[0] > 1:
        tmp = out_dir / f"rebel-{stamp}"
        tmp.mkdir(exist_ok=True)
        for i, f in enumerate(arr):
            Image.fromarray(f).save(tmp / f"{i:05d}.png")
        mp4 = out_dir / f"rebel-{stamp}.mp4"
        rc = os.system(
            f'ffmpeg -y -loglevel error -framerate {req.get("fps",16)} '
            f'-i "{tmp}/%05d.png" -c:v libx264 -pix_fmt yuv420p "{mp4}"'
        )
        if rc == 0 and mp4.exists():
            return str(mp4)
        return str(tmp / "00000.png")

    single = arr[0] if arr.ndim == 4 else arr
    png = out_dir / f"rebel-{stamp}.png"
    Image.fromarray(single).save(png)
    return str(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=None, help="path to the ComfyUI directory")
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--listen", default="127.0.0.1")
    pm = ap.add_mutually_exclusive_group()
    pm.add_argument("--disable-pinned-memory", dest="disable_pinned_memory",
                    action="store_true",
                    help="disable ComfyUI pinned host memory (RebelUI default)")
    pm.add_argument("--enable-pinned-memory", dest="disable_pinned_memory",
                    action="store_false",
                    help="allow ComfyUI pinned host memory")
    ap.set_defaults(disable_pinned_memory=True)
    a = ap.parse_args()

    folder_paths, comfy = bootstrap_comfy(
        a.comfy, disable_pinned_memory=a.disable_pinned_memory
    )
    Path(a.out).mkdir(parents=True, exist_ok=True)
    runner = Runner(folder_paths, comfy)

    from aiohttp import web
    print(f"\n  RebelUI  ->  http://{a.listen}:{a.port}\n")
    web.run_app(
        make_app(runner, folder_paths, comfy, a.out),
        host=a.listen,
        port=a.port,
        print=None,
    )


if __name__ == "__main__":
    main()

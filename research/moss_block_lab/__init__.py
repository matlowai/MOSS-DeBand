import re


def parse_layers(spec):
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


class MOSSBlockSwap:
    """Copy exact weights from `donor` into `model` for chosen transformer layers and weight types.

    layers: "21-41,3,4"; empty = none. parts: comma list matched against the key after `layers.N.`
    (qkv, o, ffn_up_gate, ffn_down, norm) or "all". include_other also swaps every non-layer key
    (embedders, time embedding, shared modulation, final head, text embedder).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "donor": ("MODEL",),
                             "layers": ("STRING", {"default": "0-41"}),
                             "parts": ("STRING", {"default": "all"}),
                             "include_other": ("BOOLEAN", {"default": False}),
                             "ratio": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "swap"
    CATEGORY = "model/merging"

    def swap(self, model, donor, layers, parts, include_other, ratio):
        want = parse_layers(layers)
        parts = [p.strip() for p in parts.split(",") if p.strip()]
        m = model.clone()
        kp = donor.get_key_patches("diffusion_model.")
        n = 0
        for k in kp:
            mm = re.search(r"\.layers\.(\d+)\.(.+)$", k)
            if mm:
                if int(mm.group(1)) not in want:
                    continue
                if "all" not in parts and not any(p in mm.group(2) for p in parts):
                    continue
            elif not include_other:
                continue
            m.add_patches({k: kp[k]}, ratio, 1.0 - ratio)
            n += 1
        print(f"MOSSBlockSwap: swapped {n} tensors (layers={layers}, parts={parts}, other={include_other}, ratio={ratio})")
        return (m,)


NODE_CLASS_MAPPINGS = {"MOSSBlockSwap": MOSSBlockSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSBlockSwap": "MOSS Block Swap"}


import os
import numpy as np
import torch
import folder_paths
import comfy.utils
import comfy.patcher_extension

PROBES = {}


def column_spectrum(t):
    """t: (B, C, H, W). Remove each row's mean along W, average down the rows, FFT along W, sum power over channels -> (B, W//2+1)."""
    t = t - t.mean(dim=3, keepdim=True)
    col = t.mean(dim=2)
    return (torch.fft.rfft(col, dim=-1).abs() ** 2).sum(dim=1)


class MOSSStripeProbe:
    """Records, for every transformer layer and model call, the column-stripe spectrum of
    (a) the logit lens: the residual stream passed through the model's own output head,
    (b) the lens of what this layer added, and (c) the raw residual stream.
    Writes output/moss/probe/<tag>.npz after every model call."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "head_from": (folder_paths.get_filename_list("diffusion_models"),),
                             "tag": ("STRING", {"default": "probe"}),
                             "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                             "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "probe"
    CATEGORY = "model/debug"

    def probe(self, model, head_from, tag, width, height):
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("diffusion_models", head_from))
        head = {k: sd[k].float() for k in ("final_norm_video.weight", "final_linear_video.weight", "video_head_modulation.weight", "video_head_modulation.bias")}
        del sd
        H, W = height // 16, width // 16
        n_layers = model.model.diffusion_model.layers.__len__()
        out_path = os.path.join(folder_paths.get_output_directory(), "moss", "probe", tag + ".npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        state = PROBES[tag] = {"calls": [], "cur": None}

        def lens(x, vec, dev):
            hw = {k: v.to(dev) for k, v in head.items()}
            mod = torch.nn.functional.linear(vec.float(), hw["video_head_modulation.weight"], hw["video_head_modulation.bias"])
            shift, scale = mod.unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * hw["final_norm_video.weight"] * (1 + scale) + shift
            return torch.nn.functional.linear(y, hw["final_linear_video.weight"])  # (B, HW, 128)

        def grid(t):  # (B, HW, C) -> (B, C, H, W)
            return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], H, W)

        def make(i):
            def wrap(args, extra):
                to = args["transformer_options"]
                x_in = args["img"][:, :H * W].float().clone()  # blocks update x in place
                out = extra["original_block"](args)
                x_out = out["img"][:, :H * W].float()
                l_in, l_out = lens(x_in, args["vec"], x_out.device), lens(x_out, args["vec"], x_out.device)
                row = torch.stack([column_spectrum(grid(l_out)), column_spectrum(grid(l_out - l_in)), column_spectrum(grid(x_out))], dim=1)  # (B, 3, nb)
                if i == 0 or state["cur"] is None:
                    state["cur"] = {"sigma": float(to["sigmas"].flatten()[0]), "cou": list(to.get("cond_or_uncond", [])),
                                    "spec": np.full((x_out.shape[0], n_layers, 3, W // 2 + 1), np.nan, np.float32)}
                state["cur"]["spec"][:, i] = row.cpu().numpy()
                if i == n_layers - 1:
                    state["calls"].append(state["cur"])
                    state["cur"] = None
                    np.savez(out_path, sigma=np.array([c["sigma"] for c in state["calls"]]),
                             cou=np.array([str(c["cou"]) for c in state["calls"]]),
                             spec=np.array([c["spec"] for c in state["calls"]], dtype=object), allow_pickle=True)
                return out
            return wrap

        m = model.clone()
        for i in range(n_layers):
            m.set_model_patch_replace(make(i), "dit", "single_block", i)
        return (m,)


NODE_CLASS_MAPPINGS["MOSSStripeProbe"] = MOSSStripeProbe
NODE_DISPLAY_NAME_MAPPINGS["MOSSStripeProbe"] = "MOSS Stripe Probe"


from safetensors import safe_open
from safetensors.torch import save_file, load_file

GRAMS = {}  # tag -> {"layers.N.proj.<name>": [G (fp32, on `device`), rows]}
PROJ = ("qkv", "o", "ffn_up_gate", "ffn_down")


class MOSSGramCapture:
    """Accumulates G = sum x x^T of the inputs to each block linear (qkv, o, ffn_up_gate, ffn_down) for the chosen layers,
    over every model call (all steps, cond and uncond). G lives on the model's GPU in fp32, keyed by tag."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "tag": ("STRING", {"default": "flat"}),
                             "layers": ("STRING", {"default": "21-41"}),
                             "reset": ("BOOLEAN", {"default": False}),
                             "image_centered": ("BOOLEAN", {"default": False}),
                             "sigma_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "capture"
    CATEGORY = "model/debug"

    def capture(self, model, tag, layers, reset, image_centered=False, sigma_max=1.0):
        """image_centered: image tokens only, each row's mean along W removed (drops the per-token DC / modulation / colour
        direction). sigma_max: only model calls at sigma <= sigma_max are accumulated."""
        if reset or tag not in GRAMS:
            GRAMS[tag] = {}
        store = GRAMS[tag]
        blocks = model.model.diffusion_model.layers
        grid = {}

        def shape_wrapper(executor, x, *args, **kwargs):
            grid["hw"] = x.shape[-2:]
            return executor(x, *args, **kwargs)

        def hook_for(key):
            def hook(module, args):
                x = args[0].detach()
                if image_centered:
                    H, W = grid["hw"]
                    x = x[:, :H * W].unflatten(1, (H, W)).float()
                    x = x - x.mean(dim=2, keepdim=True)
                x = x.reshape(-1, x.shape[-1]).float()
                if key not in store:
                    store[key] = [torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float32), 0]
                store[key][0].addmm_(x.T, x)
                store[key][1] += x.shape[0]
            return hook

        def make(i):
            def wrap(args, extra):
                if float(args["transformer_options"]["sigmas"].max()) > sigma_max:
                    return extra["original_block"](args)
                hs = [blocks[i].proj[name].register_forward_pre_hook(hook_for(f"layers.{i}.proj.{name}")) for name in PROJ]
                try:
                    return extra["original_block"](args)
                finally:
                    for h in hs:
                        h.remove()
            return wrap

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "moss_gram_grid", shape_wrapper)
        for i in sorted(parse_layers(layers)):
            m.set_model_patch_replace(make(i), "dit", "single_block", i)
        return (m,)


class MOSSGramSave:
    """Writes GRAMS[tag] to output/moss/grams/<tag>.safetensors (G normalised per row, plus row counts) and frees it when
    `save` is on. Takes the decoded images only to run after sampling."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), "tag": ("STRING", {"default": "flat"}), "save": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model/debug"

    def save(self, images, tag, save):
        store = GRAMS.get(tag, {})
        print(f"MOSSGramSave: {tag}: {len(store)} linears, rows/linear {next(iter(store.values()))[1] if store else 0}")
        if save and store:
            out = {}
            for k, (G, n) in store.items():
                out[k + ".G"] = (G / n).cpu().contiguous()
                out[k + ".rows"] = torch.tensor(n, dtype=torch.int64)
            path = os.path.join(folder_paths.get_output_directory(), "moss", "grams", tag + ".safetensors")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_file(out, path)
            del GRAMS[tag]
            print(f"MOSSGramSave: wrote {path}")
        return {}


class MOSSGramSolve:
    """Closed-form edit of SFT toward Base, per linear, from activation statistics:
        min ||(W' - W_base) X_flat||^2 + mu ||(W' - W_sft) X_gen||^2 + eps ||W' - W_sft||^2
        => D = W' - W_sft = Delta G_f (G_f + mu G_g + eps I)^-1,   Delta = W_base - W_sft,   eps = rho tr(G_f + mu G_g) / d
    G are per-row Gram matrices from MOSSGramSave. Applied to `model` as diff patches (scaled by strength).
    num_tag: Gram of the part of the flat activations to move toward Base (e.g. late-sigma, image-only, row-centred, captured on
    the same trajectories); D = Delta G_num (G_f + mu G_g + eps I)^-1 with G_num in G_f's row units: Base on that part, SFT elsewhere.
    num_complement: use G_f - G_num instead (Base on everything except that part)."""
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list("diffusion_models")
        return {"required": {"model": ("MODEL",),
                             "sft_file": (files,), "base_file": (files,),
                             "flat_tag": ("STRING", {"default": "flat"}), "gen_tag": ("STRING", {"default": "gen"}),
                             "layers": ("STRING", {"default": "21-41"}),
                             "parts": ("STRING", {"default": "qkv,o,ffn_up_gate,ffn_down"}),
                             "mu": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                             "rho": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 10.0, "step": 0.0001}),
                             "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                             "device": ("STRING", {"default": "cuda:0"}),
                             "num_tag": ("STRING", {"default": ""}),
                             "num_complement": ("BOOLEAN", {"default": False}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "solve"
    CATEGORY = "model/merging"

    def solve(self, model, sft_file, base_file, flat_tag, gen_tag, layers, parts, mu, rho, strength, device, num_tag="", num_complement=False):
        gdir = os.path.join(folder_paths.get_output_directory(), "moss", "grams")
        gf = safe_open(os.path.join(gdir, flat_tag + ".safetensors"), "pt", device=device)
        gn = safe_open(os.path.join(gdir, num_tag + ".safetensors"), "pt", device=device) if num_tag else None
        gg = safe_open(os.path.join(gdir, gen_tag + ".safetensors"), "pt", device=device) if mu > 0 else None
        fs = safe_open(folder_paths.get_full_path_or_raise("diffusion_models", sft_file), "pt", device=device)
        fb = safe_open(folder_paths.get_full_path_or_raise("diffusion_models", base_file), "pt", device=device)
        want = [p.strip() for p in parts.split(",") if p.strip()]
        patches, report = {}, []
        for i in sorted(parse_layers(layers)):
            for name in want:
                k = f"layers.{i}.proj.{name}"
                delta = fb.get_tensor(k + ".weight").double() - fs.get_tensor(k + ".weight").double()
                A = gf.get_tensor(k + ".G").double()
                if gn is not None:
                    rhs = gn.get_tensor(k + ".G").double() * (gn.get_tensor(k + ".rows").item() / gf.get_tensor(k + ".rows").item())
                    if num_complement:
                        rhs = A - rhs
                else:
                    rhs = A.clone()
                if gg is not None:
                    A = A + mu * gg.get_tensor(k + ".G").double()
                A.diagonal().add_(rho * A.diagonal().sum() / A.shape[0])
                L = torch.linalg.cholesky(A)
                D = torch.cholesky_solve(rhs @ delta.T, L).T  # A symmetric: D = Delta G_f A^-1
                report.append((k, float(D.norm() / delta.norm())))
                patches[f"diffusion_model.{k}.weight"] = ("diff", (D.to(torch.bfloat16).cpu(),))
        m = model.clone()
        m.add_patches(patches, strength)
        mean = sum(r for _, r in report) / max(len(report), 1)
        print(f"MOSSGramSolve: {len(patches)} linears, mu={mu} rho={rho} num={("not " if num_complement else "") + (num_tag or flat_tag)}, mean |D|/|Delta| = {mean:.3f}")
        for k, r in report:
            if k.endswith("qkv"):
                print(f"   {k.split('.proj')[0]}: " + " ".join(f"{n}={rr:.2f}" for kk, rr in report for n in [kk.rsplit('.', 1)[1]] if kk.startswith(k.rsplit('.', 1)[0] + '.')))
        return (m,)


NODE_CLASS_MAPPINGS.update({"MOSSGramCapture": MOSSGramCapture, "MOSSGramSave": MOSSGramSave, "MOSSGramSolve": MOSSGramSolve})
NODE_DISPLAY_NAME_MAPPINGS.update({"MOSSGramCapture": "MOSS Gram Capture", "MOSSGramSave": "MOSS Gram Save", "MOSSGramSolve": "MOSS Gram Solve"})


class MOSSRopeOverride:
    """Diagnostic: rebuild MOSS-Image RoPE positions with the image column/row coordinates scaled, the image translated
    relative to the text (image-only shift), or everything translated (common shift, a null control for relative RoPE)."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "col_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01}),
                             "row_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01}),
                             "image_shift": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.5}),
                             "common_shift": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.5}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/debug"

    def patch(self, model, col_scale, row_scale, image_shift, common_shift):
        dm = model.get_model_object("diffusion_model")

        def rope(h, w, text_len, device):
            ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
            ids[:h * w, 1] = text_len
            ids[:h * w, 2] = ((torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * row_scale).repeat_interleave(w)
            ids[:h * w, 3] = ((torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * col_scale + image_shift).repeat(h)
            ids[h * w:, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
            ids[:, 2:] += common_shift
            return dm.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

        m = model.clone()
        m.add_object_patch("diffusion_model.rope", rope)
        return (m,)


NODE_CLASS_MAPPINGS["MOSSRopeOverride"] = MOSSRopeOverride
NODE_DISPLAY_NAME_MAPPINGS["MOSSRopeOverride"] = "MOSS RoPE Override"

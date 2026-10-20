#!/usr/bin/env python3
"""Export a MOSSGramSolve edit (D = Delta G_f (G_f + mu G_g + eps I)^-1 per linear) as a ComfyUI full-difference LoRA (.diff keys).
Solved per linear on the GPU in float32 (the ridge keeps it well conditioned for rho >= ~0.01); only one linear's tensors are resident.

  comfyui_venv/bin/python workflows/moss/export_gramsolve.py <out_name> --mu 0 --rho 1.0 --layers 21-41
Writes models/loras/<out_name>.safetensors (bf16) and prints per-linear |D|/|Delta|.
"""
import argparse, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "moss_block_lab"))
ap = argparse.ArgumentParser()
ap.add_argument("out_name")
ap.add_argument("--mu", type=float, default=0.0)
ap.add_argument("--rho", type=float, default=1.0)
ap.add_argument("--layers", default="21-41")
ap.add_argument("--flat", default="flat")
ap.add_argument("--gen", default="gen")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()


def parse_layers(spec):
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


G = os.path.join(ROOT, "output", "moss", "grams")
gf = safe_open(os.path.join(G, a.flat + ".safetensors"), "pt")
gg = safe_open(os.path.join(G, a.gen + ".safetensors"), "pt") if a.mu > 0 else None
fs = safe_open(os.path.join(ROOT, "models", "diffusion_models", "moss_image_sft_bf16.safetensors"), "pt")
fb = safe_open(os.path.join(ROOT, "models", "diffusion_models", "moss_image_base_bf16.safetensors"), "pt")
out = {}
for i in parse_layers(a.layers):
    for name in ("qkv", "o", "ffn_up_gate", "ffn_down"):
        k = f"layers.{i}.proj.{name}"
        dev = a.device
        delta = fb.get_tensor(k + ".weight").to(dev, torch.float32) - fs.get_tensor(k + ".weight").to(dev, torch.float32)
        A = gf.get_tensor(k + ".G").to(dev)
        rhs = A.clone()
        if gg is not None:
            A = A + a.mu * gg.get_tensor(k + ".G").to(dev)
        A.diagonal().add_(a.rho * A.diagonal().sum() / A.shape[0])
        D = torch.cholesky_solve(rhs @ delta.T, torch.linalg.cholesky(A)).T
        out[f"diffusion_model.{k}.diff"] = D.to(torch.bfloat16).cpu().contiguous()
        print(f"{k}: |D|/|Delta| = {float(D.norm() / delta.norm()):.3f}", flush=True)
path = os.path.join(ROOT, "models", "loras", a.out_name + ".safetensors")
save_file(out, path, metadata={"mu": str(a.mu), "rho": str(a.rho), "layers": a.layers, "grams": f"{a.flat},{a.gen}",
                               "recipe": "D = (W_base - W_sft) G_f (G_f + mu G_g + rho tr(.)/d I)^-1"})
print("wrote", path, round(os.path.getsize(path) / 1e9, 2), "GB")

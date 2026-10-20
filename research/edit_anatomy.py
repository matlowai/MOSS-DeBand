#!/usr/bin/env python3
"""Where each DeBand fix edits SFT, and how much it changes each layer's output on solid-colour vs general activations.

  comfyui_venv/bin/python workflows/moss/edit_anatomy.py   (CPU; writes output/moss/edit_anatomy.json)
Per layer and linear (qkv, o, ffn_up_gate, ffn_down), with Delta = W_base - W_sft and D a fix's weight change:
  w2   = |D|^2            (weight-change energy; Delta's own is the full revert)
  dot  = <D, Delta>       (how much of D points toward Base)
  ef   = tr(D G_f D^T)    (output-change energy on the solid-colour activations, flat Gram, layers 21-41 only)
  eg   = tr(D G_g D^T)    (same on the general-prompt activations, gen Gram)
fp32. Gram solves for any rho come from one eigendecomposition of G_f: D = Delta U diag(l / (l + eps)) U^T, eps = rho tr(G_f) / d.
"""
import json, os
import torch
from safetensors import safe_open

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "output", "moss", "edit_anatomy.json")
PROJ = ("qkv", "o", "ffn_up_gate", "ffn_down")
RHOS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
torch.set_num_threads(16)

sft = safe_open(os.path.join(ROOT, "models/diffusion_models/moss_image_sft_bf16.safetensors"), "pt")
base = safe_open(os.path.join(ROOT, "models/diffusion_models/moss_image_base_bf16.safetensors"), "pt")
gf = safe_open(os.path.join(ROOT, "output/moss/grams/flat.safetensors"), "pt")
gg = safe_open(os.path.join(ROOT, "output/moss/grams/gen.safetensors"), "pt")
kij = safe_open(os.path.join(ROOT, "models/loras/moss_image_sft_lora_blocks_21_to_49_rank_512_bf16.safetensors"), "pt")
kij_keys = set(kij.keys())


def explicit(D, delta, Gf, Gg, DtD=None):
    r = {"w2": float(D.pow(2).sum()), "dot": float((D * delta).sum())}
    if Gf is not None:
        DtD = D.T @ D if DtD is None else DtD  # tr(D G D^T) = <G, D^T D>
        r["ef"] = float((DtD * Gf).sum())
        r["eg"] = float((DtD * Gg).sum())
    return r


rows = []
for i in range(42):
    for name in PROJ:
        k = f"layers.{i}.proj.{name}"
        delta = base.get_tensor(k + ".weight").float() - sft.get_tensor(k + ".weight").float()
        row = {"layer": i, "proj": name}
        Gf = Gg = None
        if 21 <= i <= 41:
            Gf, Gg = gf.get_tensor(k + ".G").float(), gg.get_tensor(k + ".G").float()
        DtD = delta.T @ delta if Gf is not None else None
        row["revert"] = explicit(delta, delta, Gf, Gg, DtD)
        lk = f"diffusion_model.{k}.lora_up.weight"
        if lk in kij_keys:  # Kijai's LoRA at strength -1 (no alpha: scale 1)
            up, down = kij.get_tensor(lk).float(), kij.get_tensor(f"diffusion_model.{k}.lora_down.weight").float()
            D = -(up @ down)
            row["kijai_m1"] = explicit(D, delta, Gf, Gg, None if Gf is None else down.T @ (up.T @ up) @ down)
        if Gf is not None:
            lam, U = torch.linalg.eigh(Gf)
            lam = lam.clamp(min=0)
            P = U.T @ DtD @ U
            M = U.T @ Gg @ U
            q = P.diagonal()
            d = lam.numel()
            for rho in RHOS:
                f = lam / (lam + rho * lam.sum() / d)
                row[f"solve_rho{rho:g}"] = {"w2": float((f * f * q).sum()), "dot": float((f * q).sum()), "ef": float((f * f * lam * q).sum()),
                                            "eg": float(((f[:, None] * M * f[None, :]) * P.T).sum())}
        rows.append(row)
        print(k, {t: round(v["w2"] / row["revert"]["w2"], 3) for t, v in row.items() if isinstance(v, dict)}, flush=True)
    json.dump(rows, open(OUT, "w"))
print("wrote", OUT)

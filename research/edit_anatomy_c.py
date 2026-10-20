#!/usr/bin/env python3
"""Second pass for edit_anatomy.json: output-change energy on horizontal detail only (row-centred image activations, the
flatc / genc Grams), which leaves out the per-token colour/brightness/modulation direction that dominates the plain Grams.

  comfyui_venv/bin/python workflows/moss/edit_anatomy_c.py   (CPU, after edit_anatomy.py)
Adds efc / egc to revert, kijai_m1 and the two saved Gram-solve checkpoints (ckpt_rho0.1, ckpt_rho1) for layers 21-41.
"""
import json, os
import torch
from safetensors import safe_open

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PATH = os.path.join(ROOT, "output", "moss", "edit_anatomy.json")
torch.set_num_threads(16)

sft = safe_open(os.path.join(ROOT, "models/diffusion_models/moss_image_sft_bf16.safetensors"), "pt")
base = safe_open(os.path.join(ROOT, "models/diffusion_models/moss_image_base_bf16.safetensors"), "pt")
gfc = safe_open(os.path.join(ROOT, "output/moss/grams/flatc.safetensors"), "pt")
ggc = safe_open(os.path.join(ROOT, "output/moss/grams/genc.safetensors"), "pt")
kij = safe_open(os.path.join(ROOT, "models/loras/moss_image_sft_lora_blocks_21_to_49_rank_512_bf16.safetensors"), "pt")
ck = {t: safe_open(os.path.join(ROOT, "models/loras", f"moss_lab_gramsolve_mu0_{t[5:]}_L21-41.safetensors"), "pt") for t in ("ckpt_rho0.1", "ckpt_rho1")}

rows = json.load(open(PATH))
for row in rows:
    if not 21 <= row["layer"] <= 41:
        continue
    k = f"layers.{row['layer']}.proj.{row['proj']}"
    Gf, Gg = gfc.get_tensor(k + ".G").float(), ggc.get_tensor(k + ".G").float()
    delta = base.get_tensor(k + ".weight").float() - sft.get_tensor(k + ".weight").float()
    up, down = kij.get_tensor(f"diffusion_model.{k}.lora_up.weight").float(), kij.get_tensor(f"diffusion_model.{k}.lora_down.weight").float()
    edits = {"revert": (delta, delta.T @ delta), "kijai_m1": (-(up @ down), down.T @ (up.T @ up) @ down)}
    for t, h in ck.items():
        D = h.get_tensor(f"diffusion_model.{k}.diff").float()
        edits[t] = (D, D.T @ D)
    for t, (D, DtD) in edits.items():
        row.setdefault(t, {"w2": float(D.pow(2).sum()), "dot": float((D * delta).sum())})
        row[t]["efc"] = float((DtD * Gf).sum())
        row[t]["egc"] = float((DtD * Gg).sum())
    print(k, {t: round((row[t]["efc"] / row["revert"]["efc"]) ** 0.5, 3) for t in edits}, flush=True)
json.dump(rows, open(PATH, "w"))
print("updated", PATH)

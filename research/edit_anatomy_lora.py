#!/usr/bin/env python3
"""Add any MOSS-Image LoRA as an arm of output/moss/edit_anatomy.json (after edit_anatomy.py and edit_anatomy_c.py).

  comfyui_venv/bin/python workflows/moss/edit_anatomy_lora.py <tag> <file in models/loras>
Reads lora_up/lora_down(/alpha) or .diff keys; split q/k/v LoRA keys are placed into the fused qkv rows (q, k, v blocks of
3072 rows each), as ComfyUI does when it applies them. Per matrix the weight change at strength 1 gets w2 = |D|^2, dot = <D, Delta>,
and for layers 21-41 ef/eg (raw flat/gen Grams) and efc/egc (row-centred flatc/genc Grams); untouched matrices get no entry.
"""
import json, os, sys
import torch
from safetensors import safe_open

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PATH = os.path.join(ROOT, "output", "moss", "edit_anatomy.json")
tag, lora = sys.argv[1], sys.argv[2]
torch.set_num_threads(16)

h = safe_open(os.path.join(ROOT, "models", "loras", lora), "pt")
keys = set(h.keys())
sft = safe_open(os.path.join(ROOT, "models/diffusion_models/moss_image_sft_bf16.safetensors"), "pt")
base = safe_open(os.path.join(ROOT, "models/diffusion_models/moss_image_base_bf16.safetensors"), "pt")
grams = {n: safe_open(os.path.join(ROOT, "output/moss/grams", n + ".safetensors"), "pt") for n in ("flat", "gen", "flatc", "genc")}


def piece(name):
    """Weight change of one LoRA target (e.g. layers.33.proj.k) at strength 1, or None."""
    k = f"diffusion_model.{name}"
    if k + ".diff" in keys:
        return h.get_tensor(k + ".diff").float()
    if k + ".lora_up.weight" not in keys:
        return None
    up, down = h.get_tensor(k + ".lora_up.weight").float(), h.get_tensor(k + ".lora_down.weight").float()
    alpha = float(h.get_tensor(k + ".alpha")) if k + ".alpha" in keys else up.shape[1]
    return (up @ down) * (alpha / up.shape[1])


rows = json.load(open(PATH))
n = 0
for row in rows:
    k = f"layers.{row['layer']}.proj.{row['proj']}"
    D = piece(k)
    if row["proj"] == "qkv" and D is None:
        parts = [piece(f"layers.{row['layer']}.proj.{p}") for p in ("q", "k", "v")]
        if any(p is not None for p in parts):
            D = torch.cat([p if p is not None else torch.zeros(3072, 3072) for p in parts])
    if D is None:
        row.pop(tag, None)
        continue
    delta = base.get_tensor(k + ".weight").float() - sft.get_tensor(k + ".weight").float()
    e = {"w2": float(D.pow(2).sum()), "dot": float((D * delta).sum())}
    if 21 <= row["layer"] <= 41:
        DtD = D.T @ D
        for g, key in (("flat", "ef"), ("gen", "eg"), ("flatc", "efc"), ("genc", "egc")):
            e[key] = float((DtD * grams[g].get_tensor(k + ".G").float()).sum())
    row[tag] = e
    n += 1
    print(k, {x: round(v, 4) for x, v in e.items()}, flush=True)
json.dump(rows, open(PATH, "w"))
print(f"{tag}: {n} matrices added to {PATH}")

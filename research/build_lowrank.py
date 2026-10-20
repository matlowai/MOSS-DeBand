#!/usr/bin/env python3
"""Truncated-SVD versions of a full-rank .diff banding-fix LoRA (ComfyUI lora_up/lora_down + alpha; norm diffs kept as-is).

  CUDA_VISIBLE_DEVICES=0 comfyui_venv/bin/python workflows/moss/build_lowrank.py <src.safetensors> <rank> [<rank> ...]
Writes <src stem>_r<rank>.safetensors next to the source, in bf16.
"""
import os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, ranks = sys.argv[1], [int(r) for r in sys.argv[2:]]
with safe_open(src, "pt") as f:
    diffs = {k: f.get_tensor(k) for k in f.keys()}
svd = {}
for k, d in diffs.items():
    if d.dim() == 2:
        U, S, Vh = torch.linalg.svd(d.cuda().float(), full_matrices=False)
        svd[k] = (U, S, Vh)
for r in ranks:
    out = {}
    for k, d in diffs.items():
        base = k[:-len(".diff")]
        if d.dim() != 2:
            out[k] = d
            continue
        U, S, Vh = svd[k]
        root = S[:r].sqrt()
        out[base + ".lora_up.weight"] = (U[:, :r] * root).to(torch.bfloat16).cpu().contiguous()
        out[base + ".lora_down.weight"] = (root[:, None] * Vh[:r]).to(torch.bfloat16).cpu().contiguous()
        out[base + ".alpha"] = torch.tensor(float(r))
    path = os.path.splitext(src)[0] + f"_r{r}.safetensors"
    save_file(out, path)
    print(path, round(os.path.getsize(path) / 1e6, 1), "MB")

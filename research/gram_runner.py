#!/usr/bin/env python3
"""Re-queue a Gram capture set (the API prompts embedded in output/moss/gram_capture/<src>_*.png) under a new tag.

  python3 workflows/moss/gram_runner.py <src> <tag> [--image-centered] [--sigma-max 0.75] [--front] [--lora file:strength]
      [--solve mu,rho,layers,strength,flat_tag,num_tag,gen_tag] [--url ...]
The last job saves the Gram to output/moss/grams/<tag>.safetensors. --solve/--lora capture on SFT + that repair + that LoRA.
"""
import argparse, glob, json, os, urllib.request
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "..", "..", "output", "moss", "gram_capture")

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("tag")
ap.add_argument("--image-centered", action="store_true")
ap.add_argument("--sigma-max", type=float, default=1.0)
ap.add_argument("--front", action="store_true")
ap.add_argument("--lora")
ap.add_argument("--solve")
ap.add_argument("--url", default="http://localhost:8188")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(CAP, f"{a.src}_[0-9][0-9][0-9]_*_00001_.png")))
for n, f in enumerate(files):
    p = json.loads(Image.open(f).info["prompt"])
    p["90"]["inputs"].update(tag=a.tag, reset=n == 0, image_centered=a.image_centered, sigma_max=a.sigma_max)
    p["91"]["inputs"].update(tag=a.tag, save=n == len(files) - 1)
    if a.solve:
        v = a.solve.split(",")
        p["190"] = {"class_type": "MOSSGramSolve", "inputs": {
            "model": ["3", 0], "sft_file": "moss_image_sft_bf16.safetensors", "base_file": "moss_image_base_bf16.safetensors",
            "flat_tag": v[4] or "flat", "gen_tag": v[6] or "gen", "layers": v[2], "parts": "qkv,o,ffn_up_gate,ffn_down",
            "mu": float(v[0]), "rho": float(v[1]), "strength": float(v[3]), "device": "cuda:0",
            "num_tag": v[5].lstrip("!"), "num_complement": v[5].startswith("!")}}
        p["90"]["inputs"]["model"] = ["190", 0]
    if a.lora:
        f_, s_ = a.lora.rsplit(":", 1)
        p["14"]["inputs"].update(lora_name=f_, strength_model=float(s_))
    p["26"]["inputs"]["filename_prefix"] = p["26"]["inputs"]["filename_prefix"].replace(f"/{a.src}_", f"/{a.tag}_")
    r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/prompt", json.dumps({"prompt": p, "front": a.front}).encode(), {"Content-Type": "application/json"})))
    assert not r.get("node_errors"), r
print(f"{a.tag}: queued {len(files)} from {a.src}")

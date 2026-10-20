#!/usr/bin/env python3
"""Render training targets from a Gram-capture prompt set (the API prompts embedded in output/moss/gram_capture/<src>_*.png).

  python3 workflows/moss/trainset_runner.py <src> <name> [--solve mu,rho,layers,strength,flat_tag,num_tag,gen_tag] [--seed-offsets 0,2000] [--url ...]
Renders go to output/moss/trainset/<name>/<src>_<idx>_<WxH>_s<seed>_00001_.png. Without --solve the plain model renders.
"""
import argparse, glob, json, os, re, urllib.request
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "..", "..", "output", "moss", "gram_capture")

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("name")
ap.add_argument("--solve")
ap.add_argument("--seed-offsets", default="0")
ap.add_argument("--url", default="http://localhost:8188")
a = ap.parse_args()

n = 0
for f in sorted(glob.glob(os.path.join(CAP, f"{a.src}_[0-9][0-9][0-9]_*_00001_.png"))):
    base = json.loads(Image.open(f).info["prompt"])
    for k in ("90", "91"):  # drop the Gram capture/save nodes
        base.pop(k, None)
    base["14"]["inputs"]["model"] = ["3", 0]
    stem = re.match(r"(.+)_00001_\.png", os.path.basename(f)).group(1)
    for off in (int(o) for o in a.seed_offsets.split(",")):
        p = json.loads(json.dumps(base))
        seed = p["22"]["inputs"]["value"] + off
        p["22"]["inputs"]["value"] = seed
        if a.solve:
            v = a.solve.split(",")
            p["90"] = {"class_type": "MOSSGramSolve", "inputs": {
                "model": ["3", 0], "sft_file": "moss_image_sft_bf16.safetensors", "base_file": "moss_image_base_bf16.safetensors",
                "flat_tag": v[4] or "flat", "gen_tag": v[6] or "gen", "layers": v[2], "parts": "qkv,o,ffn_up_gate,ffn_down",
                "mu": float(v[0]), "rho": float(v[1]), "strength": float(v[3]), "device": "cuda:0",
                "num_tag": v[5].lstrip("!"), "num_complement": v[5].startswith("!")}}
            p["14"]["inputs"]["model"] = ["90", 0]
        p["26"]["inputs"]["filename_prefix"] = f"moss/trainset/{a.name}/{stem}_s{seed}"
        r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})))
        assert not r.get("node_errors"), r
        n += 1
print(f"{a.name}: queued {n}")

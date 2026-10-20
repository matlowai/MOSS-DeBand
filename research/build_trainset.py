#!/usr/bin/env python3
"""Assemble a TrainLoraNode dataset folder in ComfyUI's input/ from rendered targets.

  comfyui_venv/bin/python workflows/moss/build_trainset.py <name> --fix output/moss/trainset/fix_gc03 [--min-db 6] [--preserve gen]
fix: renders whose HF chroma power (bandviz.hf_chroma_power) is at least --min-db below the median of the plain-SFT capture renders
of the same prompt (output/moss/gram_capture/flat_*); the absolute level depends on the colour, so the reference is per prompt.
preserve: plain SFT renders of a Gram-capture prompt set (output/moss/gram_capture/<preserve>_*.png), kept as-is.
Captions are the prompts embedded in each PNG. Writes input/<name>/ and input/<name>/manifest.json.
"""
import argparse, glob, json, os, shutil, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
import bandviz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--fix", required=True)
ap.add_argument("--min-db", type=float, default=6.0)
ap.add_argument("--preserve", default="gen")
a = ap.parse_args()

out = os.path.join(ROOT, "input", a.name)
os.makedirs(out, exist_ok=True)
manifest = {"fix": [], "rejected": [], "preserve": []}


def add(path, stem):
    shutil.copy(path, os.path.join(out, stem + ".png"))
    with open(os.path.join(out, stem + ".txt"), "w") as fh:
        fh.write(prompt(path))


def prompt(path):
    return json.loads(Image.open(path).info["prompt"])["6"]["inputs"]["value"]


ref = {}
for f in glob.glob(os.path.join(ROOT, "output", "moss", "gram_capture", "flat_[0-9][0-9][0-9]_*_00001_.png")):
    ref.setdefault(prompt(f), []).append(10 * np.log10(bandviz.hf_chroma_power(f)))
for f in sorted(glob.glob(os.path.join(ROOT, a.fix, "*_00001_.png"))):
    db = 10 * np.log10(bandviz.hf_chroma_power(f)) - float(np.median(ref[prompt(f)]))
    stem = "fix_" + os.path.basename(f).replace("_00001_.png", "")
    if db <= -a.min_db:
        add(f, stem)
        manifest["fix"].append([stem, round(db, 1)])
    else:
        manifest["rejected"].append([stem, round(db, 1)])
for f in sorted(glob.glob(os.path.join(ROOT, "output", "moss", "gram_capture", f"{a.preserve}_[0-9][0-9][0-9]_*_00001_.png"))):
    stem = "keep_" + os.path.basename(f).replace("_00001_.png", "")
    add(f, stem)
    manifest["preserve"].append(stem)
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(f"{a.name}: {len(manifest['fix'])} fix (rejected {len(manifest['rejected'])}), {len(manifest['preserve'])} preserve")

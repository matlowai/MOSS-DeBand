#!/usr/bin/env python3
"""Score DeBand arms against a reference arm (paired by prompt id and seed) from output/moss/drift/<arm>/.

  comfyui_venv/bin/python workflows/moss/drift_score.py <ref_arm> <arm> [<arm> ...]
Per image: banding (bandviz.local_score), SSIM to ref, DINOv2-base cosine distance to ref, and for photo people the ArcFace
cosine of the largest face (buffalo_l; a missing face is reported as missing, not as 0). Prints per-category mean and the worst
prompt, and writes output/moss/drift/scores.jsonl (one row per arm/image/metric).
"""
import glob, json, os, re, sys
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import AutoImageProcessor, AutoModel
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bandviz  # noqa: E402

D = os.path.join(HERE, "..", "..", "output", "moss", "drift")
CATS = {d["id"]: d["category"] for d in json.load(open(os.path.join(HERE, "bench", "drift_suite_v1.json")))}
ref, arms = sys.argv[1], sys.argv[2:]

proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face.prepare(ctx_id=-1, det_size=(640, 640))


def files(arm):
    out = {}
    for f in glob.glob(os.path.join(D, arm, "*_00001_.png")):
        m = re.match(r"(.+)__s(\d+)_00001_\.png", os.path.basename(f))
        out[(m.group(1), int(m.group(2)))] = f
    return out


def emb_dino(path):
    with torch.inference_mode():
        x = proc(images=Image.open(path).convert("RGB"), return_tensors="pt")
        v = dino(**x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1)[0]


def emb_face(path):
    faces = face.get(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return f.normed_embedding


def category(pid):
    return "bench" if pid.startswith("bench_") else "solid" if pid.startswith("solid_") else CATS.get(pid, "other")


R = files(ref)
cache = {}
rows = []
for arm in arms:
    A = files(arm)
    for key in sorted(set(R) & set(A)):
        pid, seed = key
        fr, fa = R[key], A[key]
        cat = category(pid)
        row = dict(arm=arm, id=pid, seed=seed, category=cat, band=bandviz.local_score(fa), band_ref=bandviz.local_score(fr))
        ar, aa = np.asarray(Image.open(fr).convert("RGB")), np.asarray(Image.open(fa).convert("RGB"))
        row["ssim"] = float(ssim(ar, aa, channel_axis=2, data_range=255))
        if cat == "solid":
            row["hf_db"] = 10 * np.log10(max(bandviz.hf_chroma_power(fa), 1.0) / bandviz.hf_chroma_power(fr))  # a perfectly uniform render has zero power
        if cat not in ("solid",):
            if fr not in cache:
                cache[fr] = emb_dino(fr)
            row["dino_dist"] = float(1 - (cache[fr] @ emb_dino(fa)))
        if cat == "people":
            er, ea = emb_face(fr), emb_face(fa)
            row["arcface"] = None if er is None or ea is None else float(np.dot(er, ea))
        rows.append(row)
with open(os.path.join(D, "scores.jsonl"), "a") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

for arm in arms:
    rs = [r for r in rows if r["arm"] == arm]
    print(f"\n=== {arm} vs {ref}  ({len(rs)} paired images)")
    for cat in ("solid", "bench", "people", "anime_people", "animals", "typography", "objects", "scenes"):
        cr = [r for r in rs if r["category"] == cat]
        if not cr:
            continue
        line = f"  {cat:13s} n={len(cr):2d}  banding {np.mean([r['band'] for r in cr]):.2f} (ref {np.mean([r['band_ref'] for r in cr]):.2f})  SSIM {np.mean([r['ssim'] for r in cr]):.3f}"
        if cat == "solid":
            line += f"  HF colour power vs SFT {np.mean([r['hf_db'] for r in cr]):+.1f} dB (worst {max(r['hf_db'] for r in cr):+.1f})"
        if "dino_dist" in cr[0]:
            worst = max(cr, key=lambda r: r["dino_dist"])
            line += f"  DINO dist {np.mean([r['dino_dist'] for r in cr]):.3f} (worst {worst['dino_dist']:.3f} {worst['id']})"
        if cat == "people":
            af = [r["arcface"] for r in cr if r["arcface"] is not None]
            line += f"  ArcFace {np.mean(af):.3f} (min {min(af):.3f}, missing {len(cr) - len(af)})" if af else "  ArcFace n/a"
        print(line)

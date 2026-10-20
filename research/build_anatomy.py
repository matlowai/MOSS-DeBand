#!/usr/bin/env python3
"""Build output/moss/edit_anatomy.html from output/moss/edit_anatomy.json (edit_anatomy.py) and drift/scores.jsonl.

  python3 workflows/moss/build_anatomy.py
"""
import json, math, os, statistics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "output", "moss")
rows = json.load(open(os.path.join(OUT, "edit_anatomy.json")))

# fixed series slots (colour follows the fix on every chart)
ARMS = [("solve_rho0.1", "Gram solve ρ=0.1", 1, "solve_mu0_rho0.1_L21-41"),
        ("solve_rho1", "Gram solve ρ=1", 2, "solve_mu0_rho1_L21-41"),
        ("kijai_m1", "Kijai LoRA @ −1", 3, "kijai_m1"),
        ("strong", "DeBand Strong @1.0", 7, "strong_full"),
        ("nynxz", "nynxz Destripe @1.0", 8, "nynxz_1.0"),
        ("revert21", "Exact revert 21–41", 4, "solve_mu0_L21-41"),
        ("revert32", "Quality+ (revert 32–41)", 5, "qualityplus_lora"),
        ("revert37", "DeBand (revert 37–41)", 6, "deband_lora")]
RANGES = {"revert21": (21, 41), "revert32": (32, 41), "revert37": (37, 41)}
PROJ = ("qkv", "o", "ffn_up_gate", "ffn_down")


CKPT = {"solve_rho0.1": "ckpt_rho0.1", "solve_rho1": "ckpt_rho1"}  # saved checkpoints carry the centred (efc/egc) numbers


def edit(r, arm, centred=False):
    """The arm's numbers for one linear, or None where it does not edit."""
    scaled = {"kijai_m05": ("kijai_m1", 0.5), "nynxz_1.5": ("nynxz", 1.5)}  # LoRA strength s: energies x s^2, alignment x s
    if arm in scaled:
        src, f = scaled[arm]
        k = r.get(src)
        return k and {key: v * (f if key == "dot" else f * f) for key, v in k.items()}
    if centred and arm in CKPT:
        return r.get(CKPT[arm])
    if arm in RANGES:
        lo, hi = RANGES[arm]
        return r["revert"] if lo <= r["layer"] <= hi else None
    return r.get(arm)


def ratio(a, b):
    return math.sqrt(a / b) if b > 0 else 0.0


# rendered results for tooltips
scores = {}
for line in open(os.path.join(OUT, "drift", "scores.jsonl")):
    s = json.loads(line)
    scores[(s["arm"], s["id"], s["seed"])] = s


def rendered(arm):
    rs = [s for (a, _, _), s in scores.items() if a == arm]
    if not rs:
        return None
    hf = [s["hf_db"] for s in rs if s.get("hf_db") is not None]
    af = [s["arcface"] for s in rs if s.get("arcface") is not None]
    return {"hf": statistics.fmean(hf) if hf else None, "arc": statistics.fmean(af) if af else None}


layers = sorted({r["layer"] for r in rows})
by = {(r["layer"], r["proj"]): r for r in rows}

# chart 1: weight-change size per layer (sqrt of energy over the 4 linears, relative to a full revert of that layer)
size = {}
for key, _, _, _ in ARMS:
    size[key] = []
    for L in layers:
        rs = [by[(L, p)] for p in PROJ]
        num = sum(edit(r, key)["w2"] for r in rs if edit(r, key))
        size[key].append(ratio(num, sum(r["revert"]["w2"] for r in rs)))
per_matrix = {key: {f"{L}": {p: (round(ratio(edit(by[(L, p)], key)["w2"], by[(L, p)]["revert"]["w2"]), 3) if edit(by[(L, p)], key) else 0)
                             for p in PROJ} for L in layers} for key, _, _, _ in ARMS}

# chart 2: output-change size on solid-colour vs general activations, layers 21-41 summed
glayers = [L for L in layers if "ef" in by[(L, "qkv")]["revert"]]
grows = [by[(L, p)] for L in glayers for p in PROJ]
ef0, eg0 = sum(r["revert"]["ef"] for r in grows), sum(r["revert"]["eg"] for r in grows)


centred = all("efc" in r["revert"] for r in grows)
efc0, egc0 = (sum(r["revert"]["efc"] for r in grows), sum(r["revert"]["egc"] for r in grows)) if centred else (0, 0)


def point(key):
    es = [edit(r, key) for r in grows]
    p = {"x": ratio(sum(e["eg"] for e in es if e), eg0), "y": ratio(sum(e["ef"] for e in es if e), ef0)}
    es = [edit(r, key, True) for r in grows]
    if centred and all(e is None or "efc" in e for e in es):
        p.update(xc=ratio(sum(e["egc"] for e in es if e), egc0), yc=ratio(sum(e["efc"] for e in es if e), efc0))
    return p


rhos = sorted({float(k[len("solve_rho"):]) for k in grows[0] if k.startswith("solve_rho")})
sweep = []
for rho in rhos:
    p = point(f"solve_rho{rho:g}")
    p.update(rho=rho, w=ratio(sum(r[f"solve_rho{rho:g}"]["w2"] for r in grows), sum(r["revert"]["w2"] for r in grows)),
             rendered=rendered(f"solve_mu0_rho{rho:g}_L21-41"))
    sweep.append(p)
points = []
for key, label, slot, arm in ARMS:
    p = point(key)
    p.update(key=key, label=label, slot=slot, rendered=rendered(arm),
             w=ratio(sum(edit(r, key)["w2"] for r in grows if edit(r, key)), sum(r["revert"]["w2"] for r in grows)))
    points.append(p)
plain = rendered("swap055_L21-41")

# chart 3: per-layer output-change size on solids vs general prompts
sel = {}
for key in ("strong", "solve_rho1", "kijai_m1", "nynxz"):
    sel[key] = {"solid": [], "gen": []}
    for L in glayers:
        rs = [by[(L, p)] for p in PROJ]
        es = [edit(r, key) for r in rs]
        f, g = ("efc", "egc") if centred else ("ef", "eg")
        es = [edit(r, key, centred) for r in rs]
        sel[key]["solid"].append(ratio(sum(e[f] for e in es if e), sum(r["revert"][f] for r in rs)))
        sel[key]["gen"].append(ratio(sum(e[g] for e in es if e), sum(r["revert"][g] for r in rs)))

# outcome benchmarks: rendered results (scores.jsonl) against weight-change share of the whole Base-SFT change (all 42 layers)
total = sum(r["revert"]["w2"] for r in rows)


def share(key, scale=1.0):
    if key == "base":
        return 1.0
    return scale * scale * sum(e["w2"] for r in rows for e in [edit(r, key)] if e) / total


def outcome(arm):
    rs = [s for (a, _, _), s in scores.items() if a == arm]
    hf = [s["hf_db"] for s in rs if s.get("hf_db") is not None]
    af = [s["arcface"] for s in rs if s.get("arcface") is not None]
    dd = [s["dino_dist"] for s in rs if s.get("dino_dist") is not None]
    return {"clean": -statistics.median(hf), "clean_mean": -statistics.fmean(hf), "worst": -max(hf), "face": statistics.fmean(af),
            "dino": statistics.fmean(dd), "n": len(hf)}


OUTCOMES = []
for label, slot, arm, key, scale in [("nynxz Destripe @1.0", 8, "nynxz_1.0", "nynxz", 1.0), ("nynxz Destripe @1.5", 8, "nynxz_1.5", "nynxz", 1.5),
                                     ("DeBand Strong @1.0", 7, "strong_full", "strong", 1.0), ("DeBand Strong @1.2", 7, "gcsolve_mu0.3_rho0.05_x1.2_L21-41", "strong", 1.2),
                                     ("Gram solve ρ=0.1", 1, "solve_mu0_rho0.1_L21-41", "solve_rho0.1", 1.0), ("Gram solve ρ=1", 2, "solve_mu0_rho1_L21-41", "solve_rho1", 1.0),
                                     ("Kijai LoRA @ −1", 3, "kijai_m1", "kijai_m1", 1.0), ("Kijai LoRA @ −0.5", 3, "kijai_m05", "kijai_m1", 0.5),
                                     ("Exact revert 21–41", 4, "solve_mu0_L21-41", "revert21", 1.0), ("Quality+ (revert 32–41)", 5, "qualityplus_lora", "revert32", 1.0),
                                     ("DeBand (revert 37–41)", 6, "deband_lora", "revert37", 1.0), ("Base model", 0, "base", "base", 1.0)]:
    o = outcome(arm)
    o.update(label=label, slot=slot, share=share(key, scale))
    OUTCOMES.append(o)

data = {"arms": [[k, l, s] for k, l, s, _ in ARMS], "outcomes": OUTCOMES, "layers": layers, "size": size, "per_matrix": per_matrix,
        "sweep": sweep, "points": points, "plain": plain, "glayers": glayers, "sel": sel, "centred": centred}
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "anatomy_template.html")).read()
open(os.path.join(OUT, "edit_anatomy.html"), "w").write(html.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
print("edit_anatomy.html written")

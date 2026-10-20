#!/usr/bin/env python3
"""Rebuild output/moss/compare_solve.html from output/moss/drift/scores.jsonl (latest row per arm/item/seed).

  python3 workflows/moss/build_compare.py [--public <dir> --arms a,b,c]
Arms listed in ARMS get a label (and an optional tag); arms that are scored but not listed appear under their folder name.
--public writes a self-contained copy for hosting: <dir>/index.html with only the given arms and WebP copies of their renders
(<dir>/img/<arm>/<item>__s<seed>.webp, quality 90; the colour view on lossy WebP is close to, not identical to, the PNGs).
"""
import argparse, datetime, glob, json, os, statistics
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "output", "moss")

# (arm, label, tag) in menu order; the first is the default middle column
ARMS = [
    ("solve_mu0_rho0.1_L21-41", "Gram solve ρ=0.1, 21–41 (no detail protection)", ""),
    ("win_rho0.1_s13-25", "Gram solve ρ=0.1, steps 13–25 only", "new"),
    ("gcsolve_mu0.3_rho0.1_L21-41", "ρ=0.1 solve + protect general-prompt detail μ=0.3, 21–41", "new"),
    ("strong_full", "DeBand Strong @1.0 (full-rank LoRA, μ0.3 ρ0.05 solve)", "release"),
    ("gcsolve_mu0.3_rho0.05_x1.2_L21-41", "DeBand Strong @1.2 (same solve ×1.2)", "release"),
    ("strong_r512", "DeBand Strong compressed to rank 512", "compression test"),
    ("strong_r256", "DeBand Strong compressed to rank 256", "compression test"),
    ("strong_r128", "DeBand Strong compressed to rank 128", "compression test"),
    ("strong_r64", "DeBand Strong compressed to rank 64", "compression test"),
    ("sft_cfg25", "SFT at CFG 2.5, no SLG/TSR (reference for the *_cfg25 arms)", "reference"),
    ("nynxz_1.5_cfg25", "nynxz @1.5 at CFG 2.5, no SLG/TSR", "matched"),
    ("gcsolve_mu0.3_rho0.05_cfg25", "Our μ0.3 ρ0.05 solve at CFG 2.5, no SLG/TSR", "matched"),
    ("nynxz_1.0", "nynxz Destripe LoRA @1.0 (64 gradient-picked directions, 1.9 MB)", "external"),
    ("nynxz_1.5", "nynxz Destripe LoRA @1.5 (their 1024 px setting)", "external"),
    ("nynxz_1.5_gc03r005", "nynxz Destripe @1.5 stacked on μ0.3 ρ0.05 solve", "new"),
    ("dbA", "Trained de-band LoRA, 600 steps (arm A)", "overnight"),
    ("dbA_stale", "Arm A LoRA + μ0.3 repair fitted on plain SFT", "overnight"),
    ("dbB", "Arm A LoRA + μ0.3 repair refitted on A's activations (arm B)", "overnight"),
    ("dbC1", "Interleaved round 1: LoRA 200 steps", "overnight"),
    ("dbC1_rep", "Interleaved round 1 + refitted repair", "overnight"),
    ("dbC2", "Interleaved round 2 (repair + LoRA 400 steps)", "overnight"),
    ("dbC3", "Interleaved round 3 (repair + LoRA 600 steps)", "overnight"),
    ("dbC_final", "Interleaved final: LoRA 600 steps + last refitted repair (arm C)", "overnight"),
    ("gcsolve_mu0.3_rho0.05_L21-41", "ρ=0.05 solve + protect general detail μ=0.3, 21–41", "new"),
    ("gcsolve_mu0.3_rho0.1_x1.2_L21-41", "ρ=0.1 solve + protect detail μ=0.3, strength 1.2", "new"),
    ("gcsolve_mu0.5_rho0.1_L21-41", "ρ=0.1 solve + protect general detail μ=0.5, 21–41", "new"),
    ("gcsolve_mu1_rho0.1_L21-41", "ρ=0.1 solve + protect general-prompt detail μ=1, 21–41", "new"),
    ("gcsolve_mu3_rho0.1_L21-41", "ρ=0.1 solve + protect general-prompt detail μ=3, 21–41", "new"),
    ("dlsolve_rho0.1_L21-41", "ρ=0.1 solve, keep SFT on late-step detail, 21–41", "new"),
    ("dsolve_rho0.1_L21-41", "DC-only solve ρ=0.1 (Base on colour/brightness part only), 21–41", "new"),
    ("dsolve_rho0.1_x1.5_L21-41", "DC-only solve ρ=0.1 ×1.5, 21–41", "new"),
    ("dsolve_rho0.1_x2_L21-41", "DC-only solve ρ=0.1 ×2, 21–41", "new"),
    ("dsolve_rho0.001_L21-41", "DC-only solve ρ=0.001, 21–41", "new"),
    ("csolve_rho0.1_L21-41", "Centred-only solve ρ=0.1 (Base on horizontal variation only), 21–41", "dead end"),
    ("csolve_rho0.001_L21-41", "Centred-only solve ρ=0.001, 21–41", "dead end"),
    ("split_rho0.1_L32-41", "Gram solve ρ=0.1, layers 32–41 only", "new"),
    ("split_rho0.1_L21-31", "Gram solve ρ=0.1, layers 21–31 only", "new"),
    ("solve_mu0_rho0.3_L21-41", "Gram solve ρ=0.3, 21–41", ""),
    ("solve_mu0_rho1_L21-41", "Gram solve ρ=1, 21–41", ""),
    ("qualityplus_lora", "Quality+ (shipped 37–41 + 32–36)", "shipped"),
    ("deband_lora", "DeBand LoRA (shipped 37–41)", "shipped"),
    ("kijai_m1", "Kijai rank-512 LoRA @ −1 (layers 3, 4, 21–41)", ""),
    ("kijai_m05", "Kijai rank-512 LoRA @ −0.5 (layers 3, 4, 21–41)", "new"),
    ("swap055_L21-41", "Plain revert ×0.55 (exact Base−SFT blend), 21–41", "control"),
    ("solve_mu0_L21-41", "Exact revert 21–41", ""),
    ("solve_mu0_L32-41", "Exact revert 32–41", ""),
    ("base", "Base (MOSS-Image Base)", "reference"),
]
HIDE = ("sft", "snoise", "smp_", "rope_")  # diagnostics, not fix candidates

rows = {}
for line in open(os.path.join(OUT, "drift", "scores.jsonl")):
    r = json.loads(line)
    rows[(r["arm"], r["id"], r["seed"])] = r

S = {}
for (arm, pid, seed), r in rows.items():
    S.setdefault(arm, {})[f"{pid}__s{seed}"] = {k: r.get(k) for k in ("band", "ssim", "dino_dist", "arcface", "hf_db")}
    S.setdefault("sft", {})[f"{pid}__s{seed}"] = {"band": r["band_ref"]}

ap = argparse.ArgumentParser()
ap.add_argument("--public")
ap.add_argument("--arms")
cli = ap.parse_args()

labels = {a: (l, t) for a, l, t in ARMS}
order = [a for a, _, _ in ARMS if a in S or os.path.isdir(os.path.join(OUT, "drift", a))] + sorted(a for a in S if a not in labels and not a.startswith(HIDE))
if cli.public:
    keep = cli.arms.split(",")
    order = [a for a in keep if a in order or a == "sft"]
    S = {a: v for a, v in S.items() if a in keep or a == "sft"}
arms_js = [[a, labels.get(a, (a, ""))[0]] for a in order]


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


board = []
for a in order:
    rs = [r for (arm, _, _), r in rows.items() if arm == a]
    solid = {c: mean([r.get("hf_db") for r in rs if r["id"] == "solid_" + c]) for c in ("grey", "blue", "beige")}
    sv = [r["hf_db"] for r in rs if r.get("hf_db") is not None]
    board.append(dict(arm=a, label=labels.get(a, (a, ""))[0], tag=labels.get(a, ("", ""))[1], **solid,
                      solid=mean(sv), solid_med=statistics.median(sv) if sv else None, solid_worst=max(sv) if sv else None,
                      arcface=mean([r.get("arcface") for r in rs if r["category"] == "people"]),
                      ssim=mean([r["ssim"] for r in rs if r["category"] != "solid"]),
                      dino=mean([r.get("dino_dist") for r in rs]),
                      band=mean([r["band"] for r in rs if r["category"] == "bench"]), n=len(rs)))

html = open(os.path.join(HERE, "compare_template.html")).read()
html = (html.replace("__S__", json.dumps(S)).replace("__ARMS__", json.dumps(arms_js, ensure_ascii=False))
        .replace("__BOARD__", json.dumps(board, ensure_ascii=False)).replace("__WHEN__", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
if cli.public:
    for a in set(order) | {"sft", "base"}:
        for f in glob.glob(os.path.join(OUT, "drift", a, "*_00001_.png")):
            dst = os.path.join(cli.public, "img", a, os.path.basename(f).replace("_00001_.png", ".webp"))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                Image.open(f).convert("RGB").save(dst, quality=90, method=6)
    html = html.replace("__IMG__", json.dumps("img/{arm}/{item}__s{seed}.webp"))
    open(os.path.join(cli.public, "index.html"), "w").write(html)
    print(f"{cli.public}/index.html: {len(order)} arms")
else:
    html = html.replace("__IMG__", json.dumps("drift/{arm}/{item}__s{seed}_00001_.png"))
    open(os.path.join(OUT, "compare_solve.html"), "w").write(html)
if not cli.public:
    print(f"compare_solve.html: {len(order)} arms")

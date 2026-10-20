#!/usr/bin/env python3
"""Vertical banding score + diagnostic image.

  comfyui_venv/bin/python workflows/moss/bandviz.py img.png [img2.png ...] [-o outdir]

MOSS banding is in color, not brightness, so the score works on two opponent-color channels
(R-G and B-(R+G)/2): high-pass each row (subtract a 9 px moving average), average down the
columns, FFT each column profile, sum the two magnitudes, and report strongest peak / median.
Real vertical bands make one period stand far above the rest. Writes <name>_band.png per image, plus band_compare.png
when given several images.
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SURFACE, INK, INK2, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
EDGE = 8  # columns dropped at each side; the moving average is invalid there
BAND = (10, 48)  # stripe periods (px) searched for the peak; MOSS bands sit around 16-22 px, 4/8 px is VAE grid texture


def lora_chain(meta):
    """(lora_name, strength) for each LoraLoaderModelOnly, ordered from the model loader outward."""
    g = json.loads(meta)
    users = {n["inputs"]["model"][0]: k for k, n in g.items()
             if n["class_type"] == "LoraLoaderModelOnly" and isinstance(n["inputs"].get("model"), list)}
    out, cur = [], next((k for k, n in g.items() if n["class_type"] == "UNETLoader"), None)
    while cur in users:
        cur = users[cur]
        out.append((g[cur]["inputs"]["lora_name"], g[cur]["inputs"]["strength_model"]))
    return out


def setup_rows(path):
    """Plain-language generation setup read from the ComfyUI prompt embedded in the PNG, as (label, text) rows."""
    meta = Image.open(path).info.get("prompt")
    if not meta:
        return [("Setup", "generation settings are not embedded in this file")]
    nodes = {n["class_type"]: n["inputs"] for n in json.loads(meta).values()}
    rows = []
    unet = nodes.get("UNETLoader", {}).get("unet_name", "?")
    kind = ("SFT: the fully fine-tuned release (the version that bands)" if "sft" in unet
            else "Base: the pre-fine-tune release" if "base" in unet else "")
    rows.append(("Model", f"MOSS-Image {kind}  [{unet}]"))
    loras = lora_chain(meta)
    for i, (n, w) in enumerate(loras):
        what = ("Kijai's SFT-difference LoRA, blocks 21-49 only, rank 512" if "blocks_21_to_49" in n
                else "Kijai's SFT-difference LoRA, all blocks, rank 256" if "rank_256" in n
                else "Kijai's SFT-difference LoRA, all blocks, rank 512" if "rank_512" in n else "LoRA")
        how = ("negative strength subtracts that share of the SFT changes" if w < 0
               else "positive strength adds that share of the SFT changes")
        rows.append((f"LoRA {i + 1}" if len(loras) > 1 else "LoRA", f"{what} at strength {w:+g} ({how})  [{n}]"))
    if not loras:
        rows.append(("LoRA", "none"))
    clip = nodes.get("CLIPLoader")
    if clip:
        rows.append(("Text encoder", clip["clip_name"]))
    samp = ("ER-SDE (eta {eta:g}, noise {s_noise:g}, stages {max_stage})".format(**nodes["SamplerER_SDE"]) if "SamplerER_SDE" in nodes
            else nodes.get("KSamplerSelect", {}).get("sampler_name", "?"))
    sch, sc = nodes.get("BasicScheduler", {}), nodes.get("SamplerCustom", {})
    lat = nodes.get("EmptyFlux2LatentImage", {})
    rows.append(("Sampling", f"sampler {samp}, scheduler {sch.get('scheduler', '?')}, {sch.get('steps', '?')} steps, CFG {sc.get('cfg', '?')}, "
                             f"seed {sc.get('noise_seed', '?')}, {lat.get('width', '?')}×{lat.get('height', '?')}"))
    tsr, slg = nodes.get("TemporalScoreRescaling"), nodes.get("SkipLayerGuidanceDiTSimple")
    rows.append(("Model patches", ("Temporal Score Rescaling ON (k {tsr_k:g}, sigma {tsr_sigma:g})".format(**tsr) if tsr else "Temporal Score Rescaling OFF") + "  ·  " +
                 ("Skip-Layer Guidance ON (layers {single_layers}, {start_percent:.0%}-{end_percent:.0%} of steps)".format(**slg) if slg else "Skip-Layer Guidance OFF")))
    prompt = next((n["inputs"]["text"] for n in json.loads(meta).values()
                   if n["class_type"] == "CLIPTextEncode" and isinstance(n["inputs"].get("text"), str) and n["inputs"]["text"].strip()
                   and "worst quality" not in n["inputs"]["text"]), None)
    if prompt:
        rows.append(("Prompt", textwrap.shorten(prompt, 230, placeholder=" ...")))
    return rows


def short_setup(path):
    """Compact setup label for the comparison chart, e.g. 'SFT + 21-49 LoRA @ -0.5 · ER-SDE'."""
    meta = Image.open(path).info.get("prompt")
    if not meta:
        return os.path.splitext(os.path.basename(path))[0]
    nodes = {n["class_type"]: n["inputs"] for n in json.loads(meta).values()}
    unet = nodes.get("UNETLoader", {}).get("unet_name", "")
    s = "SFT" if "sft" in unet else "Base" if "base" in unet else os.path.splitext(unet)[0]
    for n, w in lora_chain(meta):
        kind = "21-49 LoRA" if "blocks_21_to_49" in n else "r256 LoRA" if "rank_256" in n else "r512 LoRA" if "rank_512" in n else os.path.splitext(n)[0]
        s += f" + {kind} @ {w:+g}"
    if not lora_chain(meta):
        s += ", no LoRA"
    s += " · " + ("ER-SDE" if "SamplerER_SDE" in nodes else nodes.get("KSamplerSelect", {}).get("sampler_name", "?"))
    if "TemporalScoreRescaling" not in nodes:
        s += " · no TSR"
    if "SkipLayerGuidanceDiTSimple" not in nodes:
        s += " · no skip-layer"
    return s


CYCLES = 48  # MOSS SFT stripes: ~48 cycles across the image width (16 px at 768, 21.4 px at 1024)


def local_score(path, wobble=6):
    """Fingerprint banding score that tolerates meandering stripes. 2D spectrum of each opponent-color channel;
    energy within +-wobble vertical-frequency bins (near-vertical stripes) at the known stripe frequency
    (CYCLES across the width, +-1 bin), divided by the median of the same measure at nearby horizontal
    frequencies (+-3..12 bins). About 1 means no stripe."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    W = a.shape[1]
    prof = 0
    for ch in (R - G, B - (R + G) / 2):
        F = np.abs(np.fft.fft2(ch - ch.mean()))
        prof = prof + np.concatenate([F[:wobble + 1], F[-wobble:]], axis=0).sum(0)[:W // 2]
    k0 = round(CYCLES * W / a.shape[1])
    peak = prof[k0 - 1:k0 + 2].max()
    ring = np.r_[prof[k0 - 12:k0 - 2], prof[k0 + 3:k0 + 13]]
    return float(peak / np.median(ring))


def analyze(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    cols, spec = [], 0
    for ch in (R - G, B - (R + G) / 2):
        hp = ch - np.stack([np.convolve(r, np.ones(9) / 9, mode="same") for r in ch])
        col = hp[:, EDGE:-EDGE].mean(0)
        col = col - col.mean()
        cols.append(col)
        spec = spec + np.abs(np.fft.rfft(col))[1:]
    n = len(cols[0])
    periods = n / np.arange(1, len(spec) + 1)
    in_band = (periods >= BAND[0]) & (periods <= BAND[1])
    k = int(np.flatnonzero(in_band)[spec[in_band].argmax()]) + 1
    chroma = np.clip((a - a.mean(2, keepdims=True)) * 5 + 128, 0, 255).astype(np.uint8)
    return dict(img=a.astype(np.uint8), chroma=chroma, cols=cols, spec=spec, n=n, k=k,
                score=spec[k - 1] / np.median(spec), period=n / k, col_std=float(np.hypot(*[c.std() for c in cols])))


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def diagnostic(path, r, out):
    """Pixel-exact layout: image panels are drawn 1:1 at native resolution (dpi 100, 1 px = 1 px)."""
    name = os.path.basename(path)
    H, W = r["img"].shape[:2]
    M, GAP, HEAD, LBL, STRIP, PLOT = 70, 70, 330, 92, 56, 560
    FW = 2 * W + GAP + 2 * M
    FH = HEAD + LBL + H + LBL + STRIP + GAP + LBL + PLOT + 60
    fig = plt.figure(figsize=(FW / 100, FH / 100), dpi=100, facecolor=SURFACE)

    def add(x, y, w, h):  # top-left pixel coordinates -> figure axes
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])

    def text(x, y, s, size, color, **kw):  # y = top of the text, in pixels
        fig.text(x / FW, 1 - y / FH, s, color=color, fontsize=size, va="top", **kw)

    def label(x, y, title, hint):
        text(x, y + 4, title, 14, INK)
        text(x, y + 28, textwrap.fill(hint, max(60, W // 8)), 11.5, INK2, linespacing=1.35)

    text(M, 22, f"MOSS-Image vertical color banding check  ·  score {r['score']:.1f}  ·  stripe width {r['period']:.1f} px", 22, INK)
    text(M, 62, "Score = how strongly one stripe width repeats across the image, compared with a typical stripe width (peak ÷ median of the spectrum, bottom right). "
                "Higher = more banding.", 12.5, INK2)
    text(M, 84, "Reference on the goose bench (same prompt, seed 11): full SFT model scores 41 and is visibly banded; "
                "SFT + blocks 21-49 LoRA at -0.5 scores about 16 and looks clean.", 12.5, INK2)
    text(M, 122, f"Setup for this image ({name})", 14, INK)
    for i, (k, v) in enumerate(setup_rows(path)):
        text(M, 150 + i * 24, k, 12.5, INK)
        text(M + 150, 150 + i * 24, v, 12.5, INK2)

    xr = M + W + GAP
    y = HEAD
    label(M, y, "Generated image", "The image exactly as MOSS produced it, shown pixel for pixel with no resizing. Banding is subtle here: faint vertical color stripes, easiest to spot in flat areas like ceilings, walls and skin.")
    label(xr, y, "Color-only view (5× boosted)", "Brightness is removed and the remaining color is boosted 5×, so tiny color shifts become obvious. A banded image shows evenly spaced vertical stripes running top to bottom; a clean one shows only random grain and object edges.")
    y += LBL
    for x, im in ((M, r["img"]), (xr, r["chroma"])):
        ax = add(x, y, W, H)
        ax.imshow(im, interpolation="nearest")
        ax.axis("off")
    y += H
    label(xr, y + 4, "Average color detail of each column (top: red-green, bottom: blue-yellow)", "Each column of the color view is averaged down to one value and lined up pixel for pixel with the view above. A banded image turns into a regular barcode with a steady rhythm; a clean one looks like random static.")
    y += LBL + 4
    c1, c2 = r["cols"]
    m = max(np.abs(c1).max(), np.abs(c2).max()) or 1
    strip = np.concatenate([np.tile(c1, (STRIP // 2, 1)), np.tile(c2, (STRIP // 2, 1))])
    ax = add(xr + EDGE, y, r["n"], STRIP)
    ax.imshow(strip, cmap="gray", vmin=-m, vmax=m, aspect="auto", interpolation="nearest")
    ax.axis("off")
    y += STRIP + GAP

    label(M, y, "Color detail per column", "The same column averages drawn as a line, where height is how strongly the color shifts in that column. Banding shows up as peaks repeating at a steady spacing across the whole width, not as one-off spikes from objects.")
    label(xr, y, "Which stripe widths repeat (spectrum of the column detail)", "Breaks the column line into repeating patterns and shows how strong each stripe width is. One tall spike inside the shaded 10-48 px range means banding, and the score is that spike's height divided by the dashed median.")
    y += LBL
    ax = add(M + 50, y, W - 50, PLOT - 50)
    style(ax)
    ax.plot(np.arange(EDGE, EDGE + len(c1)), np.hypot(c1, c2), color=BLUE, linewidth=1.2)
    ax.set_xlim(0, W)
    ax.set_xlabel("column (x position in pixels)", color=MUTED, fontsize=11)

    ax = add(xr + 50, y, W - 50, PLOT - 50)
    style(ax)
    periods = r["n"] / np.arange(1, len(r["spec"]) + 1)
    sel = periods <= 128
    ax.axvspan(*BAND, color=GRID, alpha=0.5, linewidth=0)
    ax.plot(periods[sel], r["spec"][sel], color=BLUE, linewidth=2)
    med = np.median(r["spec"])
    ax.axhline(med, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(125, med, "median ", color=MUTED, fontsize=11, va="bottom", ha="right")
    pk = r["spec"][r["k"] - 1]
    ax.plot([r["period"]], [pk], "o", markersize=10, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"score {r['score']:.1f}  (stripes every {r['period']:.1f} px)", (r["period"], pk), xytext=(14, -4),
                textcoords="offset points", color=INK, fontsize=13, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlim(2, 128)
    ax.set_xlabel("stripe width in pixels (log scale); shaded = range searched for banding, 4 and 8 px spikes are normal VAE texture", color=MUTED, fontsize=11)
    ax.set_ylabel("strength", color=MUTED, fontsize=11)

    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def compare(results, out):
    results = sorted(results, key=lambda t: t[1]["score"])
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(results) + 2.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    names = [short_setup(p) for p, _ in results]
    scores = [r["score"] for _, r in results]
    ax.barh(range(len(results)), scores, color=BLUE, height=0.6)
    ax.set_yticks(range(len(results)), names)
    for i, (_, r) in enumerate(results):
        ax.text(r["score"], i, f"  {r['score']:.1f}", va="center", color=INK2, fontsize=10)
    ax.set_xlim(0, max(scores) * 1.12)
    ax.tick_params(axis="y", colors=INK2, labelsize=10.5)
    ax.set_xlabel("banding score (peak ÷ median of the column-color spectrum)", color=MUTED, fontsize=10)
    fig.text(0.01, 0.985, "MOSS-Image vertical color banding, higher = more banding", color=INK, fontsize=14, va="top")
    fig.text(0.01, 0.985 - 0.32 / fig.get_figheight(), "Same prompt and seed for every bar. Goose bench reference: full SFT 41 is visibly banded; SFT + 21-49 LoRA @ -0.5 at ~16 looks clean.",
             color=INK2, fontsize=10.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def hf_chroma_power(path):
    """Total chroma power at periods under 40 px, any orientation (centre 512x512 crop): the orientation-free artifact measure
    for solid colours, where any colour texture is artifact. The vertical-only banding score misses rotated/diagonal residue."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    H, W = a.shape[:2]
    a = a[H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256]
    win = np.outer(np.hanning(512), np.hanning(512))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(512) * 512, np.fft.fftfreq(512) * 512))
    total = 0.0
    for ch in (a[..., 0] - a[..., 1], a[..., 2] - (a[..., 0] + a[..., 1]) / 2):
        total += float((np.abs(np.fft.fft2((ch - ch.mean()) * win)) ** 2)[k >= 512 / 40].sum())
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("-o", "--outdir")
    p.add_argument("--no-plots", action="store_true", help="print scores only")
    a = p.parse_args()
    results = []
    for path in a.images:
        r = analyze(path)
        results.append((path, r))
        print(f"{os.path.basename(path):28s} score={r['score']:5.1f} period={r['period']:5.1f}px col_std={r['col_std']:.3f}")
        if not a.no_plots:
            outdir = a.outdir or os.path.dirname(os.path.abspath(path))
            diagnostic(path, r, os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + "_band.png"))
    if len(results) > 1 and not a.no_plots:
        compare(results, os.path.join(a.outdir or os.path.dirname(os.path.abspath(a.images[0])), "band_compare.png"))


if __name__ == "__main__":
    main()

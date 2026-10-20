# MOSS-Image SFT vertical stripes: research notes (October 2026)

This is the full write-up behind the DeBand LoRAs: what the stripe artifact is, every fix we and others tried, how they were measured, and what didn't work. It is written to be read by people and by the LLM helping them, so it states facts, numbers and file locations plainly. The interactive comparison of all fixes is at [https://matlowai.github.io/MOSS-DeBand/](https://matlowai.github.io/MOSS-DeBand/) (source in `docs/`). Weights live on [Hugging Face](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand).

## 1. Which fix to use

| You want | Use | Trade-off |
|---|---|---|
| Stripes reduced, everything else as SFT made it | [nynxz's Destripe LoRA](https://huggingface.co/nynxz/MOSS-Image) @1.0 (768 px) / 1.5 (1024 px) | Gentlest fix measured (faces 0.94–0.96 similar to SFT). Only partly cleans flat colour areas (about −4 dB; stripes fainter but still visible in the colour view). |
| Clean flat backgrounds, skies, studio backdrops | **DeBand Strong** ([HF `loras/`](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand/tree/main/loras), experimental) at 1.0, or 1.2 for the most reliable cleanup | Strongest cleanup short of the Base model (at 1.2: typical −10.7 dB, worst −5.5 dB on solids; faces 0.77). Faces and fine detail move more (0.83 similarity), and it is tuned to the CFG 3 + SLG + TSR preset sampler; at CFG 2.5 without SLG/TSR it drifts much more. |
| The original one-file fix | DeBand main fix (layers 37–41) / Quality+ (32–41) | Exact Base weights in the last layers. Helps most at 768 px and on the vertical stripe score; on 1024 px solids the orientation-free measure shows little net change (main) or about −3 dB (Quality+). |
| No stripes at all | MOSS-Image Base | Different model: faces and style change a lot (0.45 similarity). |

Stacking nynxz's LoRA on top of DeBand Strong adds nothing over Strong alone.

## 2. The artifact

- **What it is:** faint, evenly spaced vertical colour stripes in smooth areas (skies, fog, walls, backdrops). A chroma shift with almost no brightness change; invisible until colour is boosted, then obvious.
- **Count, not size:** about 45–65 cycles across the image width (≈48 at 1024 px) at every tested resolution and aspect ratio (1–2 MP; 1:1, 4:3, 3:4, 16:9, 9:16). Always vertical.
- **Tied to position encoding:** rescaling only the horizontal RoPE positions moves the count (column scale 0.8 → ≈64 cycles, 1.25 → ≈40); a common shift of all positions does nothing; shifting the image relative to the text halves it. Not a VAE or fixed pixel period.
- **Mostly a damping failure:** the ER-SDE sampler re-injects noise every step and SFT fails to clean up that band of it. Lowering ER-SDE `s_noise` nearly removes the stripes but fades and blurs images (0.7: contrast loss; 0.5: text disappears; 0: blank) — a diagnostic, not a fix. CFG scales the stripes (CFG 1 ≪ CFG 3 ≪ CFG 5); SLG off helps somewhat; TSR is irrelevant; Euler roughly halves them.
- **Where in the model:** layers 21–41 (of 42). Reverting 37–41 to Base fixes most cases at 768 px; 32–36 matters at larger sizes. Gradient attribution (nynxz) puts the causal directions in layers 24–41, mostly the MLP `ffn_up_gate`.
- **When:** the stripe forms mid-sampling (roughly steps 5–14 of 25) and is visible in the model's x0 prediction.
- **Not the VAE:** it appears with every VAE tried and only with SFT, never with Base (confirmed independently in the [OpenMOSS SFT stripe thread](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/discussions/2)).
- **Where it probably comes from (unverified hypothesis):** community members suggest the SFT stage was trained partly on outputs of GPT Image 2/2.5, whose images carry a similar faint vertical colour-stripe pattern (described there as a watermark-like artifact). That fits what we measured: a pattern learned from training images would be fixed relative to the image width (a fixed cycle count, tied to the horizontal position encoding), would appear only in the fine-tuned SFT and not in Base, and would be a narrow, separable behaviour (which is why 64 directions can remove it). Testing it would mean running the same colour-view and spectrum measurement on GPT Image outputs; we have not done that.

## 3. Approaches compared

Every approach adds back some piece **D** of **Δ = W_Base − W_SFT** (what fine-tuning changed). They differ in how D is chosen.

1. **Block revert (DeBand v1, published):** D = Δ in chosen layers. Picks by location only; undoes everything SFT learned there.
2. **Compressed revert (Kijai's rank-512 LoRA at −1 / −0.5):** a low-rank approximation of Δ over layers 3, 4, 21–41. Measured as non-selective: it changes layer outputs on solid-colour inputs (83 % of a full revert) about as much as on general inputs (79 %).
3. **Closed-form Gram solve (DeBand Strong, new):** per linear, the smallest weight change that makes the layer output what Base outputs on the inputs it sees while SFT draws solid colours, while keeping SFT's output on the horizontal detail of general prompts:
   `D = Δ · G_flat · (G_flat + μ·G_detail + ε·I)⁻¹`. Picks by input statistics. Selective (92 % output change on solid-colour detail vs 75 % on general detail at ρ=0.1), but undoes everything those inputs touch, and it is fitted to the sampler the statistics were recorded under.
4. **Gradient-picked directions (nynxz Destripe):** SVD of each layer's Δ into rank-1 directions; one backward pass scores every direction by its contribution to the 48-cycle stripe energy of the x0 prediction over steps 4–13; the 64 strongest are subtracted (25 matrices, rank 1–7, 1.9 MB). Picks by measured effect. Same directions at negative strength raise stripe energy ≈22× (their sign check).
5. **Training (negative):** a rank-32 LoRA trained on clean solid renders plus SFT's own general renders. No effect at 200 steps; at 600 steps it learned "remove fine detail" and degraded everything (blur, colour shifts, faces 0.12 similar).

### 3a. How nynxz's LoRA is so small and still works

The numbers, measured from their file and our weight analysis:

| | nynxz Destripe | DeBand Strong (Gram solve) | DeBand v1 main fix |
|---|---|---|---|
| Matrices touched | 25 of 252 | 84 (all of layers 21–41) | 20 (layers 37–41) |
| Stored numbers | 945,152 (1.9 MB) | ≈2.4 billion (4.8 GB) | ≈570 million (1.1 GB) |
| Share of SFT's weight change undone (energy) | **0.09 %** of all of Δ (0.2 % of layers 24–41) | ≈30 % of layers 21–41 | 100 % of layers 37–41 |

How the recipe gets there (from their model card, their discussion post and the metadata embedded in the LoRA: K = 256, 768 px, seeds 11 and 14, steps 4–13, ER-SDE, CFG 2.5):

1. **Break the change into pieces.** Any matrix change Δ is a sum of rank-1 pieces, Δ = Σⱼ σⱼ·uⱼ·vⱼᵀ (an SVD). Each piece is one "input pattern vⱼ → output pattern uⱼ" rule that fine-tuning added. Storing one piece costs only (rows + columns) numbers, e.g. 19,456 for an MLP up/gate matrix instead of 50 million for the full matrix.
2. **Score every piece with one backward pass.** Make the stripe a single number: the energy of the 48-cycle column pattern in the model's predicted clean image, summed over steps 4–13 of real sampling runs. Backpropagate it once to get the gradient G = ∂E/∂W for every weight matrix. The effect of undoing piece j is then, to first order, just a dot product: score ≈ σⱼ·uⱼᵀ·G·vⱼ. With the gradients in hand, scoring all 256 pieces of all 252 matrices costs a few matrix-vector products. No per-piece renders and no search.
3. **Remove only the top 64.** Subtract the 64 pieces that feed the stripe most. They land in 25 matrices (layers 24–41, mostly the MLP's `ffn_up_gate`), 1–7 pieces each. 48 of the 64 are among the 8 largest pieces of their matrix, but most of each matrix's large pieces are not picked, and 227 matrices are left alone entirely.
4. **Check it's causal.** Adding the same pieces back with the opposite sign makes the stripes ≈22× stronger. A correlation wouldn't do that.

**Why it's so gentle:** the stripe is one narrow behaviour (one frequency, colour only, one part of sampling), so a handful of pieces carry most of it. Everything else SFT learned (faces, style, prompt following) lives in the other 99.9 % of the change, which the LoRA never touches. Choosing by location (whole layers) or by input statistics (the Gram solve) can't be that precise, because those criteria also grab pieces that matter for everything else.

**Why it's only a partial fix on our screen (≈ −4 dB on 1024 px solids, vs ≈ −16 for Base):** the score is first-order (it predicts the effect of tiny changes; removing a whole piece is a large change); it was computed at 768 px with CFG 2.5 on 4 prompts and 2 seeds; it targets the vertical 48-cycle energy, so residue at other angles or frequencies isn't counted; and it stops at 64 pieces. Each of these is a knob: more prompts and seeds, the sampler settings people actually use, an orientation-free objective, an iterative "remove, re-score, repeat" loop, and a per-piece strength instead of all-or-nothing.

**A related observation from our side:** compressing DeBand Strong into a low-rank LoRA (keeping its biggest SVD pieces) keeps its drift but loses most of its cleanup, at every size tried:

| DeBand Strong size | Solids median dB | mean | worst | ArcFace |
|---|---|---|---|---|
| full rank (4.8 GB) | −6.9 | −11.2 | −2.0 | 0.83 |
| rank 512 (1.1 GB) | −4.5 | −4.9 | −0.2 | 0.84 |
| rank 256 (530 MB) | −4.3 | −4.8 | +0.6 | 0.83 |
| rank 128 (264 MB) | −2.9 | −3.3 | +0.9 | 0.86 |
| rank 64 (132 MB) | −3.1 | −3.6 | +1.0 | 0.89 |
| nynxz Destripe @1.5 (1.9 MB), for reference | −3.9 | −4.4 | −1.8 | 0.94 |

Ranking pieces by size is not ranking them by effect; that's the gap nynxz's gradient scoring closes. It's also why DeBand Strong ships at full rank or not at all.

**The lesson that transfers:** choose what to undo by the behaviour you want to change, measured through the real computation. Location (1) and input statistics (3) are proxies; gradients through sampling (4) measure the target directly and give the smallest edits.

## 4. How everything was measured

Benchmark ("screen"): 13 items through the stock preset workflow (ER-SDE, 25 steps, CFG 3, SLG layers 22/24 over 10–70 %, TSR 0.95): solid grey/blue/beige at 1024², two banding-bench scenes at 768×1024, three portraits, one anime, two animals, a chalkboard, an interior; seed 11, plus seed 14 (solids + portraits) and seed 15 (solids) for the main arms. All numbers are relative to plain SFT with the same prompt and seed.

- **Solids, HF dB (primary):** orientation-free colour-detail power at periods under 40 px (centre 512² crop of the R−G and B−(R+G)/2 channels) relative to SFT, in dB; lower = cleaner. Catches stripes rotated into diagonals.
- **ArcFace:** identity similarity of the largest face to SFT's face for the same prompt/seed (1.0 = same person).
- **DINO distance / SSIM:** overall drift on non-solid items.
- **Vertical score (bandviz `local_score`):** 48-cycle vertical fingerprint, ≈1.0 clean. Useful but fooled by rotated residue and real vertical patterns (pleats, sails).

Single solid images swing 10–30 dB between seeds because a solid tends to come out either fully clean or still striped. Report median, mean and worst case together; treat differences under ~3 dB (or ~0.03 ArcFace) as noise.

Results (screen; solids = grey/blue/beige × seeds 11/14/15, faces = 3 portraits × seeds 11/14, relative to plain SFT):

| Fix | Solids median dB | mean | worst | ArcFace mean | min | DINO dist | SSIM | solids n |
|---|---|---|---|---|---|---|---|---|
| MOSS-Image Base (reference) | −16.6 | −28.6 | −10.4 | 0.45 | 0.28 | 0.150 | 0.58 | 9 |
| DeBand v1 main fix (37–41) | +0.4 | −0.0 | +2.3 | 0.91 | 0.72 | 0.042 | 0.86 | 9 |
| DeBand v1 Quality+ (32–41) | −3.0 | −3.2 | +1.3 | 0.85 | 0.67 | 0.067 | 0.73 | 9 |
| Kijai rank-512 LoRA @ −0.5 | −1.8 | −1.9 | +1.1 | 0.87 | 0.70 | 0.056 | 0.79 | 9 |
| Kijai rank-512 LoRA @ −1 | −5.8 | −6.0 | −1.3 | 0.76 | 0.59 | 0.080 | 0.69 | 9 |
| nynxz Destripe @1.0 | −3.2 | −3.5 | −1.9 | 0.94 | 0.78 | 0.028 | 0.93 | 9 |
| nynxz Destripe @1.5 | −3.9 | −4.4 | −1.8 | 0.94 | 0.71 | 0.031 | 0.92 | 9 |
| Plain 55 % blend toward Base, 21–41 (control) | −4.9 | −5.6 | −4.7 | 0.77 | 0.64 | 0.074 | 0.68 | 3 |
| Gram solve ρ=0.1, no detail protection | −6.6 | −13.6 | −3.8 | 0.72 | 0.50 | 0.091 | 0.66 | 9 |
| **DeBand Strong** (Gram solve μ=0.3 ρ=0.05, full rank) | −6.9 | −11.2 | −2.0 | 0.83 | 0.66 | 0.065 | 0.71 | 9 |
| **DeBand Strong at strength 1.2** | −10.7 | −13.5 | −5.5 | 0.77 | 0.66 | 0.078 | 0.67 | 9 |
| Gram solve μ=0.3 ρ=0.1 at strength 1.2 | −10.0 | −10.7 | −5.1 | 0.80 | 0.67 | 0.077 | 0.68 | 9 |
| nynxz @1.5 stacked on Strong's solve | −6.9 | −7.1 | −3.1 | 0.82 | 0.65 | 0.071 | 0.71 | 9 |
| Trained de-band LoRA, 600 steps (failed) | −3.1 | −2.6 | +5.2 | 0.12 | 0.03 | 0.363 | 0.56 | 9 |

At nynxz's sampler settings (CFG 2.5, no SLG/TSR; relative to SFT at the same settings):

| Fix | Solids median dB | mean | worst | ArcFace mean |
|---|---|---|---|---|
| nynxz @1.5 | −4.6 | −4.6 | −3.0 | 0.96 |
| DeBand Strong's solve | −5.6 | −7.6 | −2.6 | 0.62 |

## 5. Findings worth knowing

**Measuring**
- Vertical-only stripe scores are fooled: several solves rotated the stripe into short diagonals and scored "clean". Use an orientation-free measure on solids and look at the colour view.
- Means get dragged by the occasional perfectly clean solid (one grey reached −102 dB); medians and worst cases tell the typical story.
- In every layer about half of the input energy is one direction: each token's overall colour/brightness plus the timestep modulation shift. Any edit touching it looks "big" by raw output-change energy (ρ=10 changes ~1 % of the weights but shows 79 % "output change"). Measure selectivity on row-centred activations. On that measure (share of a full revert's output change, solid-colour vs general detail): nynxz's Destripe 24 % vs 16 % (1.5×, out-of-sample), Gram solve ρ=1 76 % vs 55 % (1.4×), ρ=0.1 92 % vs 75 % (1.2×), Kijai 83 % vs 79 % (1.06×), block reverts equal. DeBand Strong reads 83 % vs 43 % (1.9×), but it was fitted to protect exactly that general-detail statistic, so that number is in-sample.

**The Gram solve**
- Genuinely selective: a plain 55 % blend toward Base over the same layers (a similar-size edit) reaches only −5.6 dB at the same face drift.
- Protect detail, not everything: preserving general prompts' full statistics (μ on the raw general Gram) kills the fix, because the colour/brightness direction is shared. Preserving only their horizontal detail (μ=0.3 on the row-centred Gram) keeps most of the cleanup and lifts face similarity from 0.72 to 0.83.
- Matching Base only on horizontal detail does nothing for stripes (≈0 dB) while still moving faces. Matching Base only on the remainder (per-row colour/brightness plus text tokens) gives ≈ −5 dB with faces at 0.86. With a shared denominator the two edits add exactly to the full one; the extra image-level effect of using both comes from nonlinear sampling.
- It needs all of layers 21–41 (either half alone: −3 to −5 dB) and the early/middle steps (applying it only in steps 13–25: ≈ −1 dB).
- It is sampler-specific: fitted under CFG 3 + SLG + TSR, it keeps faces at 0.83 there but 0.62 at CFG 2.5 without SLG/TSR. nynxz's directions keep 0.94–0.96 on both.

**Other**
- Per-layer weight-change maps and selectivity charts for every fix are in [edit anatomy](https://matlowai.github.io/MOSS-DeBand/edit_anatomy.html).
- Training MOSS-Image in ComfyUI's `TrainLoraNode` works after a one-line dtype fix in the training path (q/k cast back after norm + RoPE), with the memory compiler off and `checkpoint_depth=2` (13.6 GB at 1024², 1.26 s/step on an RTX 5090). Depth 1 runs out of memory at ~27.5 GB.

## 6. Beyond MOSS: the same lesson on a character LoRA

The same question — keep a concept when asked for, change nothing else — appears with character LoRAs ([DeTangle](https://matlowai.github.io/DeTangle/)). Its "A-side surgery" refits one LoRA factor from captured activations, the same family as the Gram solve, and shows the same weakness. On two of eight test scenes (a sci-fi "space station" and a fantasy "wizard workshop", the only prompts ending in "Photorealistic live-action science-fiction/fantasy film still.") the character's likeness collapses after surgery and the face falls back to the base model's own default for the scene; for the space station that default face looks Ripley-like, but measured against the model's concept of Sigourney Weaver (her studio portraits rendered by name) it is a different person: 0.27 for the base model's default woman and 0.33 after surgery, against 0.75–0.82 for her own renders. The base model's default woman leans slightly toward Weaver in every scene (beach and arcade too), so there is no sci-fi-specific Sigourney association; surgery simply hands the face back to the model's generic woman, and styling does the rest.

Probe results (424 renders, same protocol as the DeTangle study):
- Only transformer blocks 8–15 matter: restoring the original LoRA there brings likeness back (0.36–0.55); every other block window, including the first, does nothing.
- Those same blocks bring the unwanted bleed back (+0.19 to +0.30): the parts that let the character win against a strong scene prior are the parts that leak it into untriggered prompts. Input statistics cannot separate them.
- The film-still sentence is a real but partial cause: adding it to other scenes cuts post-surgery likeness by 0.15–0.26 while the original LoRA is unaffected; stripping all cinematic phrasing recovers most of it.
- After surgery the trigger loses to any explicitly described identity (a named actress or an ethnicity descriptor).

The proposed next step is nynxz's method applied there: score the removed LoRA directions in blocks 8–15 by their effect on likeness and bleed, per scene, and keep only what helps without leaking. (The character LoRA itself is private and not published.)

## 7. Lineage and credits

- **Kijai** observed that SFT's later blocks cause the banding and that a negative SFT-difference LoRA reduces it — the starting point.
- **DeBand v1 (MATLOWAI):** block-swap ablations located layers 37–41 / 32–36; `bandviz` (48-cycle fingerprint and colour view); exact-revert LoRAs and preset workflows.
- **nynxz:** took the 48-cycle stripe signature, the step window and the layer range from DeBand, replaced "revert whole layers" with gradient-scored SVD directions, and measured with `bandviz` on 12 held-out images. Their write-up: [model card](https://huggingface.co/nynxz/MOSS-Image), [discussion with OpenMOSS](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/discussions/8).
- **Community threads:** the [OpenMOSS SFT stripe thread](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/discussions/2) (first reports with videos, the VAE debunk, the GPT Image hypothesis) and nynxz's [write-up for the MOSS team](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/discussions/8), which asks whether SFT used position-scaling, bucketing or resized/synthetic data with this pattern, and offers per-direction scores. No reply from OpenMOSS as of 2026-10-19.
- **This update:** the orientation-free benchmark, Gram solves (DeBand Strong), edit anatomy, negative results, and the head-to-head including nynxz's LoRA on matched settings.

## 8. Open questions

- nynxz-style gradient scoring with an orientation-free objective, scored across several sampler settings — likely the best of both (selective and complete).
- Why SFT damps that band less than Base: a replay harness that feeds the same noisy latents to both and measures per-step gain at the stripe frequency is designed but not built.
- Whether a fixed SFT checkpoint is possible upstream (see nynxz's discussion with OpenMOSS).
- Whether GPT Image outputs carry the same 48-cycle colour pattern (the origin hypothesis above).

## 9. Files in this update

| Path | What |
|---|---|
| [HF: `loras/moss_image_sft_deband_strong_full_bf16.safetensors`](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand/tree/main/loras) | DeBand Strong (experimental), see §1 |
| `docs/` (GitHub Pages) | Static pages: `index.html` (all fixes side by side, colour view, leaderboard), `edit_anatomy.html`, `learnings.html` |
| `research/` | Tools, benchmark prompts, lab node, every score row (`research/scores/scores.jsonl`); see `research/README.md` |

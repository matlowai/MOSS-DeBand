# Research tools for MOSS-Image SFT DeBand

Everything used to measure and build the fixes in this repo. The findings are in [../RESEARCH.md](../RESEARCH.md); the interactive comparison is at [matlowai.github.io/MOSS-DeBand](https://matlowai.github.io/MOSS-DeBand/).

## Setup

1. Copy this folder's scripts to `ComfyUI/workflows/moss/` (they find ComfyUI two levels up) and `moss_block_lab/` to `ComfyUI/custom_nodes/moss_block_lab/`, then restart ComfyUI.
2. Models: `moss_image_sft_bf16.safetensors`, `moss_image_base_bf16.safetensors`, `qwen3.5_9b_int8_convrot.safetensors` and `flux2-vae.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image).
3. Scoring needs `numpy`, `scikit-image`, `transformers` (DINOv2-base) and `insightface` (buffalo_l, CPU is fine).

## The benchmark ("screen")

`drift_runner.py <arm> [options]` queues 13 items through ComfyUI's API with the stock preset workflow (`moss_banding_fix_presets_api.json`, preset 5 = plain SFT, ER-SDE, 25 steps, CFG 3, SLG 22/24, TSR): 3 solid colours (grey, blue, beige), 2 banding-bench scenes at 768×1024, 3 people, 1 anime, 2 animals, 1 typography, 1 interior. Renders land in `output/moss/drift/<arm>/<item>__s<seed>_00001_.png`.

```bash
python3 drift_runner.py sft                                                    # reference
python3 drift_runner.py base --unet moss_image_base_bf16.safetensors
python3 drift_runner.py mylora --lora my_fix.safetensors:1.0
python3 drift_runner.py mylora --lora my_fix.safetensors:1.0 --seeds 14 --only solid_grey,solid_blue,solid_beige,p_young_woman_east_asian
python3 drift_runner.py solve --solve 0.3,0.05,21-41,1,,,genc                 # in-memory Gram solve (needs the Grams below)
python3 drift_runner.py cfg25 --cfg 2.5 --no-slg --no-tsr                      # other sampler settings (score against an SFT arm with the same settings)
```

`drift_score.py <ref_arm> <arm> [...]` pairs images by item and seed and appends rows to `output/moss/drift/scores.jsonl`:

| Field | Meaning |
|---|---|
| `hf_db` | Solids only. Orientation-free colour-detail power (periods under 40 px, centre 512² crop, R−G and B−(R+G)/2) relative to the reference, in dB. Lower = cleaner. The primary stripe metric. |
| `band` | `bandviz.local_score`: vertical 48-cycle stripe fingerprint, ≈1.0 = clean. Fooled by rotated residue and real vertical patterns. |
| `ssim`, `dino_dist` | Overall drift from the reference image. |
| `arcface` | People only. Identity similarity of the largest face to the reference image's face (1.0 = same person). |

`build_compare.py` turns `scores.jsonl` into the comparison page (any number of columns, colour view, sortable leaderboard). `--public <dir> --arms a,b,c` writes a hostable copy with WebP images. `auto_score.py <arms...>` scores and rebuilds every 10 minutes until the ComfyUI queue is empty.

## Closed-form Gram solve

The lab node `MOSSGramCapture` records, for each linear in layers 21–41, the Gram matrix of its inputs (Σ x xᵀ) over every model call of a render. Options: `image_centered` (image tokens only, each row's mean along the width removed) and `sigma_max` (only late steps). `MOSSGramSave` writes them to `output/moss/grams/<tag>.safetensors` (8 GB for 21–41).

`gram_runner.py <src> <tag>` replays a capture prompt set under a new tag (`--image-centered`, `--sigma-max`, `--lora`, `--solve` to capture on an edited model). The captures used here are 36 solid-colour prompts (`flat`) and 36 general prompts (`gen`) at 1024², 1216×832 and 832×1216; the prompts are embedded in the capture PNGs that `MOSSGramSave` runs alongside.

`MOSSGramSolve` (in-memory) and `export_gramsolve.py` (to a LoRA file) compute, per linear, with Δ = W_Base − W_SFT:

```
D = Δ · G_num · (G_flat + μ·G_gen + ε·I)⁻¹      ε = ρ · tr(G_flat + μ·G_gen) / d
```

`G_num` defaults to `G_flat` (match Base on solid-colour inputs); `num_tag` / `!num_tag` target only a captured part or its complement. DeBand Strong is `μ=0.3, ρ=0.05, gen=genc` (the general Gram with the per-row colour/brightness removed, so it protects detail without blocking the fix). `build_lowrank.py <file> <ranks...>` makes truncated-SVD LoRAs from any full-difference file.

## Edit anatomy

`edit_anatomy.py` (+ `edit_anatomy_c.py` for the horizontal-detail measure) computes, for each fix and layer, the weight-change size relative to a full revert and the output-change energy tr(D·G·Dᵀ) on solid-colour vs general activations; `build_anatomy.py` draws it. `edit_anatomy_lora.py <tag> <lora>` adds any LoRA (lora_up/down or .diff keys; split q/k/v mapped into fused qkv) as another arm. Measure selectivity on the centred Grams: on raw Grams one per-token colour/brightness/modulation direction holds about half the energy and makes any edit look large.

## Training (negative result, kept for reproduction)

`trainset_runner.py`, `build_trainset.py`, `train_runner.py` drive ComfyUI's `TrainLoraNode` on MOSS. Needed: run the trainer instance with `--disable-comfy-compiler`, `checkpoint_depth=2` at 1024², and a training-path dtype fix in `comfy/ldm/moss_image/model.py` (q/k cast back to the input dtype after norm + RoPE). A rank-32 de-band LoRA did nothing at 200 steps and degraded images at 600 (see RESEARCH.md).

The 72 capture prompts (with sizes and seeds) ship as 128 px stand-in PNGs in `gram_capture/` that carry each original render's embedded API prompt; copy them to `ComfyUI/output/moss/gram_capture/` and `gram_runner.py flat <tag>` / `gram_runner.py gen <tag>` will replay them exactly.

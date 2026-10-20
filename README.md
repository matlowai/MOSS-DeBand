# MOSS-DeBand

Fixing the faint vertical colour stripes that **MOSS-Image SFT** paints into skies, fog, walls and studio backdrops: every fix side by side, how each one works, and the tools to measure it.

- **Interactive comparison:** [matlowai.github.io/MOSS-DeBand](https://matlowai.github.io/MOSS-DeBand/). Pick any fixes as columns, same prompt and seed, colour view (press C) to see the stripes, sortable leaderboard.
- **Write-up:** [RESEARCH.md](RESEARCH.md). What the artifact is, five approaches compared on one benchmark, how nynxz's 1.9 MB LoRA works, what didn't work, open questions.
- **Weights and ComfyUI workflows:** [MATLOWAI/MOSS-Image-SFT-DeBand](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand) on Hugging Face (plus a one-file [int8 version](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand-int8)).
- **Tools:** [research/](research/README.md). Benchmark runner, orientation-free stripe metric, Gram-solve lab nodes, edit-anatomy analysis, every score row.

## Which fix

| You want | Use |
|---|---|
| Stripes reduced, everything else as SFT made it | [nynxz's Destripe LoRA](https://huggingface.co/nynxz/MOSS-Image) (1.0 at 768 px, 1.5 at 1024 px). The gentlest fix; partial cleanup on flat colour areas. |
| The cleanest flat areas | DeBand Strong (experimental, [HF](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand/tree/main/loras)) at 1.0–1.2. More cleanup, more drift, tuned to the CFG 3 + SLG + TSR preset. |
| One file, no LoRAs | [DeBand int8](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand-int8) |

Numbers for all of them are in [RESEARCH.md §4](RESEARCH.md#4-how-everything-was-measured).

## The one idea worth taking away

All of these fixes undo part of what SFT's fine-tuning changed. The difference is how they choose what to undo: by location (revert whole layers), by input statistics (a closed-form solve on the activations of solid-colour images), or by measured effect (score every rank-1 piece of the change by the gradient of the stripe itself, then remove the worst). The last one, nynxz's, undoes 0.09 % of the change and keeps images almost identical; choosing by effect, measured through the real sampling process, is what makes edits small and precise. The same lesson shows up in character-LoRA cleanup ([DeTangle](https://matlowai.github.io/DeTangle/)).

## Credits

[OpenMOSS](https://huggingface.co/OpenMOSS-Team) for MOSS-Image, [Comfy-Org](https://huggingface.co/Comfy-Org/MOSS-Image) for the ComfyUI repack, [Kijai](https://huggingface.co/Kijai/MOSS-image_experimental) for the first pointer to SFT's later blocks, [nynxz](https://huggingface.co/nynxz/MOSS-Image) for the gradient-attributed Destripe LoRA, and the people in the [OpenMOSS stripe thread](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/discussions/2).

Code and docs: Apache-2.0 (see [LICENSE](LICENSE)).

#!/usr/bin/env python3
"""Queue one arm of the DeBand screen through the ComfyUI API: drift suite (1024x1024) + banding bench (768x1024) + solid colours.

  python3 workflows/moss/drift_runner.py <arm> [model options] [sampler options] [--full | --only id,id] [--front]
Model options (chained in this order on the SFT UNet): --swap ratio,layers,parts (MOSSBlockSwap toward Base),
  --solve mu,rho,layers[,strength] (MOSSGramSolve), --rope col_scale,row_scale,image_shift,common_shift (MOSSRopeOverride),
  --lora file:strength (first LoRA slot), --unet file, --preset N.
Sampler options: --cfg, --no-slg, --no-tsr, --euler, --s-noise.
Renders go to output/moss/drift/<arm>/<id>__s<seed>_00001_.png.
"""
import argparse, copy, json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SOLID = "A completely flat, uniform, solid {c} background that fills the entire frame. Nothing else: no objects, no texture, no pattern, no gradient, no vignette, no text. An evenly lit seamless studio backdrop of one single color."
SCREEN = {"solid_grey", "solid_blue", "solid_beige", "bench_fog_forest", "bench_fjord_sky", "p_elder_man_west_african", "p_young_woman_east_asian",
          "p_child_boy_black", "a_anime_girl_school", "x_golden_retriever", "x_red_fox_snow", "t_cafe_chalkboard", "o_kitchen_scene"}

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--swap", help="ratio,layers,parts for MOSSBlockSwap toward Base (parts like qkv;o;ffn_up_gate;ffn_down or all)")
ap.add_argument("--solve", action="append", default=[], help="mu,rho,layers[,strength[,flat_tag[,num_tag or !num_tag for its complement[,gen_tag]]]] for MOSSGramSolve (layers like 21-41); repeat to chain layer groups")
ap.add_argument("--rope", help="col_scale,row_scale,image_shift,common_shift for MOSSRopeOverride")
ap.add_argument("--lora", help="file:strength replacing the first LoRA slot")
ap.add_argument("--window", help="file:strength:lo-hi: apply a LoRA as a hook only for sampling steps lo..hi (1-indexed, of 25)")
ap.add_argument("--unet", default="moss_image_sft_bf16.safetensors")
ap.add_argument("--preset", type=int, default=5, help="preset index of the stock workflow (5 = Off)")
ap.add_argument("--cfg", type=float)
ap.add_argument("--no-slg", action="store_true")
ap.add_argument("--no-tsr", action="store_true")
ap.add_argument("--euler", action="store_true", help="Euler instead of ER-SDE")
ap.add_argument("--s-noise", type=float)
ap.add_argument("--seeds", default="11")
ap.add_argument("--full", action="store_true", help="all 32 items instead of the 13-item screen set")
ap.add_argument("--only", help="comma list of item ids (or extra solid/size items like solid_grey@768x1360)")
ap.add_argument("--front", action="store_true", help="put the jobs at the front of the ComfyUI queue")
ap.add_argument("--url", default="http://localhost:8188")
a = ap.parse_args()

wf = json.load(open(os.path.join(HERE, "moss_banding_fix_presets_api.json")))
api, names = wf["prompt"], wf["presets"]
items = [(d["id"], d["prompt"], 1024, 1024) for d in json.load(open(os.path.join(HERE, "bench/drift_suite_v1.json")))]
items += [("bench_" + d["id"], d["prompt"], 768, 1024) for d in json.load(open(os.path.join(HERE, "bench/prompts_v1.json")))
          if d["id"] in ("fog_forest", "fjord_sky", "goose", "oil_still_life", "product_glass")]
colors = {"grey": "neutral mid grey", "blue": "soft sky blue", "beige": "warm beige"}
items += [("solid_" + k, SOLID.format(c=c), 1024, 1024) for k, c in colors.items()]
if a.only:
    by_id = {i[0]: i for i in items}
    picked = []
    for tok in a.only.split(","):
        if "@" in tok:  # e.g. solid_grey@768x1360
            base, size = tok.split("@")
            w, h = (int(v) for v in size.split("x"))
            picked.append((f"{base}_{w}x{h}", by_id[base][1], w, h))
        else:
            picked.append(by_id[tok])
    items = picked
elif not a.full:
    items = [i for i in items if i[0] in SCREEN]

n = 0
for seed in (int(s) for s in a.seeds.split(",")):
    for pid, text, w, h in items:
        p = copy.deepcopy(api)
        p["1"]["inputs"].update(choice=names[a.preset], index=a.preset)
        p["3"]["inputs"]["unet_name"] = a.unet
        p["6"]["inputs"]["value"] = text
        p["22"]["inputs"]["value"] = seed
        p["23"]["inputs"].update(width=w, height=h)
        cur = ["3", 0]
        if a.swap:
            ratio, layers, parts = a.swap.split(",")
            p["95"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_base_bf16.safetensors", "weight_dtype": "default"}}
            p["97"] = {"class_type": "MOSSBlockSwap", "inputs": {"model": cur, "donor": ["95", 0], "layers": layers,
                       "parts": parts.replace(";", ","), "include_other": False, "ratio": float(ratio)}}
            cur = ["97", 0]
        for j, spec in enumerate(a.solve):
            v = spec.split(",")
            p[str(90 + j * 100)] = {"class_type": "MOSSGramSolve", "inputs": {
                "model": cur, "sft_file": "moss_image_sft_bf16.safetensors", "base_file": "moss_image_base_bf16.safetensors",
                "flat_tag": v[4] if len(v) > 4 and v[4] else "flat", "num_tag": v[5].lstrip("!") if len(v) > 5 else "", "num_complement": len(v) > 5 and v[5].startswith("!"), "gen_tag": v[6] if len(v) > 6 else "gen", "layers": v[2], "parts": "qkv,o,ffn_up_gate,ffn_down",
                "mu": float(v[0]), "rho": float(v[1]), "strength": float(v[3]) if len(v) > 3 else 1.0, "device": "cuda:0"}}
            cur = [str(90 + j * 100), 0]
        if a.rope:
            c, r, s, cs = (float(x) for x in a.rope.split(","))
            p["96"] = {"class_type": "MOSSRopeOverride", "inputs": {"model": cur, "col_scale": c, "row_scale": r, "image_shift": s, "common_shift": cs}}
            cur = ["96", 0]
        p["14"]["inputs"]["model"] = cur
        if a.lora:
            f, s = a.lora.rsplit(":", 1)
            p["14"]["inputs"].update(lora_name=f, strength_model=float(s))
        # sampling chain: LoRAs (17) -> TSR (18) -> SLG (19) -> scheduler (20) / SamplerCustom (24)
        sampling_model = ["19", 0]
        if a.no_slg:
            del p["19"]
            sampling_model = ["18", 0]
        if a.no_tsr:
            if a.no_slg:
                sampling_model = ["17", 0]
            else:
                p["19"]["inputs"]["model"] = ["17", 0]
            del p["18"]
        p["20"]["inputs"]["model"] = sampling_model
        p["24"]["inputs"]["model"] = sampling_model
        if a.cfg is not None:
            p["24"]["inputs"]["cfg"] = a.cfg
        if a.euler:
            p["21"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        elif a.s_noise is not None:
            p["21"]["inputs"]["s_noise"] = a.s_noise
        if a.window:
            f, s, rng = a.window.rsplit(":", 2)
            lo, hi = (int(v) for v in rng.split("-"))
            p["70"] = {"class_type": "CreateHookLoraModelOnly", "inputs": {"lora_name": f, "strength_model": float(s)}}
            kf = ([(0.0, 0.0)] if lo > 1 else []) + [((lo - 1) / 25, 1.0)] + ([(hi / 25, 0.0)] if hi < 25 else [])
            prev = None
            for i, (pct, mult) in enumerate(kf):
                p[str(71 + i)] = {"class_type": "CreateHookKeyframe", "inputs": {"strength_mult": mult, "start_percent": pct}}
                if prev:
                    p[str(71 + i)]["inputs"]["prev_hook_kf"] = [prev, 0]
                prev = str(71 + i)
            p["75"] = {"class_type": "SetHookKeyframes", "inputs": {"hooks": ["70", 0], "hook_kf": [prev, 0]}}
            for cid, src in (("76", "7"), ("77", "8")):
                p[cid] = {"class_type": "ConditioningSetProperties", "inputs": {"cond_NEW": [src, 0], "strength": 1.0, "set_cond_area": "default", "hooks": ["75", 0]}}
            p["24"]["inputs"]["positive"], p["24"]["inputs"]["negative"] = ["76", 0], ["77", 0]
        p["26"]["inputs"]["filename_prefix"] = f"moss/drift/{a.arm}/{pid}__s{seed}"
        r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/prompt", json.dumps({"prompt": p, "front": a.front}).encode(), {"Content-Type": "application/json"})))
        assert not r.get("node_errors"), r
        n += 1
print(f"{a.arm}: queued {n}")

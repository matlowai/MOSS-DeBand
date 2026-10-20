#!/usr/bin/env python3
"""Train one LoRA chunk on MOSS-Image SFT through ComfyUI's TrainLoraNode, wait for it, and copy the result to
models/loras/moss_train/<name>.safetensors (so it can be resumed with --resume or rendered with drift_runner --lora).

  python3 workflows/moss/train_runner.py <name> --data <input folder> [--steps 600] [--lr 1e-4] [--rank 32] [--resume moss_train/<name>_<steps>_steps_00001_.safetensors]
      [--solve mu,rho,layers,strength,flat_tag,num_tag,gen_tag] [--url http://localhost:8189]
--solve trains on SFT with a MOSSGramSolve repair patched in (the LoRA learns on top of the repaired model).
The optimizer state does not carry over a --resume (TrainLoraNode only reloads the LoRA weights).
"""
import argparse, glob, json, os, shutil, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--data", required=True)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="[None]")
ap.add_argument("--solve")
ap.add_argument("--url", default="http://localhost:8189")
a = ap.parse_args()

p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_sft_bf16.safetensors", "weight_dtype": "default"}},
     "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors", "type": "moss_image", "device": "default"}},
     "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
     "4": {"class_type": "LoadImageTextDataSetFromFolder", "inputs": {"folder": a.data}},
     "5": {"class_type": "MakeTrainingDataset", "inputs": {"images": ["4", 0], "vae": ["3", 0], "clip": ["2", 0], "texts": ["4", 1]}},
     "8": {"class_type": "ResolutionBucket", "inputs": {"latents": ["5", 0], "conditioning": ["5", 1]}},
     "6": {"class_type": "TrainLoraNode", "inputs": {
         "model": ["1", 0], "latents": ["8", 0], "positive": ["8", 1], "batch_size": 1, "grad_accumulation_steps": 1, "steps": a.steps,
         "learning_rate": a.lr, "rank": a.rank, "optimizer": "AdamW", "loss_function": "MSE", "seed": a.seed, "training_dtype": "bf16",
         "lora_dtype": "bf16", "quantized_backward": False, "algorithm": "LoRA", "gradient_checkpointing": True, "checkpoint_depth": 2,
         "offloading": False, "existing_lora": a.resume, "bucket_mode": True, "bypass_mode": False}},
     "7": {"class_type": "SaveLoRA", "inputs": {"lora": ["6", 0], "prefix": f"loras/moss_train/{a.name}", "steps": ["6", 2]}}}
if a.solve:
    v = a.solve.split(",")
    p["90"] = {"class_type": "MOSSGramSolve", "inputs": {
        "model": ["1", 0], "sft_file": "moss_image_sft_bf16.safetensors", "base_file": "moss_image_base_bf16.safetensors",
        "flat_tag": v[4] or "flat", "gen_tag": v[6] or "gen", "layers": v[2], "parts": "qkv,o,ffn_up_gate,ffn_down",
        "mu": float(v[0]), "rho": float(v[1]), "strength": float(v[3]), "device": "cuda:0",
        "num_tag": v[5].lstrip("!"), "num_complement": v[5].startswith("!")}}
    p["6"]["inputs"]["model"] = ["90", 0]

r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})))
assert not r.get("node_errors"), r
pid = r["prompt_id"]
t0 = time.time()
while True:
    h = json.load(urllib.request.urlopen(f"{a.url}/history/{pid}"))
    if pid in h and h[pid]["status"].get("completed") is not None:
        break
    time.sleep(10)
st = h[pid]["status"]
assert st["status_str"] == "success", st
src = sorted(glob.glob(os.path.join(ROOT, "output", "loras", "moss_train", f"{a.name}_*_steps_*.safetensors")), key=os.path.getmtime)[-1]
dst = os.path.join(ROOT, "models", "loras", "moss_train", a.name + ".safetensors")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)
# TrainLoraNode reads the resume step count from "<name>_<steps>_steps_" in the file name, so keep that name for --resume
os.link(dst, os.path.join(os.path.dirname(dst), os.path.basename(src)))
print(f"{a.name}: trained {a.steps} steps in {time.time() - t0:.0f}s -> {dst} (resume as moss_train/{os.path.basename(src)})")

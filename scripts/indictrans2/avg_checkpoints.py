#!/usr/bin/env python3
"""
Average the LoRA adapter weights of the last N checkpoints (SWA / checkpoint averaging).
Usage: python3 scripts/avg_checkpoints.py \
           --ckpt-dir ckpts/it2_en-mni_prim \
           --last 3 \
           --out  ckpts/it2_en-mni_prim_avg/final
"""
import argparse, glob, os, re, shutil, torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True,
                help="Directory containing checkpoint-N subdirs")
ap.add_argument("--last",     type=int, default=3)
ap.add_argument("--out",      required=True,
                help="Output path for the averaged adapter")
a = ap.parse_args()

# Find all checkpoint-N subdirs, sort by step number
pattern = os.path.join(a.ckpt_dir, "checkpoint-*")
ckpts = sorted(
    glob.glob(pattern),
    key=lambda p: int(re.search(r"(\d+)$", p).group(1))
)[-a.last:]

assert ckpts, f"No checkpoints found under {a.ckpt_dir}"
print(f"Averaging {len(ckpts)} checkpoints: {[os.path.basename(c) for c in ckpts]}")

acc, keys = None, None
for c in ckpts:
    sd = load_file(os.path.join(c, "adapter_model.safetensors"))
    if acc is None:
        acc  = {k: v.clone().float() for k, v in sd.items()}
        keys = list(acc.keys())
    else:
        for k in keys:
            acc[k] += sd[k].float()

for k in keys:
    acc[k] /= len(ckpts)

os.makedirs(a.out, exist_ok=True)
save_file(
    {k: v.to(torch.float32) for k, v in acc.items()},
    os.path.join(a.out, "adapter_model.safetensors")
)
shutil.copy(
    os.path.join(ckpts[-1], "adapter_config.json"),
    os.path.join(a.out,     "adapter_config.json")
)
print(f"Averaged {len(ckpts)} checkpoints → {a.out}")

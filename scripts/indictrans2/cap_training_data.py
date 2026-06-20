#!/usr/bin/env python3
"""
Cap training data with fixed-ratio allocation across sources.
Official: keep all (up to cap). BT: fixed reservation. External: fill remainder.

Usage:
  python3 scripts/cap_training_data.py \
      --merged   data/train_ready/contra_en-as_all.tsv \
      --official data/train_ready/off_en-as.tsv \
      --out      data/train_ready/contra_en-as_capped.tsv \
      --cap      100000 \
      --bt-reserve 25000
"""
import argparse, random, os

ap = argparse.ArgumentParser()
ap.add_argument("--merged",      required=True)
ap.add_argument("--official",    required=True)
ap.add_argument("--out",         required=True)
ap.add_argument("--cap",         type=int, default=100_000)
ap.add_argument("--bt-reserve",  type=int, default=25_000,
                help="Rows reserved for BT regardless of other sources")
ap.add_argument("--bt-tag",      default="<bt> ")
ap.add_argument("--seed",        type=int, default=42)
a = ap.parse_args()

# Load official src sentences for bucket detection
official_src = set()
with open(a.official, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts:
            official_src.add(parts[0].strip())

official, external, bt = [], [], []
with open(a.merged, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        src = parts[0]
        if src.startswith(a.bt_tag):
            bt.append(line)
        elif src in official_src:
            official.append(line)
        else:
            external.append(line)

print(f"Buckets — official: {len(official)}  external: {len(external)}  bt: {len(bt)}")

random.seed(a.seed)
random.shuffle(external)
random.shuffle(bt)


OFFICIAL_CAP = a.cap - a.bt_reserve   # never let official crowd out BT entirely
off_take = min(len(official), OFFICIAL_CAP)

# Step 1: take all official (cap if somehow exceeds total cap)
off_take = min(len(official), a.cap)
selected_off = official[:off_take]
remaining = a.cap - off_take

# Step 2: reserve BT slots (only what's available and fits)
bt_take = min(len(bt), a.bt_reserve, remaining)
selected_bt = bt[:bt_take]
remaining -= bt_take

# Step 3: fill remainder with external
ext_take = min(len(external), remaining)
selected_ext = external[:ext_take]

selected = selected_off + selected_bt + selected_ext

# If total still under cap (sources were small), top up with remaining BT
if len(selected) < a.cap and bt_take < len(bt):
    extra_bt = bt[bt_take: bt_take + (a.cap - len(selected))]
    selected += extra_bt
    bt_take += len(extra_bt)

# Interleave all sources during training
random.shuffle(selected)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for line in selected:
        f.write(line + "\n")

print(f"Written {len(selected)} rows → {a.out}")
print(f"  official: {off_take} | external: {ext_take} | bt: {bt_take}")

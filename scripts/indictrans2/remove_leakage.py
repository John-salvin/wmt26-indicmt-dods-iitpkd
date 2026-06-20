#!/usr/bin/env python3
"""
scripts/remove_leakage.py
Strip any row from a training TSV where either column matches a gold
sentence (case-insensitive), then report the new count.

Usage:
  python3 scripts/remove_leakage.py \
      --train data/train_ready/contra_bodo-en_capped.tsv \
      --gold  data/wmt25_gold/en-bodo.csv \
      --gold-cols en bodo \
      --out   data/train_ready/contra_bodo-en_capped_clean.tsv
"""
import argparse, csv

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--gold", required=True)
ap.add_argument("--gold-cols", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

gold_set = set()
with open(a.gold, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        for col in a.gold_cols:
            val = (row.get(col) or "").strip().lower()
            if val:
                gold_set.add(val)

kept, dropped = 0, 0
with open(a.train, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        src, tgt = parts[0].strip(), parts[1].strip()
        if src.lower() in gold_set or tgt.lower() in gold_set:
            dropped += 1
            continue
        out.write(line if line.endswith("\n") else line + "\n")
        kept += 1

print(f"Kept {kept} rows, dropped {dropped} leaking rows - {a.out}")

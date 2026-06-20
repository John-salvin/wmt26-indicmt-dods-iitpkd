#!/usr/bin/env python3
"""
Check for overlap between monolingual data and the gold/test reference.
Usage: python3 scripts/leakage_check.py \
           --mono data/mono/mono_as.txt \
           --gold data/en-as.gold.csv \
           --gold-col as
"""
import argparse, csv, sys

ap = argparse.ArgumentParser()
ap.add_argument("--mono",     required=True)
ap.add_argument("--gold",     required=True)
ap.add_argument("--gold-col", required=True, help="Column name in gold CSV")
a = ap.parse_args()

mono_set = set(l.strip() for l in open(a.mono, encoding="utf-8") if l.strip())

overlap = 0
with open(a.gold, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ref = (row.get(a.gold_col) or "").strip()
        if ref in mono_set:
            overlap += 1
            print(f"LEAK: {ref[:80]}")

if overlap == 0:
    print("✅  No leakage found.")
else:
    print(f"⚠️  {overlap} overlapping sentences — remove them before BT!")
    sys.exit(1)

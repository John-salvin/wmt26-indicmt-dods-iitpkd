#!/usr/bin/env python3
"""
Convert WMT26 parallel CSV to headerless 2-col TSV (src TAB tgt).
Usage: python3 scripts/csv_to_tsv.py \
           --csv data/en-mni.train.csv \
           --src-name en --tgt-name mni \
           --out data/train_ready/off_en-mni.tsv
"""
import argparse, csv, sys

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

ap = argparse.ArgumentParser()
ap.add_argument("--csv",      required=True)
ap.add_argument("--src-name", required=True)
ap.add_argument("--tgt-name", required=True)
ap.add_argument("--out",      required=True)
a = ap.parse_args()

n = 0
with open(a.csv, newline="", encoding="utf-8") as f, \
     open(a.out, "w", encoding="utf-8") as o:
    for row in csv.DictReader(f):
        s = (row.get(a.src_name) or "").strip().replace("\t", " ")
        t = (row.get(a.tgt_name) or "").strip().replace("\t", " ")
        if s and t:
            o.write(f"{s}\t{t}\n")
            n += 1

print(f"Wrote {n} pairs → {a.out}")

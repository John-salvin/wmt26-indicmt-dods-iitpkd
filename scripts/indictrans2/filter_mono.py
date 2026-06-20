#!/usr/bin/env python3
"""
Length-filter, dedup, and cap a raw monolingual text file.
Usage: python3 scripts/filter_mono.py --inp data/mono/mono_as_raw.txt \
                                        --out data/mono/mono_as.txt \
                                        --cap 150000
"""
import argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("--inp",       required=True,  help="Raw mono file (one sentence/line)")
ap.add_argument("--out",       required=True,  help="Filtered output file")
ap.add_argument("--cap",       type=int, default=150000)
ap.add_argument("--min-words", type=int, default=3)
ap.add_argument("--max-words", type=int, default=80)
ap.add_argument("--seed",      type=int, default=42)
a = ap.parse_args()

seen, keep = set(), []
with open(a.inp, encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s or s in seen:
            continue
        nw = len(s.split())
        if a.min_words <= nw <= a.max_words:
            seen.add(s)
            keep.append(s)

random.seed(a.seed)
random.shuffle(keep)
keep = keep[:a.cap]

with open(a.out, "w", encoding="utf-8") as f:
    f.write("\n".join(keep) + "\n")

print(f"Wrote {len(keep)} sentences → {a.out}")



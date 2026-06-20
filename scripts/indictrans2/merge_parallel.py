#!/usr/bin/env python3
"""
Merge multiple parallel TSVs, deduplicate on (src, tgt),
length-filter, and cap at a maximum number of rows.

Usage:
  python3 scripts/merge_parallel.py \
      --lang as \
      --inp-dir data/parallel/as \
      --out    data/parallel/as/merged_en-as_raw.tsv \
      --cap    150000
"""
import argparse, os, random, glob

ap = argparse.ArgumentParser()
ap.add_argument("--lang",    required=True)
ap.add_argument("--inp-dir", required=True)
ap.add_argument("--out",     required=True)
ap.add_argument("--cap",     type=int, default=150_000)
ap.add_argument("--min-words-src", type=int, default=3)
ap.add_argument("--max-words-src", type=int, default=100)
ap.add_argument("--min-words-tgt", type=int, default=2)
ap.add_argument("--max-words-tgt", type=int, default=100)
ap.add_argument("--seed",    type=int, default=42)
a = ap.parse_args()

tsv_files = sorted(glob.glob(os.path.join(a.inp_dir, "*.tsv")))
tsv_files = [f for f in tsv_files if "merged" not in f]   # skip self

print(f"Merging {len(tsv_files)} files for {a.lang}:")
for f in tsv_files:
    print(f"  {f}")

seen, pairs = set(), []
for fpath in tsv_files:
    n_before = len(pairs)
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, tgt = parts[0].strip(), parts[1].strip()
            if not en or not tgt:
                continue
            key = (en.lower(), tgt.lower())
            if key in seen:
                continue
            nw_en  = len(en.split())
            nw_tgt = len(tgt.split())
            if (a.min_words_src <= nw_en  <= a.max_words_src and
                a.min_words_tgt <= nw_tgt <= a.max_words_tgt):
                # Simple length-ratio filter (avoids misaligned pairs)
                ratio = nw_en / nw_tgt
                if 0.3 <= ratio <= 3.0:
                    seen.add(key)
                    pairs.append((en, tgt))
    print(f"  +{len(pairs) - n_before} from {os.path.basename(fpath)}")

random.seed(a.seed)
random.shuffle(pairs)
# Priority order: keep highest-quality sources at top (already shuffled, so
# re-sort is not done; quality is managed by source selection above)
pairs = pairs[:a.cap]

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for en, tgt in pairs:
        f.write(f"{en}\t{tgt}\n")

print(f"\nTotal after dedup+filter: {len(pairs)} pairs → {a.out}")

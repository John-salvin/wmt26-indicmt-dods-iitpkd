#!/usr/bin/env python
"""
extract_mono.py
----------------
Extract one column from a parallel CSV/TSV as a plain one-sentence-per-line
monolingual .txt file, for use with `run_indictrans2.py backtranslate`.

Usage:
    python extract_mono.py --infile data/en-as.train.csv --col as --out data/bt/mono_as.txt
    python extract_mono.py --infile data/en-mni.train.csv --col mni --out data/bt/mono_mni.txt

Options:
    --dedup       drop exact-duplicate lines (recommended; parallel corpora
                   often contain repeated sentences)
    --shuffle     shuffle line order (uses a fixed seed for reproducibility)
"""
import argparse
import csv
import random
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--col", required=True, help="column name to extract (e.g. as, mni)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    delim = "\t" if args.infile.endswith((".tsv", ".tab")) else ","

    with open(args.infile, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if args.col not in reader.fieldnames:
            sys.exit(
                f"[extract_mono] column '{args.col}' not found in {args.infile}. "
                f"Available columns: {reader.fieldnames}"
            )
        lines = []
        for row in reader:
            s = (row.get(args.col) or "").strip()
            if s:
                lines.append(s)

    n_before = len(lines)
    if args.dedup:
        seen = set()
        deduped = []
        for s in lines:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        lines = deduped

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(lines)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[extract_mono] {args.infile} col='{args.col}': "
          f"{n_before} rows -> {len(lines)} lines written -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

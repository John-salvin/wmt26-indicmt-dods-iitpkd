#!/usr/bin/env python3
"""
Select monolingual sentences closest to the test-domain centroid.
Needs sentence-transformers and the offline mpnet model.
Usage: python3 scripts/domain_filter.py \
           --model models/mpnet \
           --query data/en-mni.gold.csv --query-col mni \
           --pool  data/mono/mono_mni.txt \
           --out   data/mono/mono_mni_dom.txt \
           --top   100000
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model",     required=True)
ap.add_argument("--query",     required=True)
ap.add_argument("--query-col", default=None,
                help="CSV column name; omit if query file is plain text")
ap.add_argument("--pool",      required=True)
ap.add_argument("--out",       required=True)
ap.add_argument("--top",       type=int, default=100000)
a = ap.parse_args()

from sentence_transformers import SentenceTransformer

def load_lines(path, col=None):
    if path.endswith(".csv") and col:
        import pandas as pd
        return pd.read_csv(path)[col].astype(str).dropna().tolist()
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

print("Loading model …")
model = SentenceTransformer(a.model)

print("Encoding query …")
q_vecs = model.encode(load_lines(a.query, a.query_col),
                       normalize_embeddings=True, batch_size=128)
centroid = q_vecs.mean(axis=0, keepdims=True)

print("Loading pool …")
pool = load_lines(a.pool)

print(f"Encoding {len(pool)} pool sentences …")
p_vecs = model.encode(pool, normalize_embeddings=True,
                       batch_size=128, show_progress_bar=True)

sims = (p_vecs @ centroid.T).ravel()
top_idx = np.argsort(-sims)[: a.top]

with open(a.out, "w", encoding="utf-8") as f:
    for i in top_idx:
        f.write(pool[i] + "\n")

print(f"Kept {min(a.top, len(pool))} domain-matched sentences → {a.out}")

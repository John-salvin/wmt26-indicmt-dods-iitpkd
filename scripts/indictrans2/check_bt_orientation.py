#!/usr/bin/env python3
"""
Quick sanity check: print the first 5 rows of a BT TSV and their detected scripts.
Helps confirm which column is English vs target language.
Usage: python3 scripts/check_bt_orientation.py data/bt/bt_en-mni.tsv
"""
import sys, unicodedata

SCRIPT_RANGES = {
    "Bengali/Assamese": (0x0980, 0x09FF),
    "Meitei Mayek":     (0xABC0, 0xABFF),
    "Devanagari":       (0x0900, 0x097F),
    "Latin":            (0x0041, 0x007A),
}

def detect_script(text):
    counts = {s: 0 for s in SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else "Unknown"

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        s0, s1 = parts[0][:60], parts[1][:60]
        print(f"Row {i}:")
        print(f"  col0 [{detect_script(s0)}]: {s0}")
        print(f"  col1 [{detect_script(s1)}]: {s1}")

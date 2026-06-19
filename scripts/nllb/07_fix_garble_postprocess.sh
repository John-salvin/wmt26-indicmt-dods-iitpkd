#!/bin/bash
# Run from repo root after activating the env described in docs/REPRODUCE_NLLB.md
set -e
cd /scratch/$USER/wmt26  # adjust to your cluster path

# Some outputs had a small number of repetitive/degenerate lines
# (e.g. "ti-no ti-no ti-no", "bwkhak bwkhak bwkhak").
# fix_garble.py detects them and regenerates only those lines with the same adapter.
# This is post-hoc and uses the SAME adapter as the original translate -- no data change.

python scripts/fix_garble.py \
  --adapter ckpts/nllb_en-trp/final \
  --src-csv data/test/en-trp.csv \
  --hyp     outputs/submit/DoDS-IITPKD_primary_en_to_trp.txt \
  --out     outputs/submit/DoDS-IITPKD_primary_en_to_trp.txt

python scripts/fix_garble.py \
  --adapter ckpts/nllb_trp-en/final \
  --src-csv data/test/trp-en.csv \
  --hyp     outputs/submit/DoDS-IITPKD_primary_trp_to_en.txt \
  --out     outputs/submit/DoDS-IITPKD_primary_trp_to_en.txt

python scripts/fix_garble.py \
  --adapter ckpts/nllb_en-kha_bt/final \
  --src-csv data/test/en-kha.csv \
  --hyp     outputs/submit/DoDS-IITPKD_primary_en_to_kha.txt \
  --out     outputs/submit/DoDS-IITPKD_primary_en_to_kha.txt

python scripts/fix_garble.py \
  --adapter ckpts/nllb_kha-en_bt/final \
  --src-csv data/test/kha-en.csv \
  --hyp     outputs/submit/DoDS-IITPKD_primary_kha_to_en.txt \
  --out     outputs/submit/DoDS-IITPKD_primary_kha_to_en.txt

python scripts/fix_garble.py \
  --adapter ckpts/nllb_en-kha_ext/final \
  --src-csv data/test/en-kha.csv \
  --hyp     outputs/submit/DoDS-IITPKD_contrastive_en_to_kha.txt \
  --out     outputs/submit/DoDS-IITPKD_contrastive_en_to_kha.txt

#!/bin/bash
# Run from repo root after activating the env described in docs/REPRODUCE_NLLB.md
set -e
cd /scratch/$USER/wmt26  # adjust to your cluster path

# Generate bt_kha-en.tsv from monolingual Khasi using kha->en baseline
sbatch run.sbatch src/run_nllb.py backtranslate \
  --infile data/bt/mono_kha.txt --src kha --tgt en --src-col kha \
  --adapter ckpts/nllb_kha-en/final \
  --out data/bt/bt_kha-en.tsv \
  --num-beams 4 --batch-size 16

# Generate bt_en-kha.tsv from monolingual English using en->kha baseline
sbatch run.sbatch src/run_nllb.py backtranslate \
  --infile data/bt/mono_en.txt --src en --tgt kha --src-col en \
  --adapter ckpts/nllb_en-kha/final \
  --out data/bt/bt_en-kha.tsv \
  --num-beams 4 --batch-size 16

# Generate bt_lus-en.tsv (mono_lus -> en using lus->en baseline)
sbatch run.sbatch src/run_nllb.py backtranslate \
  --infile data/bt/mono_lus.txt --src lus --tgt en --src-col lus \
  --adapter ckpts/nllb_lus-en/final \
  --out data/bt/bt_lus-en.tsv \
  --num-beams 4 --batch-size 16

# Generate bt_en-lus.tsv (mono_en -> lus using en->lus baseline)
sbatch run.sbatch src/run_nllb.py backtranslate \
  --infile data/bt/mono_en.txt --src en --tgt lus --src-col en \
  --adapter ckpts/nllb_en-lus/final \
  --out data/bt/bt_en-lus.tsv \
  --num-beams 4 --batch-size 16

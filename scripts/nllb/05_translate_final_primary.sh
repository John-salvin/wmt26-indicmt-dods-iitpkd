#!/bin/bash
# Run from repo root after activating the env described in docs/REPRODUCE_NLLB.md
set -e
cd /scratch/$USER/wmt26  # adjust to your cluster path

# en->kha
sbatch run.sbatch src/run_nllb.py translate \
  --infile data/test/en-kha.csv --src en --tgt kha --src-col src \
  --adapter ckpts/nllb_en-kha_bt/final \
  --out outputs/submit/DoDS-IITPKD_primary_en_to_kha.txt

# kha->en
sbatch run.sbatch src/run_nllb.py translate \
  --infile data/test/kha-en.csv --src kha --tgt en --src-col src \
  --adapter ckpts/nllb_kha-en_bt/final \
  --out outputs/submit/DoDS-IITPKD_primary_kha_to_en.txt

# en->lus
sbatch run.sbatch src/run_nllb.py translate \
  --infile data/test/en-lus.csv --src en --tgt lus --src-col src \
  --adapter ckpts/nllb_en-lus_bt/final \
  --out outputs/submit/DoDS-IITPKD_primary_en_to_lus.txt

# lus->en
sbatch run.sbatch src/run_nllb.py translate \
  --infile data/test/lus-en.csv --src lus --tgt en --src-col src \
  --adapter ckpts/nllb_lus-en_bt/final \
  --out outputs/submit/DoDS-IITPKD_primary_lus_to_en.txt

# en->trp  (official-only adapter)
sbatch run.sbatch src/run_nllb.py translate \
  --infile data/test/en-trp.csv --src en --tgt trp --src-col src \
  --adapter ckpts/nllb_en-trp/final \
  --out outputs/submit/DoDS-IITPKD_primary_en_to_trp.txt

# trp->en  (official-only adapter)
sbatch run.sbatch src/run_nllb.py translate \
  --infile data/test/trp-en.csv --src trp --tgt en --src-col src \
  --adapter ckpts/nllb_trp-en/final \
  --out outputs/submit/DoDS-IITPKD_primary_trp_to_en.txt

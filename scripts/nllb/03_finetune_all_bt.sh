#!/bin/bash
# Run from repo root after activating the env described in docs/REPRODUCE_NLLB.md
set -e
cd /scratch/$USER/wmt26  # adjust to your cluster path

# en->kha PRIMARY (BLEU 22.26)
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-kha.train.csv --train data/en-kha.train.csv \
  --train data/bt/bt_kha-en.tsv \
  --src en --tgt kha --src-col en --tgt-col kha \
  --out ckpts/nllb_en-kha_bt \
  --init-adapter ckpts/nllb_en-kha/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 2 --batch-size 8 --grad-accum 4

# kha->en PRIMARY (BLEU 15.97)
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-kha.train.csv --train data/en-kha.train.csv \
  --train data/bt/bt_en-kha.tsv \
  --src kha --tgt en --src-col kha --tgt-col en \
  --out ckpts/nllb_kha-en_bt \
  --init-adapter ckpts/nllb_kha-en/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 2 --batch-size 8 --grad-accum 4

# en->lus PRIMARY (BLEU 18.19)
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-lus.train.csv --train data/en-lus.train.csv \
  --train data/bt/bt_lus-en.tsv \
  --src en --tgt lus --src-col en --tgt-col lus \
  --out ckpts/nllb_en-lus_bt \
  --init-adapter ckpts/nllb_en-lus/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 2 --batch-size 8 --grad-accum 4

# lus->en PRIMARY (BLEU 22.46)
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-lus.train.csv --train data/en-lus.train.csv \
  --train data/bt/bt_en-lus.tsv \
  --src lus --tgt en --src-col lus --tgt-col en \
  --out ckpts/nllb_lus-en_bt \
  --init-adapter ckpts/nllb_lus-en/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 2 --batch-size 8 --grad-accum 4

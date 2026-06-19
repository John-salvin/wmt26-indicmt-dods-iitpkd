#!/bin/bash
# Run from repo root after activating the env described in docs/REPRODUCE_NLLB.md
set -e
cd /scratch/$USER/wmt26  # adjust to your cluster path

# en->kha baseline
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-kha.train.csv \
  --src en --tgt kha --src-col en --tgt-col kha \
  --out ckpts/nllb_en-kha \
  --dora --rslora --lora-r 64 \
  --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4

# kha->en baseline
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-kha.train.csv \
  --src kha --tgt en --src-col kha --tgt-col en \
  --out ckpts/nllb_kha-en \
  --dora --rslora --lora-r 64 \
  --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4

# en->lus baseline
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-lus.train.csv \
  --src en --tgt lus --src-col en --tgt-col lus \
  --out ckpts/nllb_en-lus \
  --dora --rslora --lora-r 64 \
  --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4

# lus->en baseline
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-lus.train.csv \
  --src lus --tgt en --src-col lus --tgt-col en \
  --out ckpts/nllb_lus-en \
  --dora --rslora --lora-r 64 \
  --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4

# en->trp baseline (PRIMARY for en->trp; no BT due to no allowed mono)
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-trp.train.csv \
  --src en --tgt trp --src-col English --tgt-col Kokborok \
  --out ckpts/nllb_en-trp \
  --dora --rslora --lora-r 64 \
  --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4

# trp->en baseline (PRIMARY for trp->en; no BT)
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-trp.train.csv \
  --src trp --tgt en --src-col Kokborok --tgt-col English \
  --out ckpts/nllb_trp-en \
  --dora --rslora --lora-r 64 \
  --lr 2e-4 --epochs 5 --batch-size 8 --grad-accum 4

#!/bin/bash
# Run from repo root after activating the env described in docs/REPRODUCE_NLLB.md
set -e
cd /scratch/$USER/wmt26  # adjust to your cluster path

# Prerequisites: external corpora prepared as TSVs under Contrastive/data/train_ready/
#   - off_kha-en.tsv, ext_kha-en.tsv, bt_kha-en.tsv (for kha->en_ext2)
#   - off_lus-en.tsv, bt_lus-en_v2.tsv  (for lus->en_ext; bt_lus-en_v2 is BT from IndicNECorp mono_lus_150k.txt)
# See docs/DATA_SOURCES.md for external corpus details (SMOL/GATITOS, Tatoeba, IndicNECorp 1.0).

# kha->en CONTRASTIVE (BLEU 17.57)  — best for this direction
sbatch run.sbatch src/run_nllb.py finetune \
  --train Contrastive/data/train_ready/off_kha-en.tsv \
  --train Contrastive/data/train_ready/ext_kha-en.tsv \
  --train Contrastive/data/train_ready/bt_kha-en.tsv \
  --src kha --tgt en --src-col 0 --tgt-col 1 \
  --out ckpts/nllb_kha-en_ext2 \
  --init-adapter ckpts/nllb_kha-en_bt/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 3 --batch-size 8 --grad-accum 4

# lus->en CONTRASTIVE (BLEU 24.55)  — best for this direction
sbatch run.sbatch src/run_nllb.py finetune \
  --train Contrastive/data/train_ready/off_lus-en.tsv \
  --train Contrastive/data/train_ready/bt_lus-en_v2.tsv \
  --src lus --tgt en --src-col 0 --tgt-col 1 \
  --out ckpts/nllb_lus-en_ext \
  --init-adapter ckpts/nllb_lus-en_bt/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 3 --batch-size 8 --grad-accum 4

# en->kha CONTRASTIVE (BLEU 21.60)
sbatch run.sbatch src/run_nllb.py finetune \
  --train Contrastive/data/train_ready/off_en-kha.tsv \
  --train Contrastive/data/train_ready/ext_en-kha.tsv \
  --train Contrastive/data/train_ready/bt_en-kha.tsv \
  --src en --tgt kha --src-col 0 --tgt-col 1 \
  --out ckpts/nllb_en-kha_ext \
  --init-adapter ckpts/nllb_en-kha_bt/final \
  --dora --rslora --lora-r 64 \
  --lr 1e-4 --epochs 3 --batch-size 8 --grad-accum 4

# en->trp CONTRASTIVE (BLEU 3.95)  — official + SMOL parallel
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-trp.train.csv data/smol_en-trp.csv \
  --src en --tgt trp --src-col English --tgt-col Kokborok \
  --out ckpts/nllb_en-trp_smol \
  --dora --rslora --lora-r 64 \
  --epochs 5 --batch-size 8 --grad-accum 4

# trp->en CONTRASTIVE (BLEU 9.47)  — official + SMOL + BT
sbatch run.sbatch src/run_nllb.py finetune \
  --train data/en-trp.train.csv data/smol_en-trp.csv data/bt_trp-en.csv \
  --src trp --tgt en --src-col Kokborok --tgt-col English \
  --out ckpts/nllb_trp-en_bt \
  --dora --rslora --lora-r 64 \
  --epochs 3 --batch-size 8 --grad-accum 4

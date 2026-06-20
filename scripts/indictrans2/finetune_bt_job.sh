#!/bin/bash
#SBATCH --job-name=ft_bt
#SBATCH --partition=gpu
#SBATCH --output=logs/ft_bt_%x_%j.out
#SBATCH --error=logs/ft_bt_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue

# Usage:
#   sbatch --job-name=ft_mni scripts/indictrans2/finetune_bt_job.sh \
#          en mni 5e-5 5 \
#          ckpts/it2_en-mni/final \       <- best existing adapter for this direction
#          ckpts/it2_en-mni_bt_ft

SRC=$1           # en
TGT=$2           # as / mni / bodo
LR=$3            # e.g. 5e-5
EPOCHS=$4        # e.g. 5
INIT_ADAPTER=$5  # path to the best existing forward adapter
OUT_NAME=$6      # output checkpoint directory name

BASE=/scratch/${USER}/wmt26
cd $BASE


BASE_MODEL=/scratch/$USER/wmt26/hf_cache/hub/models--ai4bharat--indictrans2-en-indic-1B/snapshots/10e65a9951a1e922cd109a95e8aba9357b62144b


echo "[ft] START $(date)  ${SRC}→${TGT}  lr=${LR}  epochs=${EPOCHS}"
echo "[ft] init_adapter=${INIT_ADAPTER}  out=ckpts/${OUT_NAME}"

python src/run_indictrans2.py finetune \
    --base "$BASE_MODEL" \
    --train  "data/train_ready/off_${SRC}-${TGT}.tsv" \
    --train  "data/train_ready/bt_${SRC}-${TGT}_tagged.tsv" \
    --src    "$SRC" \
    --tgt    "$TGT" \
    --src-col 0 \
    --tgt-col 1 \
    --out    "ckpts/${OUT_NAME}" \
    --init-adapter "${INIT_ADAPTER}" \
    --dora --rslora \
    --lora-r 64 --lora-alpha 128 \
    --lr    "${LR}" \
    --epochs "${EPOCHS}" \
    --batch-size 32 \
    --grad-accum 1 \
    --save-steps 300 \
    --val   "data/wmt25_gold/en-${TGT}.csv" \
    --val-src-col "${SRC}" --val-tgt-col "${TGT}" \
    --eval-steps 300 \
    --compute-bleu

echo "[ft] DONE $(date)"

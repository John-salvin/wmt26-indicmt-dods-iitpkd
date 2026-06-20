#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Usage: sbatch scripts/indictrans2/eval_job.sh <src> <tgt> <adapter_dir> <output_label>
# Example: sbatch scripts/indictrans2/eval_job.sh en mni ckpts/it2_en-mni_prim_avg/final prim_avg

SRC=$1
TGT=$2
ADAPTER=$3
LABEL=$4

BASE=/scratch/${USER}/wmt26
cd $BASE

BASE_MODEL=/scratch/$USER/wmt26/hf_cache/hub/models--ai4bharat--indictrans2-en-indic-1B/snapshots/10e65a9951a1e922cd109a95e8aba9357b62144b

python src/run_indictrans2.py translate \
    --base "$BASE_MODEL" \
    --infile    "data/wmt25_gold/en-${SRC}.csv" \
    --src       "$SRC" \
    --tgt       "$TGT" \
    --src-col   "$SRC" \
    --tgt-col   "$TGT" \
    --adapter   "$ADAPTER" \
    --out       "outputs/eval_${SRC}-${TGT}_${LABEL}.txt" \
    --score

echo "Eval done: outputs/eval_${SRC}-${TGT}_${LABEL}.txt"

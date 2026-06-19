#!/bin/bash
# =============================================================================
#  run.sbatch  ·  ONE SLURM wrapper for run_indictrans2.py and run_nllb.py
# =============================================================================
#  Prerequisites:
#    - Python 3.12.8 module (the one your venv was built from)
#    - your offline venv (no internet on the nodes)
#    - HF offline + HF_HOME pointing at your pre-downloaded model cache
#  Resume across the 12h wall-clock is handled by --requeue + `--resume auto`
#  (the script saves a checkpoint every --save-steps; on requeue it picks up the
#  latest one).
#
#  EDIT THESE TWO LINES to match your cluster paths:
#      VENV    -> the venv that has torch + peft + transformers installed
#      HF_HOME -> the dir that contains hub/models--facebook--nllb-200-*
#
#  EXAMPLES
#    # smoke-test fine-tune on the SMALLEST pair first:
#    sbatch run.sbatch src/run_nllb.py finetune \
#        --train data/en-trp.train.csv --src en --tgt trp \
#        --src-col English --tgt-col Kokborok --out ckpts/nllb_en-trp \
#        --dora --rslora --lora-r 64 --lr 2e-4 --epochs 5 \
#        --batch-size 8 --grad-accum 4
#
#    # translate a test file:
#    sbatch run.sbatch src/run_nllb.py translate \
#        --infile data/test/en-trp.csv --src en --tgt trp --src-col src \
#        --adapter ckpts/nllb_en-trp/final \
#        --out outputs/submit/DoDS-IITPKD_primary_en_to_trp.txt
# =============================================================================
#SBATCH --job-name=wmt26
#SBATCH --output=/scratch/%u/wmt26/logs/%x_%j.out
#SBATCH --error=/scratch/%u/wmt26/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11:30:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue

set -euo pipefail

# ----- EDIT ONCE -------------------------------------------------------------
VENV="/scratch/$USER/nllb_venv"
export HF_HOME="/scratch/$USER/wmt26/models"
# -----------------------------------------------------------------------------

cd "/scratch/$USER/wmt26"
mkdir -p logs

# Load modules (adjust names to match your cluster's module system)
module purge 2>/dev/null || true
module load python-3.12.8 2>/dev/null || true
module load cuda-12.4
source "$VENV/bin/activate"

# Offline mode: compute nodes have no internet; models load from HF_HOME cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

echo "=========================================="
echo "Host   : $(hostname)"
echo "Python : $(which python)  ($(python --version 2>&1))"
echo "VENV   : $VENV"
echo "HF_HOME: $HF_HOME"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
PY
echo "CMD    : python $@"
echo "=========================================="

python "$@"

echo "=========================================="
echo "Job step finished."

#!/bin/bash
# =============================================================================
#  run.sbatch  ·  ONE SLURM wrapper for run_indictrans2.py and run_nllb.py
# =============================================================================
#  Matches the env that already trains/evaluates on Madhava:
#    - Python 3.12.8 module (the one your venv was built from)
#    - your offline venv (no internet on the nodes)
#    - HF offline + HF_HOME pointing at your pre-downloaded model cache
#  Resume across the 12h wall-clock is handled by --requeue + `--resume auto`
#  (the script saves a checkpoint every --save-steps; on requeue it picks up the
#  latest one). No signal-forwarding tricks -- this mirrors your working setup.
#
#  EDIT THESE TWO LINES ONCE, then never again:
#      VENV    -> the venv that has torch + IndicTransToolkit working
#      HF_HOME -> the dir that contains hub/models--ai4bharat--indictrans2-*
#
#  EXAMPLES
#    # smoke-test fine-tune on the SMALLEST pair first (proven settings):
#    sbatch --account=kpal run.sbatch run_indictrans2.py finetune \
#        --train data/en-bodo.train.csv --src en --tgt bodo \
#        --src-col en --tgt-col bodo --out ckpts/it2_en-bodo \
#        --lora-r 16 --lora-alpha 32 --epochs 5 --batch-size 8 --grad-accum 4
#
#    # translate a test file (NO --rerank needed; plain beam = your 27.3 path):
#    sbatch --account=kpal run.sbatch run_indictrans2.py translate \
#        --infile data/test_en-bodo.csv --src en --tgt bodo --src-col en \
#        --adapter ckpts/it2_en-bodo/final --out outputs/DoDS-IITPKD_primary_en_to_bodo.txt
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
VENV="$HOME/wmt26_gpu"                       # <-- set to your working venv
export HF_HOME="/scratch/$USER/wmt26/hf_cache" # <-- set to your model cache dir
# -----------------------------------------------------------------------------

cd "/scratch/$USER/wmt26"
mkdir -p logs

# Python 3.12.8 module the venv was built from (dash, not slash). Harmless if the
# venv already works standalone; required if it depends on the module's libs.
module purge 2>/dev/null || true
module load python-3.12.8 2>/dev/null || true

source "$VENV/bin/activate"

# Offline: the nodes have no internet. Models load from HF_HOME cache only.
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
echo "CMD    : python src/$@"
echo "=========================================="

# Foreground run (mirrors your train.sbatch). On wall-clock kill, SLURM requeues
# and `--resume auto` resumes from the latest checkpoint.
python "src/$@"

echo "=========================================="
echo "Job step finished."

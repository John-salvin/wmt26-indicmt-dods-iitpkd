#!/bin/bash
#SBATCH --job-name=wmt26-train
#SBATCH --output=/scratch/%u/wmt26/logs/%x_%j.out
#SBATCH --error=/scratch/%u/wmt26/logs/%x_%j.err
#SBATCH --time=11:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /home/apps/compilers/anaconda3/2024/bin/activate
conda activate wmt26

export HF_HOME=/scratch/$USER/wmt26/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

CONFIG=${1:-configs/base.yaml}
OUTDIR=${2:-/scratch/$USER/wmt26/ckpts/default}

python src/train/train_dora.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --resume_from_checkpoint auto

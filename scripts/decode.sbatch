#!/bin/bash
#SBATCH --job-name=wmt26-decode
#SBATCH --output=/scratch/%u/wmt26/logs/%x_%j.out
#SBATCH --error=/scratch/%u/wmt26/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /home/apps/compilers/anaconda3/2024/bin/activate
conda activate wmt26

export HF_HOME=/scratch/$USER/wmt26/hf_cache
export TRANSFORMERS_OFFLINE=1

# Will be implemented in src/decode/rerank.py
python src/decode/rerank.py "$@"

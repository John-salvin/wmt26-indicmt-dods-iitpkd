#!/bin/bash
#SBATCH --job-name=wmt26-bt
#SBATCH --output=/scratch/%u/wmt26/logs/%x_%j.out
#SBATCH --error=/scratch/%u/wmt26/logs/%x_%j.err
#SBATCH --time=11:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /home/apps/compilers/anaconda3/2024/bin/activate
conda activate wmt26

export HF_HOME=/scratch/$USER/wmt26/hf_cache
export TRANSFORMERS_OFFLINE=1

# Args: base_model adapter_path mono_file out_file src_lang tgt_lang
python src/bt/forward_bt.py \
    --base_model "$1" \
    --adapter_path "$2" \
    --mono_file "$3" \
    --out_file "$4" \
    --src_lang "$5" \
    --tgt_lang "$6" \
    --batch_size 16 \
    --num_beams 5

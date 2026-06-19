# Scripts

Run these in order from the cluster working directory (`/scratch/$USER/wmt26`).
See `docs/REPRODUCE_NLLB.md` for the full reproduction guide.

## NLLB pipeline (`scripts/nllb/`)

| Script | Stage | What it does |
|---|---|---|
| `01_finetune_all_baselines.sh` | Stage 1 | Fine-tune on official data only (6 direction adapters; also serves as PRIMARY for en↔trp) |
| `02_backtranslate_all.sh` | Stage 2 prep | Generate BT TSVs for kha and lus using Stage 1 adapters |
| `03_finetune_all_bt.sh` | Stage 2 | BT-augmented adapters — PRIMARY systems for kha/lus directions |
| `04_finetune_contrastive.sh` | Stage 3 | Contrastive adapters using external corpora |
| `05_translate_final_primary.sh` | Inference | Translate test sets with the 6 primary adapters |
| `06_translate_final_contrastive.sh` | Inference | Translate test sets with the 6 contrastive adapters |
| `07_fix_garble_postprocess.sh` | Post-process | Detect and re-translate degenerate lines (kha + trp outputs) |

## Utility scripts

| Script | Purpose |
|---|---|
| `run.sbatch` | SLURM wrapper used by all jobs |
| `fix_garble.py` | Per-line garble detection and repair |

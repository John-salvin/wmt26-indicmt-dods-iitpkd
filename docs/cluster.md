# Madhava Cluster — Quick Reference

## Partitions

| Partition | Wall-clock | Nodes | What it's for |
|---|---|---|---|
| `standard` | 3 days | 36 CPU nodes | CPU jobs |
| `nht` | 3 days | 16 CPU (HT disabled) | CPU jobs needing no HT |
| `gpu` | **12 hours** | 4 × 2 H100 | Our training and inference |
| `hm` | 3 days | 4 × 1TB RAM | Big dedup, in-memory data ops |

## Storage

- `/home` — 50 GB quota, backed up, **code only**
- `/scratch` — 350 TB, no quota, NOT backed up, **data and checkpoints here**
- `/scratch/common` — shared models/datasets across users; check before re-downloading

## The 12-hour wall-clock pattern

Every training SLURM script must use `--signal=B:USR1@1800 --requeue`. SLURM sends SIGUSR1
30 minutes before kill; HuggingFace Trainer catches it, writes a checkpoint, exits cleanly.
On requeue, our trainer auto-detects the latest checkpoint and resumes.

## SLURM cheat sheet

```bash
sbatch scripts/train.sbatch configs/en_as_v0.yaml /scratch/$USER/wmt26/ckpts/en_as_v0
squeue --me
scancel <JOBID>
sacct -j <JOBID> --format=JobID,Elapsed,State,ExitCode
sinfo -p gpu
```

## Common mistakes

| Mistake | Fix |
|---|---|
| HF downloads to /home | Set `HF_HOME=/scratch/$USER/wmt26/hf_cache` |
| Checkpoints to /home | All `output_dir` must start with `/scratch/$USER/` |
| Compute node has no internet | Set `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` |
| Job killed at ~11 hr | Use the signal+requeue template |

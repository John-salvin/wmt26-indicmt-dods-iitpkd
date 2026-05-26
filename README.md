# WMT 2026 Indic MT — DoDS-IITPKD

Submission code for the [WMT 2026 Low-Resource Indic Language Translation shared task](http://www2.statmt.org/wmt26/).

Team: **DoDS-IITPKD** (IIT Palakkad)
Language pairs entered (6 pairs, 12 directions):

- **Category 1 (Moderate Data):** en↔as (Assamese), en↔lus (Mizo), en↔kha (Khasi), en↔mni (Manipuri)
- **Category 2 (Very Limited Data):** en↔bodo (Bodo), en↔trp (Kokborok)

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | YAML training and inference configs |
| `src/data/` | Data cleaning, dedup, language-ID filtering, semantic filtering |
| `src/train/` | Fine-tuning scripts (QLoRA + DoRA + rsLoRA) |
| `src/bt/` | Back-translation forward/backward and iteration orchestration |
| `src/decode/` | Beam search, MBR-COMET, CometKiwi reranking |
| `src/eval/` | Scoring (BLEU, chrF, TER, ROUGE-L, COMET) |
| `scripts/` | SLURM templates and one-shot helpers |
| `paper/` | LaTeX sources for the WMT system paper |
| `tests/` | Unit tests |
| `docs/` | Cluster, git, per-language notes |

## Quick start

See [`PLAN.md`](PLAN.md) for the full team plan, cluster setup, reading list, and timeline.

## Submission deadlines (WMT 2026)

- **June 12, 2026:** Test data released to registered participants
- **June 19, 2026 (AoE):** Run submission deadline
- **November 2026:** System paper (with EMNLP 2026 in Budapest)

## Citation

If using this code, please cite our WMT 2026 system paper (forthcoming).

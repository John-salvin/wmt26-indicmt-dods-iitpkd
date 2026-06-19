# WMT 2026 Low-Resource Indic Language Translation — DoDS-IITPKD

System description repository for the **DoDS-IITPKD** submission to the
[WMT 2026 Low-Resource Indic Language Translation shared task](http://www2.statmt.org/wmt26/).

**Language pairs entered (6 pairs, 12 directions):**
- en ↔ kha (Khasi), en ↔ lus (Mizo), en ↔ trp (Kokborok) — NLLB-based systems (this repo)
- en ↔ as (Assamese), en ↔ mni (Manipuri), en ↔ bodo (Bodo) — IndicTrans2-based systems (handled separately)

---

## System Overview

For the Khasi, Mizo, and Kokborok directions we fine-tune **NLLB-200-3.3B** using
**LoRA + DoRA + rsLoRA** (rank 64, α=16) in up to three stages:

| Stage | Data | Purpose |
|---|---|---|
| 1 | Official parallel only | Baseline adapter per direction |
| 2 | Official ×2 + back-translated synthetic | Primary adapter (kha/lus directions) |
| 3 | Stage 2 + external corpora (SMOL, Tatoeba, IndicNECorp) | Contrastive adapter |

Kokborok (trp) has no allowed monolingual data, so Stage 1 adapters serve as primaries
and Stage 3 adds only SMOL parallel data.

A post-hoc garble repair step (`scripts/fix_garble.py`) detects and re-translates
degenerate repetitive lines using the same adapter.

---

## Repository Layout

```
wmt26-indicmt-dods-iitpkd/
├── src/
│   └── run_nllb.py                  # Main pipeline: finetune / backtranslate / translate / score
├── scripts/
│   ├── run.sbatch                   # SLURM wrapper (all jobs)
│   ├── fix_garble.py                # Post-hoc repair for degenerate outputs
│   ├── README.md                    # How to use the scripts in order
│   └── nllb/
│       ├── 01_finetune_all_baselines.sh
│       ├── 02_backtranslate_all.sh
│       ├── 03_finetune_all_bt.sh
│       ├── 04_finetune_contrastive.sh
│       ├── 05_translate_final_primary.sh
│       ├── 06_translate_final_contrastive.sh
│       └── 07_fix_garble_postprocess.sh
├── configs/
│   ├── base.yaml                    # Shared LoRA defaults
│   └── nllb/
│       ├── lora_stage1_baseline.yaml
│       ├── lora_stage2_bt.yaml
│       └── lora_stage3_contrastive.yaml
├── docs/
│   ├── REPRODUCE_NLLB.md            # Step-by-step reproduction guide
│   ├── SUBMISSIONS.md               # Per-file table: adapter, data, BLEU
│   └── DATA_SOURCES.md              # Official + external corpora used
├── results/
│   └── nllb/                        # sacreBLEU dev outputs (text, committed)
├── requirements.txt                 # Pinned versions (CUDA 12.1 / H100)
└── paper/                           # LaTeX system paper
```

---

## Quick Start — Reproducing NLLB Submissions

See **[`docs/REPRODUCE_NLLB.md`](docs/REPRODUCE_NLLB.md)** for the full guide including
prerequisites and cluster setup.

```bash
# Run in order from the cluster working directory
bash scripts/nllb/01_finetune_all_baselines.sh   # Stage 1 — official-only adapters
bash scripts/nllb/02_backtranslate_all.sh         # BT data generation
bash scripts/nllb/03_finetune_all_bt.sh           # Stage 2 — BT-augmented (primaries)
bash scripts/nllb/04_finetune_contrastive.sh      # Stage 3 — external data (contrastives)
bash scripts/nllb/05_translate_final_primary.sh   # Final inference — primary
bash scripts/nllb/06_translate_final_contrastive.sh  # Final inference — contrastive
bash scripts/nllb/07_fix_garble_postprocess.sh    # Post-hoc garble repair
```

Outputs: `outputs/submit/DoDS-IITPKD_{primary,contrastive}_{src}_to_{tgt}.txt`

> Submission `.txt` files are **not** committed — this repo provides the recipe to regenerate them.

---

## Submission Summary

| Direction | Type | Adapter | Dev BLEU |
|---|---|---|---|
| en → kha | Primary | `nllb_en-kha_bt` (Stage 2) | 22.26 |
| kha → en | Primary | `nllb_kha-en_bt` (Stage 2) | 15.97 |
| kha → en | **Contrastive** ★ | `nllb_kha-en_ext2` (Stage 3) | 17.57 |
| en → lus | Primary | `nllb_en-lus_bt` (Stage 2) | 18.19 |
| lus → en | Primary | `nllb_lus-en_bt` (Stage 2) | 22.46 |
| lus → en | **Contrastive** ★ | `nllb_lus-en_ext` (Stage 3) | 24.55 |
| en → trp | Primary | `nllb_en-trp` (Stage 1) | — |
| trp → en | Primary | `nllb_trp-en` (Stage 1) | — |
| en → kha | Contrastive | `nllb_en-kha_ext` (Stage 3) | 21.60 |
| en → trp | Contrastive | `nllb_en-trp_smol` (Stage 3) | 3.95 |
| trp → en | **Contrastive** ★ | `nllb_trp-en_bt` (Stage 3) | 9.47 |
| en → lus | Contrastive | `nllb_en-lus` (Stage 1) | — |

★ Best system for that direction across primary and contrastive.
Full per-file details in [`docs/SUBMISSIONS.md`](docs/SUBMISSIONS.md).

---

## Dependencies

```bash
pip install -r requirements.txt
```

Validated on Python 3.10, CUDA 12.1, H100 GPUs. See `requirements.txt` for pinned versions.

---

## Citation

If you use this code or build on our system, please cite:

```bibtex
@inproceedings{pakray-etal-2025-findings,
  title     = {Findings of {WMT} 2025 Shared Task on Low-resource {I}ndic Languages Translation},
  author    = {Pakray, Partha and Krishna, Reddi Mohana and Pal, Santanu and Vetagiri, Advaitha
               and Dash, Sandeep Kumar and Maji, Arnab Kumar and Lyngdoh, Saralin A.
               and Laitonjam, Lenin and Jamatia, Anupam and Sambyo, Koj and Das, Ajit
               and Manna, Riyanka},
  booktitle = {Proceedings of the Tenth Workshop on Machine Translation (WMT)},
  pages     = {532--553},
  year      = {2025},
  month     = {November}
}

@inproceedings{pakray-etal-2024-findings,
  title     = {Findings of {WMT} 2024 Shared Task on Low-Resource {I}ndic Languages Translation},
  author    = {Pakray, Partha and Krishna, Reddi Mohana and Pal, Santanu and Vetagiri, Advaitha
               and Dash, Sandeep and Maji, Arnab Kumar and Das, Ajit and Laitonjam, Lenin
               and Manna, Riyanka},
  booktitle = {Proceedings of the Ninth Conference on Machine Translation},
  pages     = {654--668},
  year      = {2024}
}

@inproceedings{pal-etal-2023-findings,
  title     = {Findings of the {WMT} 2023 Shared Task on Low-Resource {I}ndic Language Translation},
  author    = {Pal, Santanu and Pakray, Partha and Laskar, Sahinur Rahman and Laitonjam, Lenin
               and Khenglawt, Vanlalmuansangi and Warjri, Sunita and Dadure, Pankaj Kundan
               and Dash, Sandeep Kumar},
  booktitle = {Proceedings of the Eighth Conference on Machine Translation (WMT)},
  pages     = {682--694},
  year      = {2023}
}

@article{kakum-etal-2023-nmt,
  title   = {Neural machine translation for limited resources {E}nglish-{N}yishi pair},
  author  = {Kakum, Nabam and Laskar, Sahinur Rahman and Sambyo, Koj and Pakray, Partha},
  journal = {S{\={a}}dhan{\={a}}},
  publisher = {Springer},
  year    = {2023}
}
```

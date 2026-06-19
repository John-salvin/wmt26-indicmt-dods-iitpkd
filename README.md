# WMT 2026 Low-Resource Indic Language Translation — DoDS-IITPKD

Submission code for the [WMT 2026 Low-Resource Indic Language Translation shared task](http://www2.statmt.org/wmt26/).

**Team:** DoDS-IITPKD (IIT Palakkad)

This repository covers the **NLLB-200 side** of the submission (Khasi, Mizo, Kokborok).
The IndicTrans2 side (Assamese, Manipuri, Bodo) is handled separately.

---

## Repository Layout

```
wmt26-indicmt-dods-iitpkd/
├── README.md
├── requirements.txt              # pinned versions (CUDA 12.1 / H100)
├── .gitignore                    # excludes ckpts/, data/, logs/, *.safetensors
├── src/
│   └── run_nllb.py               # PRIMARY pipeline: finetune / backtranslate / translate / score
├── scripts/
│   ├── run.sbatch                # SLURM wrapper (edit VENV + HF_HOME once)
│   ├── fix_garble.py             # post-hoc repair of degenerate output lines
│   ├── README.md                 # how to use the scripts in order
│   └── nllb/
│       ├── 01_finetune_all_baselines.sh
│       ├── 02_backtranslate_all.sh
│       ├── 03_finetune_all_bt.sh
│       ├── 04_finetune_contrastive.sh
│       ├── 05_translate_final_primary.sh
│       ├── 06_translate_final_contrastive.sh
│       └── 07_fix_garble_postprocess.sh
├── configs/
│   ├── base.yaml                 # shared LoRA defaults
│   └── nllb/
│       ├── lora_stage1_baseline.yaml
│       ├── lora_stage2_bt.yaml
│       └── lora_stage3_contrastive.yaml
├── docs/
│   ├── REPRODUCE_NLLB.md         # step-by-step reproduction guide
│   ├── SUBMISSIONS.md            # per-file adapter, data, pair counts, BLEU
│   └── DATA_SOURCES.md           # official + external corpora used
├── results/
│   └── nllb/                     # sacreBLEU evaluation outputs (text only)
└── tests/
    └── test_data_loader.py
```

**Not committed (generated at runtime):** `ckpts/`, `data/`, `logs/`, `outputs/submit/`, `*.safetensors`

---

## NLLB Reproduction Quickstart

Full details: [`docs/REPRODUCE_NLLB.md`](docs/REPRODUCE_NLLB.md)

```bash
# Install dependencies
pip install -r requirements.txt

# Place WMT 2026 official data under data/ and monolingual BT data under data/bt/
# Edit VENV and HF_HOME in scripts/run.sbatch (two lines at the top, once only)

bash scripts/nllb/01_finetune_all_baselines.sh      # Stage 1: official-only adapters
bash scripts/nllb/02_backtranslate_all.sh            # Stage 2 prep: BT TSVs
bash scripts/nllb/03_finetune_all_bt.sh              # Stage 2: BT-augmented (primaries)
bash scripts/nllb/04_finetune_contrastive.sh         # Stage 3: contrastive (external data)
bash scripts/nllb/05_translate_final_primary.sh      # Final inference: 6 primary outputs
bash scripts/nllb/06_translate_final_contrastive.sh  # Final inference: 6 contrastive outputs
bash scripts/nllb/07_fix_garble_postprocess.sh       # Post-hoc degenerate-line repair
```

Outputs land in `outputs/submit/DoDS-IITPKD_{primary,contrastive}_{src}_to_{tgt}.txt`.

---

## System Overview

We fine-tune **NLLB-200-3.3B** using **LoRA + DoRA + rsLoRA** (rank 64, α=16) in up to three stages:

| Stage | Data | Purpose |
|---|---|---|
| 1 | Official parallel only | Baseline adapter per direction |
| 2 | Official ×2 + self-generated BT | Primary adapter (kha/lus directions) |
| 3 | Stage 2 + external corpora (SMOL, Tatoeba, IndicNECorp) | Contrastive adapter |

Kokborok (trp) has no allowed monolingual data, so Stage 1 adapters serve as primaries
and Stage 3 adds only SMOL parallel data.

**Surrogate tag strategy:** Khasi (`kha`) and Kokborok (`trp`) lack native FLORES-200 tags.
We use `lus_Latn` as a surrogate (same Latin script). The LoRA adapter learns the actual
target language from training data.

---

## NLLB Language Pairs

> **Note on BLEU scores:** All BLEU/chrF figures below were computed against the
> **WMT 2025 gold-standard test set** (last year's released references), used as a
> development proxy. WMT 2026 test references are not yet public. All files under
> `results/nllb/` and every mention of "eval" or "dev BLEU" in this repository refer
> to this **WMT 2025 proxy evaluation**, not the WMT 2026 test set.

| Direction | Best adapter | Dev BLEU (WMT 2025 proxy) | System |
|---|---|---|---|
| en→kha (English→Khasi) | `nllb_en-kha_bt` | 22.26 | primary |
| en→lus (English→Mizo) | `nllb_en-lus_bt` | 18.19 | primary |
| en→trp (English→Kokborok) | `nllb_en-trp_smol` | 3.95 | contrastive |
| kha→en (Khasi→English) | `nllb_kha-en_ext2` | 17.57 | contrastive |
| lus→en (Mizo→English) | `nllb_lus-en_ext` | 24.55 | contrastive |
| trp→en (Kokborok→English) | `nllb_trp-en_bt` | 9.47 | contrastive |

Full per-file details (adapter, training data, pair counts): [`docs/SUBMISSIONS.md`](docs/SUBMISSIONS.md)

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key versions: `transformers==4.44.2`, `peft==0.12.0`, `accelerate==0.34.2`,
`sentencepiece==0.2.0`, `sacrebleu==2.4.3`, `datasets==2.21.0`. See `requirements.txt`
for the full pinned list (validated on Python 3.10, CUDA 12.1, H100 GPUs).

---

## Citation

If you use the training data or evaluation benchmarks from this repository, please cite
the WMT Low-Resource Indic Language Translation shared task papers:

```bibtex
@inproceedings{pakray-etal-2025-findings,
  title     = {Findings of {WMT} 2025 Shared Task on Low-Resource {I}ndic Languages Translation},
  author    = {Pakray, Partha and Krishna, Reddi Mohana and Pal, Santanu and
               Vetagiri, Advaitha and Dash, Sandeep Kumar and Maji, Arnab Kumar and
               Lyngdoh, Saralin A. and Laitonjam, Lenin and Jamatia, Anupam and
               Sambyo, Koj and Das, Ajit and Manna, Riyanka},
  booktitle = {Proceedings of the Tenth Workshop on Machine Translation (WMT)},
  pages     = {532--553},
  year      = {2025}
}

@inproceedings{pakray-etal-2024-findings,
  title     = {Findings of {WMT} 2024 Shared Task on Low-Resource {I}ndic Languages Translation},
  author    = {Pakray, Partha and Pal, Santanu and Vetagiri, Advaitha and
               Krishna, Reddi and Maji, Arnab Kumar and Dash, Sandeep and
               Laitonjam, Lenin and Lyngdoh, Sarah and Manna, Riyanka},
  booktitle = {Proceedings of the Ninth Conference on Machine Translation},
  pages     = {654--668},
  year      = {2024}
}

@inproceedings{pal-etal-2023-findings,
  title     = {Findings of the {WMT} 2023 Shared Task on Low-Resource {I}ndic Language Translation},
  author    = {Pal, Santanu and Pakray, Partha and Laskar, Sahinur Rahman and
               Laitonjam, Lenin and Khenglawt, Vanlalmuansangi and Warjri, Sunita and
               Dadure, Pankaj Kundan and Dash, Sandeep Kumar},
  booktitle = {Proceedings of the Eighth Conference on Machine Translation (WMT)},
  pages     = {682--694},
  year      = {2023}
}

@article{kakum-etal-2023-neural,
  title     = {Neural machine translation for limited resources {E}nglish-{N}yishi pair},
  author    = {Kakum, Nabam and Laskar, Sahinur Rahman and Sambyo, Koj and Pakray, Partha},
  journal   = {S{{\={a}}}dhan{{\={a}}}},
  publisher = {Springer},
  year      = {2023}
}
```

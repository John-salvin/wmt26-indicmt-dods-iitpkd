# wmt26-indicmt-dods-iitpkd

WMT 2026 Low-Resource Indic Machine Translation — DoDS, IIT Palakkad submission.

This repository covers the **NLLB-200 side** of the submission (Khasi, Mizo, Kokborok).  
The IndicTrans2 side (Assamese, Manipuri, Bodo) is handled separately by a teammate.

---

## Repository Layout

```
wmt26-indicmt-dods-iitpkd/
├── README.md
├── requirements.txt          # pinned versions
├── .gitignore                # excludes ckpts/, data/, logs/, *.safetensors
├── src/
│   └── run_nllb.py             # PRIMARY pipeline: finetune / backtranslate / translate / score
├── scripts/
│   ├── run.sbatch              # SLURM wrapper (edit VENV + HF_HOME once)
│   ├── fix_garble.py           # post-hoc repair of degenerate output lines
│   └── nllb/
│       ├── 01_finetune_all_baselines.sh
│       ├── 02_backtranslate_all.sh
│       ├── 03_finetune_all_bt.sh
│       ├── 04_finetune_contrastive.sh
│       ├── 05_translate_final_primary.sh
│       ├── 06_translate_final_contrastive.sh
│       └── 07_fix_garble_postprocess.sh
├── configs/nllb/
│   ├── lora_stage1_baseline.yaml
│   ├── lora_stage2_bt.yaml
│   └── lora_stage3_contrastive.yaml
├── docs/
│   ├── REPRODUCE_NLLB.md       # step-by-step reproduction guide
│   ├── SUBMISSIONS.md          # per-file adapter, data, BLEU details
│   └── DATA_SOURCES.md         # official + external corpora used
├── results/nllb/             # sacreBLEU dev outputs (text only, safe to commit)
└── paper/                    # LaTeX system description paper
```

**Not committed (generated at runtime):** `ckpts/`, `data/`, `logs/`, `outputs/submit/`, `*.safetensors`

---

## NLLB Reproduction Quickstart

Full details: [`docs/REPRODUCE_NLLB.md`](docs/REPRODUCE_NLLB.md)

```bash
# Install deps
pip install -r requirements.txt

# Place WMT 2026 official data under data/ and monolingual BT data under data/bt/
# Edit VENV and HF_HOME in scripts/run.sbatch (two lines, once only)

bash scripts/nllb/01_finetune_all_baselines.sh   # Stage 1: official-only adapters
bash scripts/nllb/02_backtranslate_all.sh         # Stage 2 prep: BT TSVs
bash scripts/nllb/03_finetune_all_bt.sh           # Stage 2: BT-augmented (primaries)
bash scripts/nllb/04_finetune_contrastive.sh      # Stage 3: contrastive (external data)
bash scripts/nllb/05_translate_final_primary.sh   # Final inference: 6 primary outputs
bash scripts/nllb/06_translate_final_contrastive.sh  # Final inference: 6 contrastive
bash scripts/nllb/07_fix_garble_postprocess.sh    # Post-hoc degenerate-line repair
```

Outputs land in `outputs/submit/DoDS-IITPKD_{primary,contrastive}_{src}_to_{tgt}.txt`.

---

## NLLB Language Pairs

| Direction | Best adapter | Best BLEU | System |
|---|---|---|---|
| en→kha (English→Khasi) | `nllb_en-kha_bt` | 22.26 | primary |
| en→lus (English→Mizo) | `nllb_en-lus_bt` | 18.19 | primary |
| en→trp (English→Kokborok) | `nllb_en-trp_smol` | 3.95 | contrastive |
| kha→en (Khasi→English) | `nllb_kha-en_ext2` | 17.57 | contrastive |
| lus→en (Mizo→English) | `nllb_lus-en_ext` | 24.55 | contrastive |
| trp→en (Kokborok→English) | `nllb_trp-en_bt` | 9.47 | contrastive |

**Surrogate tag strategy:** Khasi (`kha`) and Kokborok (`trp`) use the FLORES-200 tag `lus_Latn` (same Latin script). The LoRA adapter learns the actual target language from training data.

Full per-file details (adapter, training data, pair counts): [`docs/SUBMISSIONS.md`](docs/SUBMISSIONS.md)

---

## Requirements

```
transformers==4.44.2
tokenizers==0.19.1
peft==0.12.0
accelerate==0.34.2
sentencepiece==0.2.0
sacrebleu==2.4.3
datasets==2.21.0
pandas==2.2.2
```

See `requirements.txt` for the full pinned list.

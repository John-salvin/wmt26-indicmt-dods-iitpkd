# Reproducing the NLLB Submissions

## Prerequisites

- Cluster with at least one CUDA GPU (A100/V100 recommended, ≥40 GB VRAM for the 3.3B model)
- Python 3.10, CUDA-enabled PyTorch
- `transformers==4.44.2`, `tokenizers==0.19.1`, `peft==0.12.0`, `accelerate==0.34.2`,
  `sentencepiece==0.2.0`, `sacrebleu==2.4.3`, `datasets==2.21.0`, `pandas==2.2.2`
  (see `requirements.txt`)
- WMT 2026 official task data placed under `data/` as `en-kha.train.csv`, `en-lus.train.csv`,
  `en-trp.train.csv`, and test sets at `data/test/`
- Allowed monolingual data at `data/bt/mono_{en,kha,lus}.txt` (same-domain corpus used in
  the official training data, e.g. Bible-aligned text)
- For contrastive runs: external corpora (see `docs/DATA_SOURCES.md`)

## Pipeline (run in order)

```bash
# 1. Stage 1 — official-only baselines (also serves as the en->trp / trp->en PRIMARIES)
bash scripts/nllb/01_finetune_all_baselines.sh

# 2. Generate backtranslation TSVs using Stage 1 adapters
bash scripts/nllb/02_backtranslate_all.sh

# 3. Stage 2 — BT-augmented adapters (PRIMARIES for kha/lus directions)
bash scripts/nllb/03_finetune_all_bt.sh

# 4. Stage 3 — contrastive adapters using external corpora
bash scripts/nllb/04_finetune_contrastive.sh

# 5. Translate the test set with the 6 primary adapters
bash scripts/nllb/05_translate_final_primary.sh

# 6. Translate the test set with the 6 contrastive adapters
bash scripts/nllb/06_translate_final_contrastive.sh

# 7. Post-hoc repair of degenerate lines (uses SAME adapter; data-neutral)
bash scripts/nllb/07_fix_garble_postprocess.sh
```

Final 12 outputs land in `outputs/submit/DoDS-IITPKD_{primary,contrastive}_{src}_to_{tgt}.txt`.

## Notes

- Khasi (`kha`) and Kokborok (`trp`) use the FLORES-200 surrogate tag `kha_Latn` / `trp_Latn`
  respectively (same Latin script). The adapter learns the actual target language from data.
- All training uses LoRA + DoRA + rsLoRA with rank 64, α=16, dropout 0.05, bf16,
  effective batch 32 (8 × grad-accum 4), AdamW with 3% warmup.
- Beam size 4 for all decoding.
- Submission `.txt` files are **not** committed — use this recipe to regenerate them.
- Dev BLEU/chrF outputs (sacreBLEU text) are committed under `results/nllb/`.

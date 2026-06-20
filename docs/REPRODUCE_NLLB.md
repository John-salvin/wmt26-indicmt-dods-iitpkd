# Reproducing the NLLB Submissions

> **Note on evaluation:** BLEU/chrF scores reported in `docs/SUBMISSIONS.md` and in
> `results/nllb/WMT25 Gold Test Set Outputs/eval_*.txt` were computed against the
> **WMT 2025 gold-standard test set** as a development proxy. They are **not** WMT 2026
> test set scores.

## Prerequisites

- Cluster with at least one CUDA GPU (A100/V100 recommended, ≥40 GB VRAM for the 3.3B model)
- Python 3.10, CUDA-enabled PyTorch
- `transformers==4.44.2`, `tokenizers==0.19.1`, `peft==0.12.0`, `accelerate==0.34.2`,
  `sentencepiece==0.2.0`, `sacrebleu==2.4.3`, `datasets==2.21.0`, `pandas==2.2.2`
  (see `requirements.txt`)
- WMT 2026 official training data placed under `data/` as `en-kha.train.csv`,
  `en-lus.train.csv`, `en-trp.train.csv` (publicly released by task organizers)
- **Test sets** at `data/test/` — distributed exclusively to registered WMT 2026
  participants; obtain from the task organizers. Required only for final inference
  (steps 5–7) and evaluation. Training (steps 1–4) does not need the test set.
- Backtranslation monolingual files (`data/bt/mono_{en,kha,lus}.txt`) are **generated
  automatically** by `scripts/nllb/02_backtranslate_all.sh` from the training CSVs —
  these are not external data and are not committed to this repository.
- For contrastive runs: external corpora (see `docs/DATA_SOURCES.md`)

## Pipeline (run in order)

```bash
# 1. Stage 1 — official-only baselines (also serves as the en->trp / trp->en PRIMARIES)
bash scripts/nllb/01_finetune_all_baselines.sh

# 2. Generate backtranslation TSVs using Stage 1 adapters
#    (also extracts mono_{en,kha,lus}.txt from training CSVs into data/bt/)
bash scripts/nllb/02_backtranslate_all.sh

# 3. Stage 2 — BT-augmented adapters (PRIMARIES for kha/lus directions)
bash scripts/nllb/03_finetune_all_bt.sh

# 4. Stage 3 — contrastive adapters using external corpora
bash scripts/nllb/04_finetune_contrastive.sh

# 5. Translate the WMT 2026 test set with the 6 primary adapters
#    (requires data/test/ — participant-only)
bash scripts/nllb/05_translate_final_primary.sh

# 6. Translate the WMT 2026 test set with the 6 contrastive adapters
bash scripts/nllb/06_translate_final_contrastive.sh

# 7. Post-hoc repair of degenerate lines (uses SAME adapter; data-neutral)
bash scripts/nllb/07_fix_garble_postprocess.sh
```

The committed WMT 2026 test-set outputs are in `results/nllb/WMT26 Test Set Outputs/`.

## Notes

- Khasi (`kha`) and Kokborok (`trp`) use the FLORES-200 surrogate tag `lus_Latn`
  (same Latin script). The adapter learns the actual target language from data.
- All training uses LoRA + DoRA + rsLoRA with rank 64, α=16, dropout 0.05, bf16,
  effective batch 32 (8 × grad-accum 4), AdamW with 3% warmup.
- Beam size 4 for all decoding.
- Dev BLEU/chrF outputs (sacreBLEU text, WMT 2025 proxy) are committed under `results/nllb/WMT25 Gold Test Set Outputs/eval_*.txt`.
- Final WMT 2026 test-set submission files are committed under `results/nllb/WMT26 Test Set Outputs/`.

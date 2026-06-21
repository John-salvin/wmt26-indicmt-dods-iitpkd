# Reproducing the IndicTrans2 Submissions

> **Note on evaluation:** Dev-proxy BLEU figures for the IndicTrans2 side (WMT 2025
> gold as a development set) are committed in [`README_INDICTRANS2.md`](../README_INDICTRANS2.md#indictrans2-language-pairs).
> Reproduce them with `scripts/indictrans2/eval_job.sh <src> <tgt> <adapter_dir> <label>`
> once the WMT 2025 gold files are in place under `data/wmt25_gold/`.

## Prerequisites

- Cluster with at least one CUDA GPU (H100/A100 recommended)
- Python 3.10+, CUDA-enabled PyTorch
- `transformers==4.44.2`, `peft==0.12.0`, `accelerate==0.34.2`, `IndicTransToolkit==1.1.1`,
  `sacrebleu==2.4.3`, `datasets==2.21.0` (see `requirements.txt`)
- WMT 2026 official training data placed under `data/` as `en-as.train.csv`,
  `en-mni.train.csv`, `en-bodo.train.csv` (publicly released by task organizers)
- **Test sets** and WMT 2025 gold references — distributed exclusively to registered
  WMT 2026 participants; obtain from the task organizers. Required only for final
  inference and dev-proxy scoring, not for fine-tuning.
- Monolingual files for back-translation are extracted/filtered from official sources
  using `scripts/indictrans2/filter_mono.py` and `domain_filter.py`, and checked for
  leakage against the gold set with `leakage_check.py` / `remove_leakage.py` before use.
  They are not external data and are not committed to this repository.
- CometKiwi (`unbabel-comet`) and `sentence-transformers` (LaBSE filtering) are optional
  extras, only needed for `--rerank` / `--labse-filter`.

## Pipeline (run per direction)

Unlike the NLLB side, IndicTrans2 jobs are parameterized per language pair rather than
hardcoded into one script per stage — run each step once per direction (en→as, as→en,
en→mni, mni→en, en→bodo, bodo→en).

```bash
# 1. Edit VENV and HF_HOME in scripts/run.sbatch (two lines at the top, once only)

# 2. Stage 1 — official-only fine-tune (baseline adapter per direction)
python src/run_indictrans2.py finetune \
    --train data/en-as.train.csv --src en --tgt as \
    --src-col en --tgt-col as --out ckpts/it2_en-as \
    --lora-r 16 --lora-alpha 32 --epochs 5 --batch-size 16 --grad-accum 4 --lr 5e-5

# 3. (Optional) Generate back-translation pairs from allowed monolingual data
python src/run_indictrans2.py backtranslate \
    --infile data/mono_as.txt --src as --tgt en \
    --adapter ckpts/it2_as-en/final --out data/bt_as-en.tsv --labse-filter 0.75

# 4. Stage 2 — BT-augmented fine-tune, warm-started from the Stage 1 adapter
sbatch --job-name=ft_as scripts/indictrans2/finetune_bt_job.sh \
    en as 5e-5 5 ckpts/it2_en-as/final ckpts/it2_en-as_bt_ft

# 5. Stage 3 — contrastive fine-tune with official + BT + external corpora
#    (cap_training_data.py controls the fixed-ratio mix across sources)
python src/run_indictrans2.py finetune \
    --train data/train_ready/contrastive_en-as.tsv --src en --tgt as \
    --src-col 0 --tgt-col 1 --out ckpts/it2_en-as_contrastive \
    --init-adapter ckpts/it2_en-as_bt_ft/final \
    --lora-r 16 --lora-alpha 32 --epochs 3 --batch-size 16 --grad-accum 4 --lr 5e-5

# 6. (Optional) Average the last N checkpoints (SWA) over the LoRA adapter
python scripts/indictrans2/avg_checkpoints.py --adapter-dir ckpts/it2_en-as_contrastive

# 7. Translate the WMT 2026 test file (source only) -> submission txt, with CometKiwi rerank
python src/run_indictrans2.py translate \
    --infile data/test_en-as.csv --src en --tgt as --src-col en \
    --adapter ckpts/it2_en-as_contrastive/final \
    --out outputs/it2_en-as.txt --rerank

# 8. Score against WMT 2025 gold as a dev proxy
sbatch scripts/indictrans2/eval_job.sh en as ckpts/it2_en-as_contrastive/final final

# 9. Package final outputs into the submission zip
python src/run_indictrans2.py package --outputs-dir outputs --team DoDS-IITPKD
```

The committed WMT 2026 test-set outputs are in `results/indictrans2/WMT26 Test Set Outputs/`.

## Notes

- LoRA targets attention and FFN projections only (`q_proj`, `k_proj`, `v_proj`,
  `out_proj`, `encoder_attn.{q,k,v,out}_proj`, `fc1`, `fc2`) — **not** `lm_head` or the
  shared embeddings IndicTrans2 uses to route its 22-language tags. Corrupting those
  causes wrong-script or empty output in low-resource directions (en→as, en→mni).
- `lr=5e-5` is the only proven-stable learning rate; `2e-4` causes training collapse.
- All three native directions (as, mni, bodo) use the same LoRA config; see
  `configs/indictrans2/lora_stage{1,2,3}_*.yaml` for the exact per-stage settings.
- Dev BLEU/chrF++ outputs (WMT 2025 proxy) are committed under
  `results/indictrans2/WMT25 Gold Test Set Outputs/eval_*.txt`.
- Final WMT 2026 test-set submission files are committed under
  `results/indictrans2/WMT26 Test Set Outputs/`.

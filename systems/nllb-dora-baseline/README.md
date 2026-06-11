# System: NLLB-3.3B DoRA Baseline (en↔kha, en↔lus)

## Adapters (on cluster)
- ckpts/nllb_en-kha/final
- ckpts/nllb_kha-en/final
- ckpts/nllb_en-lus/final
- ckpts/nllb_lus-en/final

## Inference
bash systems/nllb-dora-baseline/infer.sh <test_csv> <direction>
Example: bash systems/nllb-dora-baseline/infer.sh data/test.csv kha-en

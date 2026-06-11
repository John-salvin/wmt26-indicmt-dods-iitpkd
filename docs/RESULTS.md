# WMT26 Eval Results — DoDS-IITPKD

## System A: NLLB-3.3B + DoRA/rsLoRA (no BT) — Baseline
Evaluated on WMT25 gold sets.

| Direction | BLEU  | chrF2 | vs WMT25 Winner |
|-----------|-------|-------|-----------------|
| en→lus    | 16.24 | 44.28 | ✅ above        |
| lus→en    | 22.20 | 57.60 | ✅ above        |
| en→kha    | 20.24 | 44.11 | ✅ above        |
| kha→en    | 10.23 | 34.77 | ❌ gap: −13.94  |

## System B: NLLB-3.3B + DoRA/rsLoRA + Back-Translation
*BT retrain jobs running on cluster (job 61444, 61445) — results TBD*

| Direction | BLEU | chrF2 | Δ vs Baseline |
|-----------|------|-------|---------------|
| kha→en    | TBD  | TBD   | TBD           |
| lus→en    | TBD  | TBD   | TBD           |

## Notes
- Khasi uses `lus_Latn` surrogate code (absent from NLLB-200)
- BT: 50K synthetic pairs, beam=4, LR=1e-4, 2 epochs, real data doubled

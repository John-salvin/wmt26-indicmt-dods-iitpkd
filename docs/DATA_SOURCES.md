# Data Sources

## Official Task Data (WMT 2026 Low-Resource Indic Translation)

- `data/en-kha.train.csv` — English ↔ Khasi parallel
- `data/en-lus.train.csv` — English ↔ Mizo parallel
- `data/en-trp.train.csv` — English ↔ Kokborok parallel (2,269 pairs)
- `data/test/{en-kha,kha-en,en-lus,lus-en,en-trp,trp-en}.csv` — released test sets

## Allowed Monolingual Data (used for BT in primaries)

Same-domain corpora aligned with the official training data distribution:

- `data/bt/mono_kha.txt` — Khasi monolingual
- `data/bt/mono_lus.txt` — Mizo monolingual
- `data/bt/mono_en.txt`  — English monolingual

These were used **only** for self-generated backtranslation; no external parallel data
was used in any primary system.

## External Corpora (used only in CONTRASTIVE systems)

- **IndicNECorp 1.0** — used for additional Mizo monolingual (`mono_lus_150k.txt`) to
  generate BT for the lus→en contrastive system (`nllb_lus-en_ext`).
- **Google SMOL / GATITOS** — used as external parallel data for the Kokborok contrastive
  systems (`nllb_en-trp_smol`, `nllb_trp-en_bt`) and the Khasi contrastive systems.
- **Tatoeba** — used as supplementary English–Khasi parallel data for the Khasi
  contrastive systems.

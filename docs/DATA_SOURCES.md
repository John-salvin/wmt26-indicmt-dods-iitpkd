# Data Sources

> **Note on evaluation:** All BLEU/chrF scores in this repository were computed against
> the **WMT 2025 gold-standard test set** used as a development proxy. They are **not**
> scores on the WMT 2026 test set.

## Official Task Data (WMT 2026 Low-Resource Indic Translation)

Source: WMT 2026 shared task official release.

- `data/en-kha.train.csv` — English ↔ Khasi parallel
- `data/en-lus.train.csv` — English ↔ Mizo parallel
- `data/en-trp.train.csv` — English ↔ Kokborok parallel (2,269 pairs)
- `data/test/{en-kha,kha-en,en-lus,lus-en,en-trp,trp-en}.csv` — released test sets

## Monolingual Data for Self-Generated Backtranslation (PRIMARY systems only)

**This is NOT external data.** These files are derived entirely from the official
WMT 2026 training distribution — they contain monolingual sentences drawn from the
same corpora used to build the official parallel data (e.g., Bible-aligned text),
with no material from outside the allowed data sources.

- `data/bt/mono_kha.txt` — Khasi monolingual sentences (from official corpus)
- `data/bt/mono_lus.txt` — Mizo monolingual sentences (from official corpus)
- `data/bt/mono_en.txt`  — English monolingual sentences (from official corpus)

These were used **only** for self-generated backtranslation (BT) to augment Stage 2
training. No external parallel or monolingual data was used in any primary system.
The BT process: Stage 1 adapter translates `mono_{tgt}.txt` → synthetic source, then
Stage 2 trains on `official ×2 + synthetic BT pairs`.

## External Corpora (CONTRASTIVE systems only)

The following external resources were used exclusively in the contrastive submissions.
No primary system uses any of these.

- **IndicNECorp 1.0** — additional Mizo monolingual (`mono_lus_150k.txt`) used to
  generate BT for the `lus→en` contrastive system (`nllb_lus-en_ext`).
- **Google SMOL / GATITOS** — external parallel data for Kokborok contrastive systems
  (`nllb_en-trp_smol`, `nllb_trp-en_bt`) and Khasi contrastive systems.
- **Tatoeba** — supplementary English–Khasi parallel data for Khasi contrastive systems.

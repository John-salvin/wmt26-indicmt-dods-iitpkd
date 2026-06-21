# WMT 2026 Low-Resource Indic Language Translation — DoDS-IITPKD

Submission code for the [WMT 2026 Low-Resource Indic Language Translation shared task](http://www2.statmt.org/wmt26/).

**Team:** DoDS-IITPKD (IIT Palakkad)

This file covers the **IndicTrans2 side** of the submission (Assamese, Manipuri, Bodo,
and the Kokborok surrogate). The NLLB-200 side (Khasi, Mizo, Kokborok) is documented in
[`README.md`](README.md).

---

## Repository Layout (IndicTrans2 side)

```
wmt26-indicmt-dods-iitpkd/
├── README_INDICTRANS2.md
├── requirements.txt                  # pinned versions (CUDA 12.1 / H100)
├── src/
│   └── run_indictrans2.py            # PRIMARY pipeline: finetune / backtranslate / translate / score / package
├── scripts/
│   ├── run.sbatch                    # SLURM wrapper (edit VENV + HF_HOME once); shared with the NLLB side
│   ├── fix_garble.py                 # post-hoc repair of degenerate output lines; shared with the NLLB side
│   └── indictrans2/
│       ├── finetune_bt_job.sh        # fine-tune / BT-stage fine-tune job, with sister-language warm start
│       ├── eval_job.sh               # translate + score against WMT 2025 gold (dev proxy)
│       ├── csv_to_tsv.py             # official CSV -> headerless 2-col TSV
│       ├── merge_parallel.py         # merge/dedup/cap multiple parallel TSVs
│       ├── cap_training_data.py      # fixed-ratio cap across official/BT/external sources
│       ├── filter_mono.py            # length-filter, dedup, cap raw monolingual text
│       ├── domain_filter.py          # select monolingual sentences nearest the test domain
│       ├── leakage_check.py          # check monolingual/gold overlap before BT
│       ├── remove_leakage.py         # strip leaked rows from a training TSV
│       ├── check_bt_orientation.py   # sanity-check which BT TSV column is which language
│       ├── avg_checkpoints.py        # checkpoint averaging (SWA) over LoRA adapters
│       ├── convert_newlines.py       # literal "\n" -> real newline repair
│       └── preprocess_mt_output.py   # MT output cleanup for scoring
└── results/
    └── indictrans2/
        ├── WMT26 Test Set Outputs/    # WMT 2026 test-set outputs (committed)
        │   ├── DoDS-IITPKD_primary_en_to_as.txt
        │   ├── DoDS-IITPKD_primary_as_to_en.txt
        │   ├── DoDS-IITPKD_primary_en_to_mni.txt
        │   ├── DoDS-IITPKD_primary_mni_to_en.txt
        │   ├── DoDS-IITPKD_primary_en_to_bodo.txt
        │   ├── DoDS-IITPKD_primary_bodo_to_en.txt
        │   ├── DoDS-IITPKD_contrastive_en_to_as.txt
        │   ├── DoDS-IITPKD_contrastive_en_to_mni.txt
        │   ├── DoDS-IITPKD_contrastive_en_to_bodo.txt
        │   ├── DoDS-IITPKD_constrained_as_to_en.txt
        │   ├── DoDS-IITPKD_constrained_mni_to_en.txt
        │   └── DoDS-IITPKD_constrained_bodo_to_en.txt
        └── WMT25 Gold Test Set Outputs/  # dev-proxy eval outputs against WMT 2025 gold (committed)
            ├── eval_en-as.txt
            ├── eval_en-as_parallel.txt
            ├── eval_en-mni.txt
            ├── eval_en-mni_mono.txt
            ├── eval_en-bodo.txt
            ├── eval_en-bodo_mono.txt
            ├── eval_as-en_bt.txt
            ├── eval_as-en_contrained.txt
            ├── eval_mni-en_bt.txt
            ├── eval_mni-en_constrained.txt
            ├── eval_bodo-en_bt.txt
            └── eval_bodo-en_constrained.txt
```

**Not committed (generated at runtime):** `ckpts/`, `data/`, `logs/`, `*.safetensors`

---

## Which Languages This Covers

IndicTrans2 natively supports the 22 *scheduled* Indian languages. Of our six WMT26
pairs that means:

| Language | Tag | Status |
|---|---|---|
| Assamese (as) | `asm_Beng` | native |
| Manipuri (mni) | `mni_Beng` | native |
| Bodo (bodo / brx) | `brx_Deva` | native |


---

## Data

### Training data (publicly available)

The WMT 2026 official parallel training files are released by the shared task organizers.
Obtain them from the official task page and place them under `data/`:

```
data/
├── en-as.train.csv      # English–Assamese
├── en-mni.train.csv     # English–Manipuri
├── en-bodo.train.csv    # English–Bodo
```

### Test sets (participant-only)

The WMT 2026 test inputs and the WMT 2025 gold-standard references are distributed
exclusively to registered participants and cannot be shared here. To run final inference
or reproduce dev-proxy scores you must obtain these files from the task organizers.

### Backtranslation monolingual files

Monolingual files used for back-translation are extracted/filtered from official sources using
(`scripts/indictrans2/filter_mono.py`, `domain_filter.py`) and checked for leakage against
the gold set (`leakage_check.py`, `remove_leakage.py`) before use. They are not committed
to this repository.

---

## IndicTrans2 Reproduction Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place WMT 2026 official training CSVs under data/
#    (test sets are participant-only — obtain from task organizers)
#    Optionally convert to headerless TSV with scripts/indictrans2/csv_to_tsv.py

# 3. Edit VENV and HF_HOME in scripts/run.sbatch (two lines at the top, once only)

# 4. Fine-tune per direction (main quality step)
python src/run_indictrans2.py finetune \
    --train data/en-as.train.csv --src en --tgt as \
    --src-col en --tgt-col as --out ckpts/it2_en-as

# 5. (Optional) Back-translate to augment training data with self-generated pairs
python src/run_indictrans2.py backtranslate \
    --infile data/mono_as.txt --src as --tgt en \
    --adapter ckpts/it2_as-en/final --out data/bt_as-en.tsv --labse-filter 0.75

# 6. Translate the WMT test file (source only) -> submission txt, with optional CometKiwi rerank
python src/run_indictrans2.py translate \
    --infile data/test_en-as.csv --src en --tgt as --src-col en \
    --adapter ckpts/it2_en-as/final --out outputs/it2_en-as.txt --rerank

# 7. Score against WMT 2025 gold as a dev proxy
python src/run_indictrans2.py translate \
    --infile data/en-as.test.csv --src en --tgt as \
    --src-col en --tgt-col as --adapter ckpts/it2_en-as/final \
    --out outputs/it2_en-as.txt --rerank --score

# 8. Package final outputs into the submission zip
python src/run_indictrans2.py package --outputs-dir outputs --team DoDS-IITPKD
```

The committed WMT 2026 test-set outputs are in `results/indictrans2/WMT26 Test Set Outputs/`.

GPU jobs on the cluster go through `scripts/run.sbatch`,
`scripts/indictrans2/finetune_bt_job.sh` (fine-tune / BT-stage fine-tune, with
sister-language adapter warm-start via `--init-adapter`), and
`scripts/indictrans2/eval_job.sh` (translate + score against gold).

---

## System Overview

We fine-tune **IndicTrans2-1B** (`ai4bharat/indictrans2-en-indic-1B` /
`ai4bharat/indictrans2-indic-en-1B`) using **LoRA** (optionally DoRA / rsLoRA) targeted at
attention and FFN projections only — `lm_head` and the shared embeddings that route
IndicTrans2's 22-language tags are left untouched, since corrupting them causes
wrong-script or empty output in low-resource directions.

Each direction proceeds through some subset of:

1. **Official-only fine-tune** — baseline adapter per direction.
2. **Back-translation augmentation** — self-generated pseudo-parallel pairs from allowed
   monolingual data, optionally LaBSE-filtered, then a follow-up fine-tune
   (`--init-adapter` warm-starts from the official-only adapter).
3. **Contrastive fine-tune** — official + BT + external public corpora
   (BPCC/PMINDIA/SMOL/etc.), with `cap_training_data.py` capping the data mix by source.
4. **Checkpoint averaging** (`avg_checkpoints.py`) over the last N checkpoints, where used.
5. **CometKiwi reference-free reranking** at inference time (`--rerank`), falling back to
   plain beam search if CometKiwi is unavailable.

**Wall-clock resumability:** training catches `SIGUSR1` (the Madhava 12-hour wall-clock
signal), checkpoints, and exits cleanly so SLURM can requeue and `--resume auto` picks up
the latest LoRA-rank-compatible checkpoint.

---

## IndicTrans2 Language Pairs

12 submission files: see `results/indictrans2/WMT26 Test Set Outputs/` for the committed
WMT 2026 test-set outputs (primary / contrastive / constrained, per the labels in each filename).

> **Note on BLEU scores:** Figures below were computed against the **WMT 2025
> gold-standard test set**, used as a development proxy (same convention as the NLLB side).
> WMT 2026 test references are not yet public. Reproduce them with
> `scripts/indictrans2/eval_job.sh <src> <tgt> <adapter_dir> <label>` once the WMT 2025 gold
> files are in place under `data/wmt25_gold/`.

| Direction | Adapter (example) | Dev BLEU (WMT 2025 proxy) | System |
|---|---|---|---|
| en→as (English→Assamese) | `ckpts/it2_en-as/final` | 26.30 | primary |
| as→en (Assamese→English) | `ckpts/it2_as-en_bt/final` | 34.20 | primary |
| en→mni (English→Manipuri) | `ckpts/it2_en-mni/final` | 7.40 | primary |
| mni→en (Manipuri→English) | `ckpts/it2_mni-en_bt/final` | 23.0 | primary |
| en→bodo (English→Bodo) | `ckpts/it2_en-bodo/final` | 36.6 | primary |
| bodo→en (Bodo→English) | `ckpts/it2_bodo-en_bt/final` | 36.7 | primary |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key versions: `transformers==4.44.2`, `peft==0.12.0`, `accelerate==0.34.2`,
`IndicTransToolkit==1.1.1`, `sacrebleu==2.4.3`, `datasets==2.21.0`. CometKiwi
(`unbabel-comet`) and `sentence-transformers` (LaBSE filtering) are optional extras,
imported lazily only inside `--rerank` / `--labse-filter` so the core fine-tune /
translate / score path works without them.

---

## Citation

See [`README.md`](README.md#citation) for the WMT shared task citation entries.

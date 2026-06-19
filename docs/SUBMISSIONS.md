# NLLB Submission Files

**Base model:** `facebook/nllb-200-3.3B`  
**LoRA config (all runs):** rank 64, α=16, DoRA + rsLoRA, dropout 0.05, bf16, effective batch 32  
**Inference:** beam search, beam size 4  

12 submission files total: 6 primary + 6 contrastive, covering 3 language pairs × 2 directions.  
★ = best system for that direction (by dev BLEU).

---

## Language Tag Assignment

| Language | Script | NLLB tag | Note |
|---|---|---|---|
| Mizo (lus) | Latin | `lus_Latn` | native |
| Khasi (kha) | Latin | `lus_Latn` | surrogate |
| Kokborok (trp) | Latin | `lus_Latn` | surrogate |

---

## 1. DoDS-IITPKD_primary_en_to_kha.txt ★

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_en-kha_bt/final` |
| Stage 1 init | `ckpts/nllb_en-kha/final` (official only, lr=2e-4, epochs=5) |
| Stage 2 data | `data/en-kha.train.csv` (×2 official) + `data/bt/bt_kha-en.tsv` (BT from mono_kha.txt) |
| Total pairs | 26,000 |
| Stage 2 hyperparams | lr=1e-4, epochs=2 |
| Data type | Constrained |
| Dev BLEU | **22.26** ★ |

## 2. DoDS-IITPKD_contrastive_en_to_kha.txt

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_en-kha_ext/final` |
| Init | `ckpts/nllb_en-kha_bt/final` (Stage 2) |
| Stage 3 data | official + SMOL/GATITOS + Tatoeba + BT |
| Stage 3 hyperparams | lr=1e-4, epochs=3 |
| Data type | Contrastive (external parallel) |
| Dev BLEU | 21.60 |

## 3. DoDS-IITPKD_primary_en_to_lus.txt ★

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_en-lus_bt/final` |
| Stage 1 init | `ckpts/nllb_en-lus/final` (official only, lr=2e-4, epochs=5) |
| Stage 2 data | `data/en-lus.train.csv` (×2 official) + `data/bt/bt_lus-en.tsv` (BT from mono_lus.txt) |
| Total pairs | 49,479 |
| Stage 2 hyperparams | lr=1e-4, epochs=2 |
| Data type | Constrained |
| Dev BLEU | **18.19** ★ |

## 4. DoDS-IITPKD_contrastive_en_to_lus.txt

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_en-lus/final` (Stage 1 baseline — no BT) |
| Training data | `data/en-lus.train.csv` (official only) |
| Data type | Constrained baseline (submitted in contrastive slot) |
| Dev BLEU | 16.24 |
| Note | BT system was stronger; promoted to primary. |

## 5. DoDS-IITPKD_primary_kha_to_en.txt

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_kha-en_bt/final` |
| Stage 1 init | `ckpts/nllb_kha-en/final` (official only, lr=2e-4, epochs=5) |
| Stage 2 data | `data/en-kha.train.csv` (×2 official) + `data/bt/bt_en-kha.tsv` (BT from mono_en.txt) |
| Total pairs | 42,336 |
| Stage 2 hyperparams | lr=1e-4, epochs=2 |
| Data type | Constrained |
| Dev BLEU | 15.97 |

## 6. DoDS-IITPKD_contrastive_kha_to_en.txt ★

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_kha-en_ext2/final` |
| Init | `ckpts/nllb_kha-en_bt/final` (Stage 2) |
| Stage 3 data | official + SMOL/GATITOS + Tatoeba + external BT |
| Stage 3 hyperparams | lr=1e-4, epochs=3 |
| Data type | Contrastive (external parallel) |
| Dev BLEU | **17.57** ★ |

## 7. DoDS-IITPKD_primary_lus_to_en.txt

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_lus-en_bt/final` |
| Stage 1 init | `ckpts/nllb_lus-en/final` (official only, lr=2e-4, epochs=5) |
| Stage 2 data | `data/en-lus.train.csv` (×2 official) + `data/bt/bt_en-lus.tsv` (BT from mono_en.txt) |
| Total pairs | 42,336 |
| Stage 2 hyperparams | lr=1e-4, epochs=2 |
| Data type | Constrained |
| Dev BLEU | 22.46 |

## 8. DoDS-IITPKD_contrastive_lus_to_en.txt ★

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_lus-en_ext/final` |
| Init | `ckpts/nllb_lus-en_bt/final` (Stage 2) |
| Stage 3 data | `off_lus-en.tsv` + `bt_lus-en_v2.tsv` (BT from IndicNECorp 1.0 mono_lus_150k.txt) |
| Total pairs | 143,535 |
| Stage 3 hyperparams | lr=1e-4, epochs=3 |
| Data type | Contrastive (IndicNECorp 1.0 external monolingual for BT) |
| Dev BLEU | **24.55** ★ |

## 9. DoDS-IITPKD_primary_en_to_trp.txt

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_en-trp/final` |
| Training data | `data/en-trp.train.csv` (official only) |
| Total pairs | 2,269 |
| Hyperparams | lr=2e-4, epochs=5 |
| Data type | Constrained (no allowed monolingual for trp) |
| Note | 366 garbled lines repaired post-hoc via `fix_garble.py` (same adapter, data-neutral) |

## 10. DoDS-IITPKD_contrastive_en_to_trp.txt ★

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_en-trp_smol/final` |
| Training data | `data/en-trp.train.csv` (official) + `data/smol_en-trp.csv` (SMOL/GATITOS) |
| Total pairs | 12,395 |
| Hyperparams | lr=2e-4, epochs=5 |
| Data type | Contrastive (SMOL/GATITOS external parallel) |
| Dev BLEU | **3.95** ★ |

## 11. DoDS-IITPKD_primary_trp_to_en.txt

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_trp-en/final` |
| Training data | `data/en-trp.train.csv` (official only) |
| Total pairs | 2,269 |
| Hyperparams | lr=2e-4, epochs=5 |
| Data type | Constrained (no allowed monolingual for trp) |
| Dev BLEU | 1.50 / chrF++ 18.39 |
| Note | Low BLEU expected at 2,269 pairs; garbled lines repaired via `fix_garble.py` |

## 12. DoDS-IITPKD_contrastive_trp_to_en.txt ★

| Field | Value |
|---|---|
| Adapter | `ckpts/nllb_trp-en_bt/final` |
| Training data | `data/en-trp.train.csv` + `data/smol_en-trp.csv` + `data/bt_trp-en.csv` (BT from english_mono_bt.txt) |
| Total pairs | 24,398 |
| Hyperparams | lr=1e-4, epochs=3 |
| Data type | Contrastive (SMOL/GATITOS external + BT augmentation) |
| Dev BLEU | **9.47** ★ |

---

## Best System per Direction

| Direction | Best adapter | Best BLEU | System |
|---|---|---|---|
| en→kha | `nllb_en-kha_bt` | 22.26 | primary |
| en→lus | `nllb_en-lus_bt` | 18.19 | primary |
| en→trp | `nllb_en-trp_smol` | 3.95 | contrastive |
| kha→en | `nllb_kha-en_ext2` | 17.57 | contrastive |
| lus→en | `nllb_lus-en_ext` | 24.55 | contrastive |
| trp→en | `nllb_trp-en_bt` | 9.47 | contrastive |

All primary systems are constrained (official data + self-generated BT from allowed monolingual).  
All contrastive systems use external data (IndicNECorp 1.0, SMOL/GATITOS, Tatoeba).  
Submission `.txt` files are **not** committed — use `docs/REPRODUCE_NLLB.md` to regenerate them.

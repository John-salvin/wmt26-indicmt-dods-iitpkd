# NLLB Submission Files

12 submission files total: 6 primary + 6 contrastive, covering 3 language pairs × 2 directions.

★ = best system for that direction (submitted as primary)

## Primary Systems

| File | Direction | Adapter | Training data | Dev BLEU | Dev chrF |
|---|---|---|---|---|---|
| `DoDS-IITPKD_primary_en_to_kha.txt` ★ | en→kha | `nllb_en-kha_bt` | official ×2 + BT | 22.26 | — |
| `DoDS-IITPKD_primary_kha_to_en.txt` | kha→en | `nllb_kha-en_bt` | official ×2 + BT | 15.97 | — |
| `DoDS-IITPKD_primary_en_to_lus.txt` ★ | en→lus | `nllb_en-lus_bt` | official ×2 + BT | 18.19 | — |
| `DoDS-IITPKD_primary_lus_to_en.txt` | lus→en | `nllb_lus-en_bt` | official ×2 + BT | 22.46 | — |
| `DoDS-IITPKD_primary_en_to_trp.txt` ★ | en→trp | `nllb_en-trp` | official only | — | — |
| `DoDS-IITPKD_primary_trp_to_en.txt` | trp→en | `nllb_trp-en` | official only | — | — |

## Contrastive Systems

| File | Direction | Adapter | Training data | Dev BLEU | Dev chrF |
|---|---|---|---|---|---|
| `DoDS-IITPKD_contrastive_en_to_kha.txt` | en→kha | `nllb_en-kha_ext` | official + SMOL + BT (Stage 3) | 21.60 | — |
| `DoDS-IITPKD_contrastive_kha_to_en.txt` ★ | kha→en | `nllb_kha-en_ext2` | official + Tatoeba + SMOL + BT (Stage 3) | 17.57 | — |
| `DoDS-IITPKD_contrastive_en_to_lus.txt` | en→lus | `nllb_en-lus` | official only (Stage 1 baseline) | — | — |
| `DoDS-IITPKD_contrastive_lus_to_en.txt` ★ | lus→en | `nllb_lus-en_ext` | official + IndicNECorp BT (Stage 3) | 24.55 | — |
| `DoDS-IITPKD_contrastive_en_to_trp.txt` | en→trp | `nllb_en-trp_smol` | official + SMOL | 3.95 | — |
| `DoDS-IITPKD_contrastive_trp_to_en.txt` ★ | trp→en | `nllb_trp-en_bt` | official + SMOL + BT | 9.47 | — |

## Notes

- chrF scores (—) should be filled in from `results/nllb/eval_*.txt` sacreBLEU outputs.
- The ★ marks indicate the best system per direction across primary and contrastive.
- For en→trp and trp→en primaries, dev BLEU scores are in `results/nllb/eval_en-trp.txt`
  and `results/nllb/eval_trp-en.txt`.
- No BT was generated for trp directions due to the absence of allowed monolingual Kokborok.

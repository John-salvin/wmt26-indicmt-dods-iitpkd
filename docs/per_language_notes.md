# Per-language quirks and notes

## Assamese (as) — Bengali script

- Use NFC Unicode normalization (NOT NFKC — NFKC conflates Bengali/Assamese chars).
- `sacrebleu` tokenizer: `flores200`.
- Watch for the ROUGE-L=0.007 bug from our 2025 submission — likely an IndicProcessor whitespace issue.
- NLLB code: `asm_Beng`. IndicTrans2 code: `asm_Beng`.

## Mizo (lus) — Roman script

- Tonal language, Sino-Tibetan.
- Bible corpus is large and useful (LushaiBible ~31k verses).
- JW300 has a Mizo subset.
- NLLB code: `lus_Latn`.

## Khasi (kha) — Roman script

- Austroasiatic, with apostrophes and diacritics (ï, ñ).
- May need ByT5 backup for character-level robustness.
- Register `<kha_Latn>` vocab token if not present in NLLB.
- NLLB code: `kha_Latn` (verify).

## Manipuri (mni) — Bengali script

- Tonal, agglutinative.
- IndicTrans2 has native support; use it primarily.
- WMT26 also has a separate `en-mni-Mtei` pair (Meitei Mayek script) — we are NOT entering it.
- NLLB code: `mni_Beng`.

## Bodo (brx) — Devanagari

- Our 2025 win. Defend.
- Tibeto-Burman.
- IndicTrans2 with IndicProcessor `<brx_Deva>` worked best.
- NLLB code: `brx_Deva` (verify).

## Kokborok (trp) — Roman or Bengali script

- Tibeto-Burman (Bodo-Garo branch), close relative of Bodo.
- Pivot strategy: train joint with Bodo, oversample with T=10.
- Tiny corpus (~2.3k in 2025). Mostly bible/wiki.
- Codes: `trp_Latn`, `trp_Beng` (likely not in NLLB by default).

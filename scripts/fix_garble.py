#!/usr/bin/env python3
"""
fix_garble.py  --  surgically repair ONLY the degenerate-repetition lines in a
finished NLLB submission file, then splice them back. Every other line stays
byte-for-byte identical, and the line count never changes.

It re-decodes the broken lines with repetition_penalty + no_repeat_ngram_size
(the only real fix for decode loops), and it SELF-CHECKS each new line: if the
re-decode is still garbled, it keeps the OLD line rather than risk making things
worse. So the worst case is "no change", never "made it worse".

RUN THIS ON THE CLUSTER, inside your nllb_venv, from /scratch/.../wmt26 .
It reuses your trained adapter + the same NLLB base, so output style matches
the rest of the file. No internet needed (offline flags are set below).

Examples (en->trp is the only direction with garble; run once per file):

  # PRIMARY en->trp  (adapter = nllb_en-trp_smol)
  python3 fix_garble.py \
    --adapter ckpts/nllb_en-trp_smol/final \
    --src-csv data/test/en-trp.csv \
    --hyp     outputs/submit/DoDS-IITPKD_primary_en_to_trp.txt \
    --out     outputs/submit/DoDS-IITPKD_primary_en_to_trp.fixed.txt

  # CONTRASTIVE en->trp  (adapter = nllb_en-trp_bt)
  python3 fix_garble.py \
    --adapter ckpts/nllb_en-trp_bt/final \
    --src-csv data/test/en-trp.csv \
    --hyp     outputs/submit/DoDS-IITPKD_contrastive_en_to_trp.txt \
    --out     outputs/submit/DoDS-IITPKD_contrastive_en_to_trp.fixed.txt

Then eyeball the printed SRC/OLD/NEW, confirm `wc -l` is still 1000, and if happy
replace the originals with the .fixed.txt files.
"""
import argparse, csv, json, os, re, sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# ---- repetition detectors (same logic as the QC audit) -------------------
def max_consec_run(s):
    t = s.split(); best = cur = 1 if t else 0
    for i in range(1, len(t)):
        if t[i] == t[i - 1]:
            cur += 1; best = max(best, cur)
        else:
            cur = 1
    return best


def max_alt_bigram_run(s):
    t = s.split(); best = 0
    for i in range(len(t) - 3):
        a, b = t[i], t[i + 1]; k = 0; j = i
        while j + 1 < len(t) and t[j] == a and t[j + 1] == b:
            k += 1; j += 2
        best = max(best, k)
    return best


def is_garbled(s, thr=3):
    return max_consec_run(s) >= thr or max_alt_bigram_run(s) >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to the LoRA adapter dir (…/final)")
    ap.add_argument("--src-csv", required=True, help="source CSV with a 'src' column (data/test/en-trp.csv)")
    ap.add_argument("--hyp", required=True, help="the finished submission .txt to repair")
    ap.add_argument("--out", required=True, help="where to write the repaired copy")
    ap.add_argument("--src-code", default="eng_Latn")
    ap.add_argument("--tgt-code", default="lus_Latn", help="surrogate FLORES code (trp -> lus_Latn)")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=256)
    a = ap.parse_args()

    # base model is recorded inside the adapter config -> no guessing
    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    base = cfg.get("base_model_name_or_path")
    print(f"[load] base={base}\n[load] adapter={a.adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    tok.src_lang = a.src_code
    forced_bos = tok.convert_tokens_to_ids(a.tgt_code)

    # source rows (DictReader skips the 'src' header automatically)
    rows = list(csv.DictReader(open(a.src_csv, encoding="utf-8")))
    if not rows:
        sys.exit("[ABORT] source CSV is empty")
    col = "src" if "src" in rows[0] else list(rows[0].keys())[0]

    # hypothesis lines (no header); preserve trailing newline
    raw = open(a.hyp, encoding="utf-8").read()
    hyp = raw.split("\n")
    trailing = bool(hyp and hyp[-1] == "")
    if trailing:
        hyp = hyp[:-1]

    if len(hyp) != len(rows):
        sys.exit(f"[ABORT] {a.hyp} has {len(hyp)} lines but {a.src_csv} has {len(rows)} rows -- "
                 "wrong source file? aborting so nothing gets misaligned.")

    bad = [i for i, line in enumerate(hyp) if is_garbled(line)]
    print(f"[scan] {len(bad)} garbled line(s) detected: {[i + 1 for i in bad]}", flush=True)
    if not bad:
        print("[done] nothing to fix; writing an identical copy.")
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(raw if trailing else "\n".join(hyp))
        return

    @torch.no_grad()
    def redecode(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=a.max_length).to(dev)
        g = model.generate(
            **enc,
            forced_bos_token_id=forced_bos,
            num_beams=a.num_beams,
            no_repeat_ngram_size=a.no_repeat_ngram_size,
            repetition_penalty=a.repetition_penalty,
            max_length=a.max_length,
        )
        return tok.batch_decode(g, skip_special_tokens=True)[0].strip()

    fixed = list(hyp)
    still_bad = []
    for i in bad:
        src = rows[i][col].strip()
        old = hyp[i]
        new = redecode(src)
        ok = not is_garbled(new)
        print(f"\n--- line {i + 1} ---")
        print(f"SRC: {src[:100]}")
        print(f"OLD: {old[:100]}")
        print(f"NEW: {new[:100]}    [{'OK, replacing' if ok else 'STILL GARBLED -> keeping OLD'}]")
        if ok:
            fixed[i] = new
        else:
            still_bad.append(i + 1)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed))
        if trailing:
            f.write("\n")

    print(f"\n[done] wrote {a.out}  ({len(fixed)} lines)")
    print(f"[done] repaired {len(bad) - len(still_bad)} / {len(bad)}; "
          f"still-garbled-kept-old: {still_bad or 'none'}")


if __name__ == "__main__":
    main()

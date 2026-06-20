#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 run_indictrans2.py  ·  DoDS-IITPKD · WMT 2026 Low-Resource Indic MT
================================================================================
ONE self-contained file. No other project file is imported. Everything the
IndicTrans2 system does -- fine-tune, back-translate, translate (with optional
CometKiwi reranking), score, and package the submission -- lives here.

It is fully resumable across the Madhava 12-hour wall-clock (SIGUSR1 + requeue).

WHICH LANGUAGES THIS FILE IS FOR
--------------------------------
IndicTrans2 natively supports the 22 *scheduled* Indian languages. Of our six
WMT26 pairs that means:
      as  (Assamese,  asm_Beng)   native
      mni (Manipuri,  mni_Beng)   native
      bodo(Bodo,      brx_Deva)   native
For Kokborok (trp) IndicTrans2 has no tag, so we use a SAME-SCRIPT surrogate
tag (asm_Beng -- Bengali script) and let the LoRA adapter learn the real
language from data. See LANG_REGISTRY below.
Mizo (lus) and Khasi (kha) are Latin-script and have no usable IndicTrans2
surrogate -> use run_nllb.py for those two pairs instead. This script will tell
you so and stop, rather than produce garbage.

QUICK USAGE  (all GPU work goes through sbatch; see run.sbatch)
--------------------------------------------------------------
  # 1) fine-tune en->as (per direction; this is the main quality step)
  python run_indictrans2.py finetune \
      --train data/en-as.train.csv --src en --tgt as \
      --src-col en --tgt-col as --out ckpts/it2_en-as

  # 2) translate the WMT test file (source only) -> submission txt
  python run_indictrans2.py translate \
      --infile data/test_en-as.csv --src en --tgt as --src-col en \
      --adapter ckpts/it2_en-as/final --out outputs/it2_en-as.txt --rerank

  # 3) score against references (e.g. WMT25 gold as a dev set)
  python run_indictrans2.py translate \
      --infile data/en-as.test.csv --src en --tgt as \
      --src-col en --tgt-col as --adapter ckpts/it2_en-as/final \
      --out outputs/it2_en-as.txt --rerank --score

  # 4) back-translation: make pseudo-parallel data with a trained adapter
  python run_indictrans2.py backtranslate \
      --infile data/mono_as.txt --src as --tgt en \
      --adapter ckpts/it2_as-en/final --out data/bt_as-en.tsv --labse-filter 0.75

  # 5) package final outputs into the submission zip
  python run_indictrans2.py package --outputs-dir outputs --team DoDS-IITPKD

SUBMISSION STRATEGY  (what each output file is, per the WMT26 rules)
-------------------------------------------------------------------
Per direction you may send up to 3 files. Only the PRIMARY is required; the
single system_description.pdf is mandatory for the whole submission. Contrastive
is OPTIONAL -- but it was our best scorer last year, and the leaderboard ranks
every submission together (best score wins regardless of label), so the
extra-data run is where the ceiling lives. Drop contrastive quietly if time runs
out; never drop a primary.
  primary       official WMT26 data (+ back-translation from official mono).
                Reliable, near-constrained.  -> DoDS-IITPKD_primary_<dir>.txt
  contrastive1  primary + external public corpora (BPCC/PMINDIA/SMOL/...) +
                both-base CometKiwi ensemble. -> DoDS-IITPKD_contrastive1_<dir>.txt
  contrastive2  optional second variant.      -> DoDS-IITPKD_contrastive2_<dir>.txt
Back-translation AND pretrained models are explicitly allowed in the primary
(the rules permit "additional monolingual resources, pretrained, etc."). Nothing
technique-side is banned; the labels only track data provenance. `package` below
enforces: every primary present + the PDF present.
================================================================================
"""

# ---------------------------------------------------------------------------
# OFFLINE-CLUSTER COMPATIBILITY (Madhava reality, June 2026)
# ---------------------------------------------------------------------------
# Every "extra" dependency is imported LAZILY inside the feature that needs it,
# so a missing package never breaks the core fine-tune / translate / score path.
#   - unbabel-comet (CometKiwi)  imported only inside --rerank ; falls back to
#                                plain beam search if absent. Submission valid.
#   - bitsandbytes               imported only inside --four-bit (QLoRA); not
#                                needed for IndicTrans2-1B on H100 anyway.
#   - sentence-transformers      imported only inside --labse-filter for BT.
# Required (must be importable):
#   torch, transformers, peft, datasets, accelerate, sacrebleu, pandas,
#   IndicTransToolkit (provides IndicProcessor + IndicEvaluator).
# All of the above are in the cluster venv per the shared requirements list.
# ---------------------------------------------------------------------------

import argparse
import csv
import os
import signal
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# ------------------------------------------------------------------ constants
DEFAULT_BASE_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"
DEFAULT_BASE_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"

# LANG_REGISTRY -- the single place that decides how each language is handled.
# tag      : the IndicTrans2 / FLORES tag the model actually sees
# native   : True if IndicTrans2 truly supports it; False if we use a surrogate
# script   : writing system (for documentation / sanity)
# use_nllb : True means "this file cannot do it well -- use run_nllb.py"
# To add a NEW language, add ONE line here. Nothing else changes.
LANG_REGISTRY = {
    "en":   dict(tag="eng_Latn", native=True,  script="Latn"),
    "as":   dict(tag="asm_Beng", native=True,  script="Beng"),
    "mni":  dict(tag="mni_Beng", native=True,  script="Beng"),   # Manipuri (Bengali script)
    "bodo": dict(tag="brx_Deva", native=True,  script="Deva"),   # Bodo
    "brx":  dict(tag="brx_Deva", native=True,  script="Deva"),   # alias for Bodo
    # ---- not native to IndicTrans2 ----------------------------------------
    "trp":  dict(tag="asm_Beng", native=False, script="Beng",   # Kokborok in Bengali script
                 surrogate_for="Kokborok",
                 note="Bengali-script surrogate (asm_Beng). Adapter learns Kokborok from data. "
                      "Optionally warm-start from a trained Bodo adapter via --init-adapter."),
    "lus":  dict(tag=None, native=False, script="Latn", use_nllb=True,
                 note="Mizo is Latin-script, not a scheduled language -> use run_nllb.py (lus_Latn)."),
    "kha":  dict(tag=None, native=False, script="Latn", use_nllb=True,
                 note="Khasi is Latin-script, no IndicTrans2 surrogate -> use run_nllb.py."),
}


def resolve(lang):
    """Return the model tag for a friendly language key, with loud warnings."""
    key = lang.lower()
    if key not in LANG_REGISTRY:
        sys.exit(f"[lang] '{lang}' is unknown. Add it to LANG_REGISTRY (one line).")
    info = LANG_REGISTRY[key]
    if info.get("use_nllb"):
        sys.exit(f"[lang] '{lang}': {info['note']}")
    if not info["native"]:
        print(f"[lang] !! '{lang}' is NOT native to IndicTrans2. Using surrogate tag "
              f"'{info['tag']}'. {info.get('note','')}", flush=True)
    return info["tag"]


# ------------------------------------------------------------------ data I/O
def read_columns(path, src_col, tgt_col=None):
    """Read a CSV/TSV (or 1-col txt) and return (src_list, tgt_list_or_None).

    - .txt / one column        -> source-only (for test inference / mono BT)
    - CSV/TSV with named cols   -> uses src_col / tgt_col
    - 2-column file, no header  -> first col = src, second = tgt
    """
    path = str(path)
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            src = [ln.rstrip("\n") for ln in f if ln.strip()]
        return src, None

    delim = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        sniff = f.readline()
        f.seek(0)
        has_header = any(c.isalpha() for c in sniff.split(delim)[0]) and (
            src_col in sniff or tgt_col is not None
        )
        if src_col and (src_col in sniff):
            reader = csv.DictReader(f, delimiter=delim)
            src, tgt = [], [] if tgt_col else None
            for row in reader:
                s = (row.get(src_col) or "").strip()
                if not s:
                    continue
                src.append(s)
                if tgt_col:
                    tgt.append((row.get(tgt_col) or "").strip())
            return src, tgt
        # headerless two-column
        reader = csv.reader(f, delimiter=delim)
        src, tgt = [], [] if tgt_col else None
        skipped = 0
        for parts in reader:
            if not parts or not parts[0].strip():
                continue
            if tgt_col is not None:
                # Require a usable second column too, or src/tgt go out of
                # sync (e.g. a handful of degenerate/empty BT-generated rows
                # with no tab). Skip the whole row, both sides, to keep
                # alignment.
                if len(parts) < 2 or not parts[1].strip():
                    skipped += 1
                    continue
                tgt.append(parts[1].strip())
            src.append(parts[0].strip())
        if skipped:
            print(f"[read_columns] {path}: skipped {skipped} row(s) with missing/empty "
                  f"second column.", flush=True)
        return src, tgt


# ------------------------------------------------------------------ model load
def load_model(base, adapter=None, four_bit=False, for_training=False):
    """Load IndicTrans2 + optional LoRA adapter. trust_remote_code is required."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    kwargs = dict(trust_remote_code=True)
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        if not for_training:
            kwargs["device_map"] = "auto"

    model = AutoModelForSeq2SeqLM.from_pretrained(base, **kwargs)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[model] loaded adapter {adapter}", flush=True)
    return model, tok


def get_processor(inference=True):
    from IndicTransToolkit import IndicProcessor
    return IndicProcessor(inference=inference)


# ------------------------------------------------------------------ translate
def generate(model, tok, ip, sentences, src_tag, tgt_tag, num_beams, max_len,
             batch_size, n_return=1):
    """Return either a flat list (n_return==1) or list-of-lists of candidates."""
    model.eval()
    device = next(model.parameters()).device
    all_out = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        pre = ip.preprocess_batch(chunk, src_lang=src_tag, tgt_lang=tgt_tag)
        enc = tok(pre, truncation=True, padding="longest", max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                num_beams=num_beams,
                num_return_sequences=n_return,
                max_length=max_len,
                early_stopping=True,
            )
        dec = tok.batch_decode(gen, skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
        post = ip.postprocess_batch(dec, lang=tgt_tag)
        if n_return == 1:
            all_out.extend(post)
        else:
            for j in range(0, len(post), n_return):
                all_out.append(post[j:j + n_return])
        print(f"[gen] {min(i + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return all_out


# CometKiwi reference-free reranking ------------------------------------------
_KIWI = None


def load_kiwi():
    """Load CometKiwi once. Model is gated on HF -- pre-download on login node."""
    global _KIWI
    if _KIWI is None:
        from comet import download_model, load_from_checkpoint
        ck = download_model("Unbabel/wmt22-cometkiwi-da")
        _KIWI = load_from_checkpoint(ck)
    return _KIWI


def rerank(sources, candidate_lists):
    """For each source, pick the candidate with the highest CometKiwi score."""
    kiwi = load_kiwi()
    data, spans = [], []
    for s, cands in zip(sources, candidate_lists):
        start = len(data)
        for c in cands:
            data.append({"src": s, "mt": c})
        spans.append((start, len(data)))
    scores = kiwi.predict(data, batch_size=32,
                          gpus=1 if torch.cuda.is_available() else 0)["scores"]
    best = []
    for (a, b), cands in zip(spans, candidate_lists):
        seg = scores[a:b]
        best.append(cands[int(max(range(len(seg)), key=lambda k: seg[k]))])
    return best


def cmd_translate(args):
    src_tag = resolve(args.src)
    tgt_tag = resolve(args.tgt)
    base = DEFAULT_BASE_EN_INDIC if args.src == "en" else DEFAULT_BASE_INDIC_EN
    base = args.base or base

    src, refs = read_columns(args.infile, args.src_col, args.tgt_col)
    print(f"[translate] {len(src)} sentences  {args.src}->{args.tgt}  base={base}", flush=True)

    model, tok = load_model(base, adapter=args.adapter)
    ip = get_processor(inference=True)

    use_rerank = args.rerank
    if use_rerank:
        try:
            load_kiwi()                     # probe now: import comet + load model, fail early
        except Exception as e:
            print(f"[rerank] CometKiwi unavailable ({type(e).__name__}: {e}).", flush=True)
            print("[rerank] Falling back to plain beam search. This is fine -- reranking is "
                  "an OPTIONAL extra (+1-2 COMET), not required for a valid submission.", flush=True)
            use_rerank = False

    if use_rerank:
        cand = generate(model, tok, ip, src, src_tag, tgt_tag,
                        num_beams=args.n_candidates, max_len=args.max_len,
                        batch_size=args.batch_size, n_return=args.n_candidates)
        hyps = rerank(src, cand)
    else:
        hyps = generate(model, tok, ip, src, src_tag, tgt_tag,
                        num_beams=args.num_beams, max_len=args.max_len,
                        batch_size=args.batch_size, n_return=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    print(f"[translate] wrote {len(hyps)} lines -> {args.out}", flush=True)

    if args.score:
        if refs is None:
            sys.exit("[score] need --tgt-col references to score.")
        score_indic(hyps, refs, tgt_tag)


def score_indic(hyps, refs, tgt_tag):
    """Official IndicTrans2 scorer (BLEU + chrF++). No external deps beyond IndicTransToolkit."""
    from IndicTransToolkit import IndicEvaluator
    ev = IndicEvaluator()
    res = ev.evaluate(tgt_lang=tgt_tag, preds=hyps, refs=refs)
    print("\n===== RESULTS (IndicEvaluator) =====")
    print(res)
    return res


# ------------------------------------------------------------------ back-translate
def cmd_backtranslate(args):
    """Translate a monolingual file and write a 2-col TSV of pseudo-parallel pairs.
    Forward BT example: mono Assamese (--src as) -> English (--tgt en) gives
    (en_synthetic, as_real) which trains en->as. We write 'tgt<TAB>src' so the
    output is already oriented as (English, Indic) -> use directly for en->indic.
    """
    src_tag = resolve(args.src)
    tgt_tag = resolve(args.tgt)
    base = DEFAULT_BASE_EN_INDIC if args.src == "en" else DEFAULT_BASE_INDIC_EN
    base = args.base or base

    mono, _ = read_columns(args.infile, args.src_col, None)
    model, tok = load_model(base, adapter=args.adapter)
    ip = get_processor(inference=True)
    hyps = generate(model, tok, ip, mono, src_tag, tgt_tag,
                    num_beams=args.num_beams, max_len=args.max_len,
                    batch_size=args.batch_size, n_return=1)

    pairs = list(zip(mono, hyps))   # (real_X, synthetic_Y)
    if args.labse_filter:
        pairs = labse_filter(pairs, args.labse_filter)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        # write as synthetic_Y <TAB> real_X so a downstream Y->X trainer is happy
        for real_x, syn_y in pairs:
            f.write(f"{syn_y}\t{real_x}\n")
    print(f"[bt] wrote {len(pairs)} pseudo-parallel pairs -> {args.out}", flush=True)


def labse_filter(pairs, thresh):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/LaBSE",
                            device="cuda" if torch.cuda.is_available() else "cpu")
    kept = []
    for i in range(0, len(pairs), 128):
        chunk = pairs[i:i + 128]
        a = m.encode([p[0] for p in chunk], convert_to_tensor=True, normalize_embeddings=True)
        b = m.encode([p[1] for p in chunk], convert_to_tensor=True, normalize_embeddings=True)
        sims = (a * b).sum(dim=1).cpu().tolist()
        kept += [p for p, s in zip(chunk, sims) if s >= thresh]
    print(f"[labse] kept {len(kept)}/{len(pairs)} at >= {thresh}", flush=True)
    return kept


# ------------------------------------------------------------------ fine-tune
# Module-level flag flipped by SIGUSR1 so a wall-clock kill saves a checkpoint.
_WANT_STOP = {"v": False}


# --- THE FIX for "You have to specify either decoder_input_ids ..." -----------
# IndicTrans2's forward() does NOT shift labels into decoder_input_ids the way
# BART/M2M100 do, and once the model is PEFT/LoRA-wrapped, neither
# DataCollatorForSeq2Seq nor IndicTransToolkit's IndicDataCollator reliably build
# them either -- both gate on hasattr(model, "prepare_decoder_input_ids_from_labels"),
# and that lookup does not survive the PEFT wrapper. So the decoder gets nothing
# and training dies on the very first step. We build decoder_input_ids ourselves,
# every batch, from the model's own config. Version-independent, no toolkit needed.
def _shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Right-shift label ids to make decoder_input_ids (Fairseq/M2M convention)."""
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    # padding positions in the labels are -100; the decoder input must use pad there
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


@dataclass
class IT2DataCollator:
    """Pad a seq2seq batch (delegated to a stock collator) and ALWAYS attach
    decoder_input_ids built from the labels. This is what unblocks IndicTrans2
    LoRA fine-tuning under transformers>=4.4x + PEFT."""
    pad_collator: Any            # DataCollatorForSeq2Seq(model=None) -- padding only
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, features, return_tensors=None):
        batch = self.pad_collator(features)         # pads input_ids/attn/labels
        batch["decoder_input_ids"] = _shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )
        return batch


def _sigusr1(signum, frame):
    print("[signal] SIGUSR1 received -> will checkpoint and exit cleanly.", flush=True)
    _WANT_STOP["v"] = True


def cmd_finetune(args):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              TrainerCallback)
    from transformers import DataCollatorForSeq2Seq

    signal.signal(signal.SIGUSR1, _sigusr1)

    src_tag = resolve(args.src)
    tgt_tag = resolve(args.tgt)
    base = DEFAULT_BASE_EN_INDIC if args.src == "en" else DEFAULT_BASE_INDIC_EN
    base = args.base or base

    # ---- data (supports multiple --train files, e.g. official + BT) ----
    src_all, tgt_all = [], []
    for fp in args.train:
        s, t = read_columns(fp, args.src_col, args.tgt_col)
        if t is None:
            sys.exit(f"[finetune] {fp} has no target column ({args.tgt_col}).")
        src_all += s
        tgt_all += t
    if getattr(args, "max_rows", 0) and args.max_rows > 0:
        src_all, tgt_all = src_all[:args.max_rows], tgt_all[:args.max_rows]
    print(f"[finetune] {len(src_all)} pairs from {len(args.train)} file(s).", flush=True)

    model, tok = load_model(base, four_bit=args.four_bit, for_training=True)
    model.config.use_cache = False              # REQUIRED with gradient checkpointing

    # Token ids used to build decoder_input_ids. Read from the model's OWN config
    # so they are always right for whichever IndicTrans2 variant is loaded.
    # (Inference already works for the team, which proves these are valid ints.)
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = tok.pad_token_id
    dec_start = model.config.decoder_start_token_id
    if dec_start is None:
        dec_start = model.config.bos_token_id if model.config.bos_token_id is not None else pad_id
    print(f"[finetune] decoder_start_token_id={dec_start}  pad_token_id={pad_id}", flush=True)

    if args.four_bit:
        model = prepare_model_for_kbit_training(model)
    # PeftModel.from_pretrained() (the --init-adapter path) reconstructs the
    # adapter from its SAVED adapter_config.json, which itself records
    # use_dora / use_rslora (and r). So --dora/--rslora on THIS run's CLI are
    # only meaningful as a *cross-check* against what's already saved -- they
    # do not change how the loaded adapter behaves. Compare requested vs saved
    # and only refuse on a genuine mismatch (requesting DoRA/rsLoRA that the
    # saved adapter does NOT have, or vice versa, which would silently produce
    # a different architecture than the user thinks).
    if args.init_adapter:
        import json
        cfg_path = Path(args.init_adapter) / "adapter_config.json"
        if cfg_path.exists():
            saved = json.loads(cfg_path.read_text())
            saved_dora = bool(saved.get("use_dora", False))
            saved_rslora = bool(saved.get("use_rslora", False))
            saved_r = saved.get("r")
            mismatches = []
            if args.dora != saved_dora:
                mismatches.append(f"--dora={args.dora} but saved adapter has use_dora={saved_dora}")
            if args.rslora != saved_rslora:
                mismatches.append(f"--rslora={args.rslora} but saved adapter has use_rslora={saved_rslora}")
            if saved_r is not None and int(saved_r) != int(args.lora_r):
                mismatches.append(f"--lora-r={args.lora_r} but saved adapter has r={saved_r}")
            if mismatches and not args.allow_init_adapter_dora_mismatch:
                sys.exit(
                    "[finetune] --init-adapter config does not match requested flags:\n"
                    + "\n".join(f"  - {m}" for m in mismatches) + "\n"
                    "[finetune] PeftModel.from_pretrained() will use the SAVED config "
                    "(r/use_dora/use_rslora), NOT the values above -- so training would "
                    "silently proceed with a different architecture than requested.\n"
                    "[finetune] Either fix the CLI flags to match the saved adapter, or "
                    "pass --allow-init-adapter-dora-mismatch to proceed anyway (the saved "
                    "config wins; the mismatched CLI flags are ignored)."
                )
            elif mismatches:
                print("[finetune] !! --allow-init-adapter-dora-mismatch set; proceeding. "
                      "Saved adapter config wins for: " + "; ".join(mismatches), flush=True)
            else:
                print(f"[finetune] --init-adapter config matches requested flags "
                      f"(r={saved_r}, use_dora={saved_dora}, use_rslora={saved_rslora}).", flush=True)
        else:
            print(f"[finetune] WARNING: {cfg_path} not found; cannot verify --dora/--rslora/"
                  f"--lora-r against the adapter being loaded.", flush=True)
    if args.init_adapter:                       # sister-language warm start (Bodo->Kokborok)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"[finetune] warm-started from {args.init_adapter}", flush=True)
    else:
        # Target only attention + FFN projections.
        # Avoid "all-linear": it captures lm_head and shared embeddings, which
        # IndicTrans2 uses for 22-language tag routing. Corrupting those causes
        # wrong-script / empty output in low-resource directions (en->as, en->mni).
        _IT2_ATTN_FFN = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "encoder_attn.q_proj", "encoder_attn.k_proj",
            "encoder_attn.v_proj", "encoder_attn.out_proj",
            "fc1", "fc2",
        ]
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="SEQ_2_SEQ_LM",
            target_modules=_IT2_ATTN_FFN,
            use_dora=args.dora, use_rslora=args.rslora,
        )
        model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()          # let grads reach LoRA through frozen embeddings
    model.print_trainable_parameters()

    ip = get_processor(inference=False)
    max_len = args.max_len

    def preprocess(batch):
        # source: add the eng_Latn/asm_Beng style tags
        s_pre = _ip_pre(ip, batch["src"], src_tag, tgt_tag, is_target=False)
        # target: normalise script, NO tags (these become the labels)
        t_pre = _ip_pre(ip, batch["tgt"], src_tag, tgt_tag, is_target=True)
        enc = tok(s_pre, truncation=True, max_length=max_len)
        lab = tok(text_target=t_pre, truncation=True, max_length=max_len)
        enc["labels"] = lab["input_ids"]
        return enc

    ds = Dataset.from_dict({"src": src_all, "tgt": tgt_all})
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names,
                desc="tokenizing")

    # Padding is delegated to the stock collator (model=None so it does NOT try,
    # and fail, to build decoder_input_ids itself); IT2DataCollator then attaches
    # decoder_input_ids every batch. This is the core fix.
    pad_collator = DataCollatorForSeq2Seq(
        tok, model=None, label_pad_token_id=-100, padding="longest"
    )
    collator = IT2DataCollator(pad_collator, pad_token_id=pad_id,
                               decoder_start_token_id=dec_start)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = _auto_resume(out, lora_r=args.lora_r) if args.resume == "auto" else (
        args.resume if args.resume not in ("", "none") else None)
    if resume:
        print(f"[finetune] resuming from {resume}", flush=True)
    elif args.resume == "auto":
        print("[finetune] no compatible checkpoint found -- starting from scratch.", flush=True)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # Cosine decay + doubled warmup prevents early overshooting that causes
        # epoch-2 collapse on small Indic datasets like en->as / en->mni.
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.four_bit else "adamw_torch",
        # Label smoothing 0.1: prevents over-sharpening on low-resource targets.
        # Root cause of 5k/2-epoch and 48k/1-epoch collapses.
        label_smoothing_factor=0.1,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        save_safetensors=True,
    )

    class StopOnSignal(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if _WANT_STOP["v"]:
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds,
                             data_collator=collator, callbacks=[StopOnSignal()])
    try:
        trainer.train(resume_from_checkpoint=resume)
    except RuntimeError as exc:
        if resume and "size mismatch" in str(exc):
            print(
                f"[finetune] !! Checkpoint {resume} has incompatible LoRA shapes "
                f"(r mismatch or target_modules changed).\n"
                f"[finetune] !! Discarding checkpoint and restarting from scratch.",
                flush=True,
            )
            trainer.train(resume_from_checkpoint=None)
        else:
            raise

    if not _WANT_STOP["v"]:
        trainer.save_model(str(out / "final"))
        print(f"[finetune] DONE. adapter -> {out/'final'}", flush=True)
    else:
        print("[finetune] stopped by wall-clock signal; SLURM will requeue and resume.",
              flush=True)


def _ip_pre(ip, batch, src_tag, tgt_tag, is_target):
    """IndicProcessor.preprocess_batch with the correct src/tgt orientation.

    CRITICAL (this is the fix for en->as / en->mni / en->trp collapse):
    For the TARGET side, IndicTrans2 expects the text transliterated INTO its
    internal (Devanagari-unified) representation. That requires passing the
    TARGET language as `src_lang` -- exactly how the official IndicTransToolkit
    training recipe calls it. Passing the real source tag (eng_Latn) leaves an
    Indic-script target in its NATIVE script, so the model is trained on the
    wrong representation and never learns: loss plateaus around 6 and the output
    is repeated tokens. Verified directly: with this orientation the target body
    is byte-identical to the working reverse direction's source body.
    Directions whose target is English (X->en) or Devanagari/Bodo (en->bodo) are
    unaffected -- the swap is a no-op for them -- so the working models do not
    change.
    """
    a, b = (tgt_tag, src_tag) if is_target else (src_tag, tgt_tag)
    try:
        return ip.preprocess_batch(batch, src_lang=a, tgt_lang=b, is_target=is_target)
    except TypeError:                      # older toolkit without is_target kwarg
        return ip.preprocess_batch(batch, src_lang=a, tgt_lang=b)


def _auto_resume(out, lora_r=None):
    """Return the latest checkpoint whose LoRA rank matches lora_r, or None.

    When --lora-r changes between runs (e.g. r=32 -> r=16) the checkpoint
    adapter weights have the wrong shape and PyTorch will raise a size-mismatch
    RuntimeError.  We detect this early by peeking at adapter_config.json inside
    the checkpoint before handing it to the Trainer.
    """
    import json
    cks = sorted(Path(out).glob("checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
    for ck in reversed(cks):          # newest first
        cfg_path = ck / "adapter_config.json"
        if not cfg_path.exists():
            # No adapter config yet (e.g. base-model-only checkpoint) -- skip.
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            ck_r = cfg.get("r")
            if lora_r is not None and ck_r is not None and int(ck_r) != int(lora_r):
                print(
                    f"[resume] skipping {ck.name}: checkpoint has r={ck_r} "
                    f"but current run uses r={lora_r}. "
                    f"Delete {out} manually if you want a clean restart.",
                    flush=True,
                )
                continue
        except Exception as e:
            print(f"[resume] could not read {cfg_path}: {e} -- skipping.", flush=True)
            continue
        return str(ck)
    return None


# ------------------------------------------------------------------ package
def cmd_package(args):
    team = args.team
    outdir = Path(args.outputs_dir)
    pairs = ["en_to_as", "as_to_en", "en_to_lus", "lus_to_en",
             "en_to_kha", "kha_to_en", "en_to_mni", "mni_to_en",
             "en_to_bodo", "bodo_to_en", "en_to_trp", "trp_to_en"]
    staging = outdir / team
    staging.mkdir(parents=True, exist_ok=True)
    missing = []
    for p in pairs:
        prim = outdir / f"{team}_primary_{p}.txt"
        if prim.exists():
            (staging / prim.name).write_text(prim.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  ok  primary {p}")
        else:
            missing.append(prim.name)
        for c in ("contrastive1", "contrastive2"):
            cf = outdir / f"{team}_{c}_{p}.txt"
            if cf.exists():
                (staging / cf.name).write_text(cf.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"  ok  {c} {p}")
    sd = outdir / "system_description.pdf"
    if sd.exists():
        (staging / sd.name).write_bytes(sd.read_bytes())
    else:
        missing.append("system_description.pdf  (MANDATORY)")
    if missing:
        print("\n[package] WARNING missing files:")
        for m in missing:
            print("   -", m)
    zpath = outdir / f"{team}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in staging.iterdir():
            z.write(f, f"{team}/{f.name}")
    print(f"\n[package] created {zpath}")
    print(f"[package] email to lrilt.wmt@gmail.com  subject:")
    print(f"          {team}: Submission File for Shared Task: Low-Resource Indic Language Translation")


# ------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description="IndicTrans2 all-in-one for WMT26 IndicMT")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--src", required=True, help="source lang key: en/as/mni/bodo/trp")
        sp.add_argument("--tgt", required=True, help="target lang key")
        sp.add_argument("--base", default=None, help="override base model id")
        sp.add_argument("--max-len", type=int, default=256)
        sp.add_argument("--batch-size", type=int, default=16)

    t = sub.add_parser("translate"); common(t)
    t.add_argument("--infile", required=True)
    t.add_argument("--src-col", default="src")
    t.add_argument("--tgt-col", default=None)
    t.add_argument("--adapter", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--num-beams", type=int, default=5)
    t.add_argument("--rerank", action="store_true", help="CometKiwi rerank over n candidates")
    t.add_argument("--n-candidates", type=int, default=10)
    t.add_argument("--score", action="store_true")

    b = sub.add_parser("backtranslate"); common(b)
    b.add_argument("--infile", required=True, help="monolingual file (txt or csv)")
    b.add_argument("--src-col", default="text")
    b.add_argument("--adapter", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--num-beams", type=int, default=5)
    b.add_argument("--labse-filter", type=float, default=0.0)

    f = sub.add_parser("finetune"); common(f)
    f.add_argument("--train", nargs="+", required=True, help="one or more parallel files")
    f.add_argument("--src-col", default="src")
    f.add_argument("--tgt-col", default="tgt")
    f.add_argument("--out", required=True)
    f.add_argument("--max-rows", type=int, default=0, help="cap training pairs (0=all); use for smoke tests")
    f.add_argument("--epochs", type=float, default=5)
    f.add_argument("--lr", type=float, default=5e-5)   # 2e-4 causes collapse; 5e-5 is the only proven-stable value
    f.add_argument("--grad-accum", type=int, default=4)
    f.add_argument("--lora-r", type=int, default=16)
    f.add_argument("--lora-alpha", type=int, default=32)  # alpha >= r; scaling = alpha/r = 2.0
    f.add_argument("--dora", action="store_true")
    f.add_argument("--rslora", action="store_true")
    f.add_argument("--four-bit", action="store_true", help="QLoRA 4-bit (not needed for 1B on H100)")
    f.add_argument("--init-adapter", default=None, help="warm-start from this adapter (sister-lang transfer)")
    f.add_argument("--allow-init-adapter-dora-mismatch", action="store_true",
                   help="if --dora/--rslora/--lora-r don't match the saved "
                        "adapter_config.json of --init-adapter, proceed anyway "
                        "(the saved config wins; default: error out)")
    f.add_argument("--save-steps", type=int, default=300)
    f.add_argument("--resume", default="auto")

    k = sub.add_parser("package")
    k.add_argument("--outputs-dir", default="outputs")
    k.add_argument("--team", default="DoDS-IITPKD")
    return p


def main():
    args = build_parser().parse_args()
    {"translate": cmd_translate, "backtranslate": cmd_backtranslate,
     "finetune": cmd_finetune, "package": cmd_package}[args.cmd](args)


if __name__ == "__main__":
    main()

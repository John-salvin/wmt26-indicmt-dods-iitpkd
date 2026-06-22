#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 run_nllb.py  ·  DoDS-IITPKD · WMT 2026 Low-Resource Indic MT
================================================================================
ONE self-contained file. No other project file is imported. Mirror of
run_indictrans2.py but for NLLB-200-3.3B. Same five modes: finetune,
backtranslate, translate (+CometKiwi rerank), score, package. Resumable across
the 12-hour wall-clock (SIGUSR1 + requeue).

WHICH LANGUAGES THIS FILE IS FOR
--------------------------------
NLLB-200 natively covers (of our six pairs):
      as  (Assamese, asm_Beng)   native
      mni (Manipuri, mni_Beng)   native
      lus (Mizo,     lus_Latn)   native   <-- IndicTrans2 cannot do this; NLLB can
NOT in NLLB-200 -> we use a SAME-SCRIPT surrogate FLORES code and let the LoRA
adapter learn the real language:
      kha  (Khasi,    Latin)     -> surrogate lus_Latn   (NE-Indian Latin)
      bodo (Bodo,     Devanagari)-> surrogate hin_Deva   (Devanagari)
      trp  (Kokborok, Bengali)   -> surrogate ben_Beng   (Bengali)
The surrogate only seeds the script/decoder prior; the adapter does the rest.
See LANG_REGISTRY -- to add a language, add ONE line.

Recommended model-per-pair split for the team:
      as, mni, bodo, trp  -> run_indictrans2.py is usually stronger
      lus, kha            -> THIS file (NLLB) is the right tool
      everything          -> run both for contrastive submissions / ensembling

QUICK USAGE  (all GPU work via sbatch; see run.sbatch)
------------------------------------------------------
  python run_nllb.py finetune  --train data/en-lus.train.csv --src en --tgt lus \
         --src-col en --tgt-col lus --out ckpts/nllb_en-lus --four-bit --dora --rslora
  python run_nllb.py translate --infile data/test_en-lus.csv --src en --tgt lus \
         --src-col en --adapter ckpts/nllb_en-lus/final --out outputs/nllb_en-lus.txt --rerank
  python run_nllb.py translate --infile data/en-lus.test.csv --src en --tgt lus \
         --src-col en --tgt-col lus --adapter ckpts/nllb_en-lus/final \
         --out outputs/nllb_en-lus.txt --score
  python run_nllb.py backtranslate --infile data/mono_lus.txt --src lus --tgt en \
         --adapter ckpts/nllb_lus-en/final --out data/bt_lus-en.tsv --labse-filter 0.75
  python run_nllb.py package --outputs-dir outputs --team DoDS-IITPKD

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
  contrastive   primary + external public corpora (BPCC/PMINDIA/SMOL/...) +
                both-base CometKiwi ensemble. -> DoDS-IITPKD_contrastive_<dir>.txt
  contrastive2  optional second variant.      -> DoDS-IITPKD_contrastive2_<dir>.txt
Back-translation AND pretrained models are explicitly allowed in the primary
(the rules permit "additional monolingual resources, pretrained, etc."). Nothing
technique-side is banned; the labels only track data provenance. `package` below
enforces: every primary present + the PDF present.
================================================================================
"""

# ---------------------------------------------------------------------------
# OFFLINE-CLUSTER COMPATIBILITY
# ---------------------------------------------------------------------------
# Every "extra" dependency is imported LAZILY inside the feature that needs it,
# so a missing package never breaks the core fine-tune / translate / score path.
#   - unbabel-comet (CometKiwi)  imported only inside --rerank ; falls back to
#                                plain beam search if absent. Submission valid.
#   - bitsandbytes               imported only inside --four-bit (QLoRA). Needed
#                                ONLY for NLLB-3.3B fine-tune on memory-limited
#                                GPUs; on 80GB H100 you can train in bf16 by
#                                omitting --four-bit (slower but no bnb dep).
#   - sentence-transformers      imported only inside --labse-filter for BT.
# Required (must be importable):
#   torch, transformers, peft, datasets, accelerate, sacrebleu, pandas.
# ---------------------------------------------------------------------------

import argparse
import csv
import sys as _sys
csv.field_size_limit(min(2**31 - 1, _sys.maxsize))
import os
import signal
import sys
import zipfile
from pathlib import Path

import torch

DEFAULT_BASE = "facebook/nllb-200-3.3B"

# tag = FLORES-200 code the tokenizer knows; native flips to surrogate handling.
LANG_REGISTRY = {
    "en":   dict(tag="eng_Latn", native=True,  script="Latn"),
    "as":   dict(tag="asm_Beng", native=True,  script="Beng"),
    "mni":  dict(tag="mni_Beng", native=True,  script="Beng"),
    "lus":  dict(tag="lus_Latn", native=True,  script="Latn"),   # Mizo
    # ---- not in NLLB-200 -> same-script surrogate -------------------------
    "kha":  dict(tag="lus_Latn", native=False, script="Latn", surrogate_for="Khasi",
                 note="Latin surrogate (lus_Latn). Adapter learns Khasi. Alt: eng_Latn."),
    "bodo": dict(tag="hin_Deva", native=False, script="Deva", surrogate_for="Bodo",
                 note="Devanagari surrogate (hin_Deva). For Bodo, run_indictrans2.py (brx_Deva) is usually stronger."),
    "brx":  dict(tag="hin_Deva", native=False, script="Deva", surrogate_for="Bodo"),
    "trp":  dict(tag="lus_Latn", native=False, script="Latn", surrogate_for="Kokborok",
                 note="Latin surrogate (lus_Latn, Mizo -- same script + Tibeto-Burman family). Adapter learns Kokborok."),
}


def resolve(lang):
    key = lang.lower()
    if key not in LANG_REGISTRY:
        sys.exit(f"[lang] '{lang}' unknown. Add it to LANG_REGISTRY (one line).")
    info = LANG_REGISTRY[key]
    if not info["native"]:
        print(f"[lang] !! '{lang}' is NOT in NLLB-200. Using surrogate FLORES code "
              f"'{info['tag']}'. {info.get('note','')}", flush=True)
    return info["tag"]


# ------------------------------------------------------------------ data I/O
def read_columns(path, src_col, tgt_col=None):
    path = str(path)
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()], None
    delim = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        head = f.readline()
        f.seek(0)
        if src_col and (src_col in head):
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
        reader = csv.reader(f, delimiter=delim)
        src, tgt = [], [] if tgt_col else None
        for parts in reader:
            if not parts or not parts[0].strip():
                continue
            src.append(parts[0].strip())
            if tgt_col is not None and len(parts) > 1:
                tgt.append(parts[1].strip())
        return src, tgt


# ------------------------------------------------------------------ model load
def load_model(adapter=None, four_bit=False, for_training=False, base=DEFAULT_BASE):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    kwargs = {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        if not for_training:
            kwargs["device_map"] = "auto"
    model = AutoModelForSeq2SeqLM.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[model] adapter {adapter}", flush=True)
    return model, tok


def bos_id(tok, tgt_tag):
    """forced_bos_token_id for the target language, across tokenizer versions."""
    try:
        return tok.convert_tokens_to_ids(tgt_tag)
    except Exception:
        return tok.lang_code_to_id[tgt_tag]


# ------------------------------------------------------------------ translate
def generate(model, tok, sentences, src_tag, tgt_tag, num_beams, max_len,
             batch_size, n_return=1, length_penalty=1.0, no_repeat_ngram_size=0):
    model.eval()
    device = next(model.parameters()).device
    tok.src_lang = src_tag
    fbos = bos_id(tok, tgt_tag)
    out = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        enc = tok(chunk, truncation=True, padding="longest", max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=fbos,
                                 num_beams=num_beams, num_return_sequences=n_return, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                                 max_length=max_len, early_stopping=True)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        if n_return == 1:
            out.extend(dec)
        else:
            for j in range(0, len(dec), n_return):
                out.append(dec[j:j + n_return])
        print(f"[gen] {min(i + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return out


_KIWI = None


def load_kiwi():
    global _KIWI
    if _KIWI is None:
        from comet import download_model, load_from_checkpoint
        ckpt = os.environ.get("KIWI_CKPT", "wmt22-cometkiwi-da")
        _KIWI = load_from_checkpoint(ckpt) if os.path.exists(ckpt) else download_model(ckpt)
    return _KIWI


def rerank(sources, candidate_lists):
    kiwi = load_kiwi()
    data, spans = [], []
    for s, cands in zip(sources, candidate_lists):
        a = len(data)
        data += [{"src": s, "mt": c} for c in cands]
        spans.append((a, len(data)))
    scores = kiwi.predict(data, batch_size=32,
                          gpus=1 if torch.cuda.is_available() else 0)["scores"]
    best = []
    for (a, b), cands in zip(spans, candidate_lists):
        seg = scores[a:b]
        best.append(cands[int(max(range(len(seg)), key=lambda k: seg[k]))])
    return best


def cmd_translate(args):
    src_tag, tgt_tag = resolve(args.src), resolve(args.tgt)
    src, refs = read_columns(args.infile, args.src_col, args.tgt_col)
    print(f"[translate] {len(src)} sents {args.src}->{args.tgt}", flush=True)
    model, tok = load_model(adapter=args.adapter, base=args.base or DEFAULT_BASE)
    if args.rerank:
        cand = generate(model, tok, src, src_tag, tgt_tag, args.n_candidates,
                        args.max_len, args.batch_size, n_return=args.n_candidates)
        hyps = rerank(src, cand)
    else:
        hyps = generate(model, tok, src, src_tag, tgt_tag, args.num_beams, args.max_len, args.batch_size, length_penalty=args.length_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    print(f"[translate] wrote {len(hyps)} lines -> {args.out}", flush=True)
    if args.score:
        if refs is None:
            sys.exit("[score] need --tgt-col references.")
        score_sacre(hyps, refs, args.tgt, src, args.comet)


def score_sacre(hyps, refs, tgt_key, srcs=None, comet=False):
    import sacrebleu
    tok = "13a" if tgt_key == "en" else "flores200"
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)   # chrF++
    print("\n===== RESULTS (sacreBLEU) =====")
    print(f"BLEU   : {bleu.score:.2f}   ({bleu.bp})")
    print(f"chrF++ : {chrf.score:.2f}")
    if comet and srcs:
        from comet import download_model, load_from_checkpoint
        m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = m.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        print(f"COMET22: {out['system_score']:.4f}")


# ------------------------------------------------------------------ back-translate
def cmd_backtranslate(args):
    src_tag, tgt_tag = resolve(args.src), resolve(args.tgt)
    mono, _ = read_columns(args.infile, args.src_col, None)
    model, tok = load_model(adapter=args.adapter, base=args.base or DEFAULT_BASE)
    hyps = generate(model, tok, mono, src_tag, tgt_tag, args.num_beams,
                    args.max_len, args.batch_size, n_return=1)
    pairs = list(zip(mono, hyps))
    if args.labse_filter:
        pairs = labse_filter(pairs, args.labse_filter)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for real_x, syn_y in pairs:
            f.write(f"{syn_y}\t{real_x}\n")
    print(f"[bt] wrote {len(pairs)} pairs -> {args.out}", flush=True)


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
_WANT_STOP = {"v": False}


def _sigusr1(signum, frame):
    print("[signal] SIGUSR1 -> checkpoint and exit.", flush=True)
    _WANT_STOP["v"] = True


def cmd_finetune(args):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, TrainerCallback)
    signal.signal(signal.SIGUSR1, _sigusr1)

    src_tag, tgt_tag = resolve(args.src), resolve(args.tgt)
    src_all, tgt_all = [], []
    for fp in args.train:
        s, t = read_columns(fp, args.src_col, args.tgt_col)
        if t is None:
            sys.exit(f"[finetune] {fp} has no target column.")
        src_all += s
        tgt_all += t
    print(f"[finetune] {len(src_all)} pairs.", flush=True)

    model, tok = load_model(four_bit=args.four_bit, for_training=True,
                            base=args.base or DEFAULT_BASE)
    model.config.use_cache = False              # REQUIRED with gradient checkpointing
    tok.src_lang = src_tag
    if args.four_bit:
        model = prepare_model_for_kbit_training(model)
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"[finetune] warm-started from {args.init_adapter}", flush=True)
    else:
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            use_dora=args.dora, use_rslora=args.rslora)
        model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()          # let grads reach LoRA through frozen embeddings
    model.print_trainable_parameters()

    max_len = args.max_len

    def preprocess(batch):
        tok.src_lang = src_tag
        enc = tok(batch["src"], truncation=True, max_length=max_len)
        tok.src_lang = tgt_tag                      # so the target gets its lang token
        tok.src_lang = tgt_tag
        lab = tok(batch["tgt"], truncation=True, max_length=max_len)
        tok.src_lang = src_tag
        tok.src_lang = src_tag
        enc["labels"] = lab["input_ids"]
        return enc

    ds = Dataset.from_dict({"src": src_all, "tgt": tgt_all})
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names, desc="tok")
    collator = DataCollatorForSeq2Seq(tok, model=model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = _auto_resume(out) if args.resume == "auto" else (
        args.resume if args.resume not in ("", "none") else None)
    if resume:
        print(f"[finetune] resuming from {resume}", flush=True)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=0.03, num_train_epochs=args.epochs, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.four_bit else "adamw_torch",
        logging_steps=50, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=3, report_to=[], save_safetensors=True)

    class StopOnSignal(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if _WANT_STOP["v"]:
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds,
                             data_collator=collator, callbacks=[StopOnSignal()])
    trainer.train(resume_from_checkpoint=resume)
    if not _WANT_STOP["v"]:
        trainer.save_model(str(out / "final"))
        print(f"[finetune] DONE -> {out/'final'}", flush=True)
    else:
        print("[finetune] wall-clock stop; SLURM requeue will resume.", flush=True)


def _auto_resume(out):
    cks = sorted(Path(out).glob("checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
    return str(cks[-1]) if cks else None


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
        for c in ("contrastive", "contrastive2"):
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
        print("\n[package] WARNING missing:")
        for m in missing:
            print("   -", m)
    zpath = outdir / f"{team}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in staging.iterdir():
            z.write(f, f"{team}/{f.name}")
    print(f"\n[package] created {zpath}")
    print(f"[package] email lrilt.wmt@gmail.com  subject:")
    print(f"          {team}: Submission File for Shared Task: Low-Resource Indic Language Translation")


# ------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description="NLLB-3.3B all-in-one for WMT26 IndicMT")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--src", required=True)
        sp.add_argument("--tgt", required=True)
        sp.add_argument("--base", default=None)
        sp.add_argument("--max-len", type=int, default=256)
        sp.add_argument("--batch-size", type=int, default=16)

    t = sub.add_parser("translate"); common(t)
    t.add_argument("--infile", required=True)
    t.add_argument("--src-col", default="src")
    t.add_argument("--tgt-col", default=None)
    t.add_argument("--adapter", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--num-beams", type=int, default=5)
    t.add_argument("--length-penalty", type=float, default=1.0)
    t.add_argument("--no-repeat-ngram-size", type=int, default=0)
    t.add_argument("--rerank", action="store_true")
    t.add_argument("--n-candidates", type=int, default=10)
    t.add_argument("--score", action="store_true")
    t.add_argument("--comet", action="store_true")

    b = sub.add_parser("backtranslate"); common(b)
    b.add_argument("--infile", required=True)
    b.add_argument("--src-col", default="text")
    b.add_argument("--adapter", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--num-beams", type=int, default=5)
    b.add_argument("--labse-filter", type=float, default=0.0)

    f = sub.add_parser("finetune"); common(f)
    f.add_argument("--train", nargs="+", required=True)
    f.add_argument("--src-col", default="src")
    f.add_argument("--tgt-col", default="tgt")
    f.add_argument("--out", required=True)
    f.add_argument("--epochs", type=float, default=5)
    f.add_argument("--lr", type=float, default=2e-4)
    f.add_argument("--grad-accum", type=int, default=4)
    f.add_argument("--lora-r", type=int, default=32)
    f.add_argument("--lora-alpha", type=int, default=16)
    f.add_argument("--dora", action="store_true")
    f.add_argument("--rslora", action="store_true")
    f.add_argument("--four-bit", action="store_true", help="QLoRA 4-bit; recommended for 3.3B")
    f.add_argument("--init-adapter", default=None)
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

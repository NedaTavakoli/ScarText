#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cardiac_mri_llm_5class.py
==========================
LLM-based classification of cardiac MRI reports into 5 LGE-characterised
scar phenotype classes using Llama-3.1 via Ollama (free, local deployment).

Classes:  ischemic | non-ischemic | mixed | no-scar | not-specified

Reports with cardiac devices or without gadolinium contrast are EXCLUDED
before LLM classification (identical to the rule-based 5-class pipeline).

Requirements:
    pip install ollama pandas openpyxl scikit-learn tqdm
    ollama pull llama3.1:8b      # ~4.7 GB, free
    ollama pull llama3.1:70b     # ~40 GB, free (requires A100 class GPU)

Usage:
    # Test on 50 reports
    python cardiac_mri_llm_5class.py --n_test 50

    # Full run against rule-based 5-class output
    python cardiac_mri_llm_5class.py --input_excel rowlevel_report_classification_5class.xlsx

    # Evaluate against existing labels
    python cardiac_mri_llm_5class.py --eval_only --pred_excel llm_5class_results.xlsx
"""

import os, re, sys, argparse, json, time
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("WARNING: ollama package not installed. Run: pip install ollama")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.environ.get("CMR_DATA_DIR", "data")   # folder with the report and annotation files (not distributed)
IN_EXCEL     = os.path.join(BASE_DIR, "rowlevel_report_classification_5class.xlsx")
OUT_EXCEL    = os.path.join(BASE_DIR, "llm_5class_results.xlsx")
SOURCE_EXCEL = os.path.join(BASE_DIR, "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx")

CLASSES     = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]
MODEL_8B    = "llama3.1:8b"
MODEL_70B   = "llama3.1:70b"

# ─────────────────────────────────────────────────────────────────────────────
# System prompt: 5-class CoT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a board-certified cardiovascular radiologist and cardiologist with 15 years of CMR reporting experience.

Your task: classify a cardiac MRI report into EXACTLY ONE of 5 scar phenotype classes based on LGE (late gadolinium enhancement) findings.

=== 5-CLASS TAXONOMY ===
1. ischemic       — LGE in subendocardial OR transmural distribution within coronary territory (LAD/RCA/LCx).
                    Key: subendocardial, transmural, CAD, infarction, STEMI, coronary territory
2. non-ischemic   — LGE in mid-wall, epicardial, patchy, or RV-insertion pattern NOT following coronary territory.
                    Key: mid-wall, epicardial, myocarditis, sarcoid, HCM, DCM, amyloid, patchy, RV insertion
3. mixed          — BOTH ischemic AND non-ischemic LGE patterns present in the SAME study.
                    Key: requires positive evidence of BOTH patterns simultaneously — NOT just disease history
4. no-scar        — LGE imaging performed, gadolinium given, NO enhancement found.
                    Key: "no LGE", "no delayed enhancement", "no scar", negative viability study
5. not-specified  — LGE present, scar mentioned, but pattern cannot be classified as ischemic or non-ischemic.
                    Use ONLY when evidence is genuinely insufficient for the above 4 classes.

=== DECISION HIERARCHY (follow in order) ===
Step 1: Does FINDINGS section contain explicit "no LGE / no delayed enhancement"? → no-scar
Step 2: Scan ENTIRE report for BOTH ischemic patterns AND non-ischemic patterns → if both present: mixed
Step 3: Is LGE subendocardial/transmural in coronary territory? → ischemic
Step 4: Is LGE mid-wall/epicardial/non-coronary? → non-ischemic
Step 5: LGE/scar confirmed but pattern unclassifiable? → not-specified

=== CRITICAL FAILURE MODE WARNINGS ===
⚠ NO-SCAR FALSE POSITIVES: IMPRESSION may say "no scar" when FINDINGS show positive LGE.
  Always READ the FINDINGS section first. Trust FINDINGS over IMPRESSION if they conflict.
⚠ MIXED UNDER-PREDICTION: Don't stop at the first pattern found.
  After finding ischemic evidence, continue reading for any non-ischemic pattern and vice versa.
⚠ NOT-SPECIFIED OVERUSE: This class is for genuinely unclassifiable cases only.
  If LGE follows any recognizable pattern, assign ischemic/non-ischemic/mixed/no-scar instead.

=== OUTPUT FORMAT (strict JSON only) ===
{
  "step1_no_lge_check": "quote or 'none'",
  "step2_ischemic_evidence": "quote or 'none'",
  "step2_nonischemic_evidence": "quote or 'none'",
  "step3_mixed_check": "yes/no + reasoning",
  "classification": "ischemic|non-ischemic|mixed|no-scar|not-specified",
  "confidence": 0.0-1.0,
  "evidence": "brief quote from report supporting classification"
}"""

USER_TEMPLATE = """Classify this cardiac MRI report into one of: ischemic, non-ischemic, mixed, no-scar, not-specified.

REPORT:
{report_text}

Return ONLY valid JSON. No preamble or explanation outside the JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM inference
# ─────────────────────────────────────────────────────────────────────────────
# llm_status values that mean "no classification was obtained" -> scar_type_llm is None
# (counted as missing, never as not-specified)
FAILED_STATUSES = ("call_failed", "parse_failed", "invalid_class")


def call_ollama(report_text: str, model: str = MODEL_8B,
                max_chars: int = 5000, retries: int = 6,
                num_ctx: int = 8192) -> dict:
    """Call Ollama API and parse structured JSON response.

    Failed calls (e.g. 'Failed to connect to Ollama') are retried with exponential
    backoff; if every attempt fails, or the reply cannot be parsed into one of the 5
    classes, scar_type_llm is None and llm_status says why. Earlier versions scored
    these cases as 'not-specified', which inflated that class.
    """
    text = str(report_text)[:max_chars]
    prompt = USER_TEMPLATE.format(report_text=text)
    options = {"temperature": 0.0, "seed": 42, "num_predict": 512}
    if num_ctx:
        options["num_ctx"] = num_ctx

    def failed(status, evidence, raw="", resp=None):
        return {"scar_type_llm": None, "llm_confidence": None,
                "llm_evidence": evidence, "llm_raw": raw[:500], "llm_status": status,
                **_diag(resp)}

    for attempt in range(retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options=options,
            )
        except Exception as e:
            if attempt < retries:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            return failed("call_failed", f"ERROR: {str(e)[:100]}")

        raw = resp["message"]["content"].strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: regex extract classification from raw text.
            # No retry: at temperature 0 the same prompt returns the same text.
            m = re.search(
                r'"classification"\s*:\s*"(ischemic|non-ischemic|mixed|no-scar|not-specified)"',
                raw, re.IGNORECASE)
            if m:
                return {"scar_type_llm": m.group(1).lower(),
                        "llm_confidence": 0.5,
                        "llm_evidence": "JSON parse failed — regex fallback",
                        "llm_raw": raw[:500], "llm_status": "regex_fallback",
                        **_diag(resp)}
            return failed("parse_failed", "JSON parse failed — no classification found",
                          raw, resp)

        cls = str(parsed.get("classification", "")).strip().lower()
        if cls not in CLASSES:
            return failed("invalid_class", f"invalid classification: {cls[:50]}", raw, resp)
        try:
            conf = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = None
        return {
            "scar_type_llm": cls,
            "llm_confidence": conf,
            "llm_evidence":   str(parsed.get("evidence", ""))[:300],
            "llm_raw":        raw[:500],
            "llm_status":     "json_ok",
            **_diag(resp),
        }


def _diag(resp) -> dict:
    """Token counts from the Ollama response (prompt truncation / output cut-off checks)."""
    if resp is None:
        return {"prompt_tokens": None, "output_tokens": None, "done_reason": None}
    return {"prompt_tokens": resp.get("prompt_eval_count"),
            "output_tokens": resp.get("eval_count"),
            "done_reason":   resp.get("done_reason")}


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing guards (identical to 7-class v6)
# ─────────────────────────────────────────────────────────────────────────────
_DEVICE_KW = ['pacemaker','pacer','implantable cardioverter','crt-d','crt-p',
               'biventricular pacemaker','device lead','pulse generator']

def _is_device(t: str) -> bool:
    tl = t.lower()
    return any(k in tl for k in _DEVICE_KW) or bool(re.search(r'\bICD\b|\bCRT-[DP]\b', tl))

def _no_contrast(t: str) -> bool:
    tl = t.lower()
    no_phrases = ['no gadolinium contrast was administered','no contrast was administered',
                  'without gadolinium','non-contrast study']
    no_headers = ['late gadolinium enhancement','delayed enhancement','viability','delayed imaging']
    return any(p in tl for p in no_phrases) or not any(h in tl for h in no_headers)

def postprocess(row: dict, text: str) -> dict:
    """Exclude device/no-contrast reports; fix known LLM failure modes."""
    row['scar_type_llm_preguard'] = row.get('scar_type_llm')
    row['guard_triggered'] = False
    # These should never appear in 5-class input, but guard anyway
    if _is_device(text):
        row['scar_type_llm'] = None  # exclude
        row['llm_status'] = 'excluded_device'
        return row
    if _no_contrast(text):
        row['scar_type_llm'] = None  # exclude
        row['llm_status'] = 'excluded_no_contrast'
        return row
    # No classification obtained: keep as missing, never as not-specified
    if row.get('llm_status') in FAILED_STATUSES:
        return row

    tl = text.lower()
    cls = row.get('scar_type_llm', 'not-specified')

    # Guard: if model says no-scar but FINDINGS has positive LGE patterns
    if cls == 'no-scar':
        pos_signals = ['subendocardial','transmural','mid-wall','midwall','epicardial',
                       'patchy lge','scar','fibrosis','lge present','enhancement present']
        if any(s in tl for s in pos_signals):
            # Check if negative phrase is only in IMPRESSION (heuristic)
            findings_match = re.search(r'findings?.*?(?=impression|$)', tl, re.DOTALL)
            if findings_match and any(s in findings_match.group() for s in pos_signals):
                row['scar_type_llm'] = 'not-specified'
                row['llm_evidence'] = 'GUARD: positive LGE found in FINDINGS despite no-scar prediction'
                row['guard_triggered'] = True

    # Ensure valid class
    if row.get('scar_type_llm') not in CLASSES:
        row['scar_type_llm'] = 'not-specified'

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, pred_col: str = 'scar_type_llm',
             true_col: str = 'scar_type') -> None:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                  classification_report)
    df_eval = df.dropna(subset=[pred_col, true_col]).copy()
    df_eval = df_eval[df_eval[true_col].isin(CLASSES) &
                      df_eval[pred_col].isin(CLASSES)]
    if len(df_eval) == 0:
        print("No valid rows for evaluation.")
        return

    y_true = df_eval[true_col].str.lower().str.strip()
    y_pred = df_eval[pred_col].str.lower().str.strip()

    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    rep   = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)

    print("=" * 65)
    print(f"LLM EVALUATION  (5-class, n={len(df_eval)})")
    print("=" * 65)
    print(f"  Accuracy:   {acc:.3f} ({acc*100:.1f}%)")
    print(f"  Macro F1:   {rep['macro avg']['f1-score']:.3f}")
    print(f"  Macro P:    {rep['macro avg']['precision']:.3f}")
    print(f"  Macro R:    {rep['macro avg']['recall']:.3f}")
    print(f"  Cohen's κ:  {kappa:.3f}")
    print()
    print(f"  {'Class':16s}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>4}")
    print("  " + "-"*44)
    for c in CLASSES:
        r = rep[c]
        print(f"  {c:16s}  {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1-score']:6.3f} {int(r['support']):4d}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_excel',  default=IN_EXCEL,
                   help='NLP 5-class output (Row_Level sheet, no text column)')
    p.add_argument('--source_excel', default=SOURCE_EXCEL,
                   help='Original 41,501-row source Excel with cardiac MRI text (header=8)')
    p.add_argument('--output_excel', default=OUT_EXCEL)
    p.add_argument('--model',        default=MODEL_8B,
                   choices=[MODEL_8B, MODEL_70B])
    p.add_argument('--n_test',       type=int, default=None,
                   help='Limit to N reports for testing')
    p.add_argument('--stratified',   action='store_true',
                   help='Sample equally from each class for test set')
    p.add_argument('--eval_only',    action='store_true',
                   help='Skip inference, only evaluate existing predictions')
    p.add_argument('--pred_excel',   default=OUT_EXCEL,
                   help='Excel file to evaluate (--eval_only mode)')
    p.add_argument('--num_ctx',      type=int, default=8192,
                   help='Ollama context window (0 = Ollama default)')
    p.add_argument('--retries',      type=int, default=6,
                   help='Retries per failed Ollama call (exponential backoff, max 60 s)')
    p.add_argument('--checkpoint',   default=None,
                   help='CSV appended after every report; rerunning resumes from it '
                        '(default: <output_excel>.checkpoint.csv)')
    p.add_argument('--done_from',    nargs='*', default=[],
                   help='Other checkpoint CSVs (e.g. other shards) whose rows count as done')
    p.add_argument('--shard',        type=int, default=0,
                   help='This worker processes rows with row_id %% num_shards == shard')
    p.add_argument('--num_shards',   type=int, default=1,
                   help='Split remaining rows across this many workers (one per GPU)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Eval-only mode ─────────────────────────────────────────────────────
    if args.eval_only:
        print(f"Loading predictions from: {args.pred_excel}")
        df = pd.read_excel(args.pred_excel, sheet_name='Row_Level')
        evaluate(df)
        return

    # ── Load input ────────────────────────────────────────────────────────
    print(f"Loading NLP classifications: {args.input_excel}")
    df = pd.read_excel(args.input_excel, sheet_name='Row_Level')
    df = df[df['scar_type'].isin(CLASSES)].copy()
    print(f"  5-class reports: {len(df):,}")

    # ── Join report text from the original source Excel ───────────────────
    # The NLP output no longer stores cardiac MRI text (removed to reduce file
    # size). We re-attach it here by matching on row_id (= 0-based row index
    # into the source file after skipping the 8-row header).
    print(f"Loading report text from: {args.source_excel}")
    src = pd.read_excel(args.source_excel, header=8,
                        usecols=['MRN', 'cardiac MRI text'])
    src.columns = src.columns.str.strip()
    src.index.name = None
    src['row_id'] = src.index   # row_id is the 0-based index after header
    src = src[['row_id', 'cardiac MRI text']].copy()
    df = df.merge(src, on='row_id', how='left')
    missing_text = df['cardiac MRI text'].isna().sum()
    if missing_text:
        print(f"  WARNING: {missing_text} rows have no report text after join")
    else:
        print(f"  Report text joined successfully for all {len(df):,} rows")

    # ── Sampling ──────────────────────────────────────────────────────────
    if args.n_test:
        if args.stratified:
            n_per = args.n_test // len(CLASSES)
            frames = []
            for c in CLASSES:
                sub = df[df['scar_type']==c]
                frames.append(sub.sample(n=min(n_per, len(sub)), random_state=42))
            df = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(n=min(args.n_test, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Test sample: {len(df)} reports")
        print(df['scar_type'].value_counts().to_string())

    # ── Inference ────────────────────────────────────────────────────────
    if not OLLAMA_AVAILABLE:
        print("ERROR: ollama not installed. Run: pip install ollama")
        sys.exit(1)

    print(f"\nRunning {args.model} via Ollama  (temp=0.0, structured CoT, "
          f"num_ctx={args.num_ctx or 'default'}, retries={args.retries})...")
    ckpt = args.checkpoint or os.path.splitext(args.output_excel)[0] + '.checkpoint.csv'
    RES_COLS = ['row_id', 'scar_type_llm', 'scar_type_llm_preguard', 'guard_triggered',
                'llm_status', 'llm_confidence', 'llm_evidence', 'prompt_tokens',
                'output_tokens', 'done_reason', 'llm_raw']
    done = {}
    for f in [ckpt] + list(args.done_from):
        if os.path.exists(f):
            prev = pd.read_csv(f).drop_duplicates('row_id', keep='last')
            n0 = len(done)
            done.update({r['row_id']: r for r in prev.to_dict('records')
                         if r.get('llm_status') != 'call_failed'})
            print(f"  Resuming from {f}: {len(done) - n0:,} reports already classified")

    def classify(row):
        text = str(row.get('cardiac MRI text', ''))
        if _is_device(text) or _no_contrast(text):
            # postprocess() excludes these regardless of the answer; skip the GPU call
            res = {'scar_type_llm': None, 'llm_status': 'excluded', **_diag(None)}
        else:
            res = call_ollama(text, model=args.model, retries=args.retries, num_ctx=args.num_ctx)
        res = postprocess(res, text)
        res['row_id'] = row['row_id']
        return res

    def append(res):
        pd.DataFrame([res]).reindex(columns=RES_COLS).to_csv(ckpt, mode='a', index=False,
                                             header=not os.path.exists(ckpt))

    todo = df[~df['row_id'].isin(done)]
    if args.num_shards > 1:
        todo = todo[todo['row_id'] % args.num_shards == args.shard]
        print(f"  Shard {args.shard}/{args.num_shards}: {len(todo):,} reports to classify")
    n_fail = 0
    pbar = tqdm(todo.iterrows(), total=len(todo), desc="Classifying")
    for _, row in pbar:
        res = classify(row)
        append(res)
        done[res['row_id']] = res
        n_fail += res.get('llm_status') == 'call_failed'
        pbar.set_postfix(call_failed=n_fail)

    # Second pass over calls that still failed (e.g. Ollama was down for a while)
    retry_ids = [k for k, r in done.items() if r.get('llm_status') == 'call_failed'
                 and (args.num_shards == 1 or k % args.num_shards == args.shard)]
    if retry_ids:
        print(f"\nRetrying {len(retry_ids):,} failed calls...")
        for _, row in tqdm(df[df['row_id'].isin(retry_ids)].iterrows(), total=len(retry_ids)):
            res = classify(row)
            append(res)
            done[res['row_id']] = res

    # ── Merge results ────────────────────────────────────────────────────
    res_df = pd.DataFrame([{c: r.get(c) for c in RES_COLS} for r in done.values()])
    df = df.merge(res_df.drop(columns=['llm_raw']), on='row_id', how='left')
    print("\nllm_status counts:")
    print(df['llm_status'].value_counts(dropna=False).to_string())

    # ── Save ─────────────────────────────────────────────────────────────
    # Drop report text before saving — keeps file size small.
    # The text is available in the source Excel if needed.
    save_cols = [c for c in df.columns if c != 'cardiac MRI text']
    df_save = df[save_cols]
    with pd.ExcelWriter(args.output_excel, engine='openpyxl') as w:
        df_save.to_excel(w, sheet_name='Row_Level', index=False)
        # Summary
        df_eval = df.dropna(subset=['scar_type_llm'])
        df_eval = df_eval[df_eval['scar_type'].isin(CLASSES) & df_eval['scar_type_llm'].isin(CLASSES)]
        if len(df_eval):
            from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
            y_t = df_eval['scar_type'].str.lower(); y_p = df_eval['scar_type_llm'].str.lower()
            rep = classification_report(y_t, y_p, labels=CLASSES, output_dict=True, zero_division=0)
            df['llm_status'].value_counts(dropna=False).rename_axis('llm_status') \
                .reset_index(name='n').to_excel(w, sheet_name='Status', index=False)
            summary = pd.DataFrame([{
                'model': args.model, 'num_ctx': args.num_ctx,
                'n_missing': int(df['scar_type_llm'].isna().sum()),
                'n_guard_triggered': int(df['guard_triggered'].fillna(False).sum()),
                'n_classified': len(df_eval),
                'accuracy': round(accuracy_score(y_t,y_p),4),
                'macro_f1': round(rep['macro avg']['f1-score'],4),
                'kappa':    round(cohen_kappa_score(y_t,y_p),4),
            }])
            summary.to_excel(w, sheet_name='Summary', index=False)
    print(f"\n✅  Saved: {args.output_excel}")

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluate(df)


if __name__ == '__main__':
    main()

# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
# Quick test (stratified 50 reports, one per class × 10):
#   python cardiac_mri_llm_5class.py --n_test 50 --stratified
#
# Full run (8B, all 22,292 reports):
#   python cardiac_mri_llm_5class.py
#
# 70B model (requires A100+):
#   python cardiac_mri_llm_5class.py --model llama3.1:70b --n_test 200 --stratified
#
# Evaluate existing results:
#   python cardiac_mri_llm_5class.py --eval_only --pred_excel llm_5class_results.xlsx
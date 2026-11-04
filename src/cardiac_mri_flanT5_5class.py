#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cardiac_mri_flanT5_5class.py
============================
FLAN-T5 (HuggingFace) row-level 5-class scar phenotype classification.
Uses PySpark RDD mapPartitions — no pyarrow required.

Architecture
------------
  Model   : google/flan-t5-base  (or -large / -xl / local folder)
  Classes : ischemic | non-ischemic | mixed | no-scar | not-specified
  Framework: HuggingFace Transformers + PySpark (local[*])

WHY --impression_only MODE EXISTS (important for reproducibility)
-----------------------------------------------------------------
FLAN-T5-base/large has a hard 512-token input limit. The full classification
system prompt used by the LLaMA pipeline consumes ~490 of those 512 tokens,
leaving only ~22 tokens (~70 characters) for the actual report text. The model
therefore never sees the FINDINGS or IMPRESSION sections where scar patterns
are described — it classifies against a truncated technique/indication header.
This caused >94% prediction collapse to "mixed" in the full-report run
(0% non-ischemic, 0% no-scar, 0% not-specified).

The --impression_only 1 flag addresses this by:
  1. Extracting only the FINDINGS + IMPRESSION sections (~250 tokens average)
  2. Using a condensed 57-word minimal prompt (~60 tokens) instead of the
     490-token full system prompt
  3. Total input: ~310 tokens — well within the 512-token budget

This is the RECOMMENDED mode for FLAN-T5-base and -large.
For full-report classification, use FLAN-T5-XL (1024 tokens) or the
Ollama/LLaMA pipeline (cardiac_mri_llm_5class.py).

Run commands
------------
  # ── RECOMMENDED: IMPRESSION+FINDINGS only (base/large) ────────────────────
  python cardiac_mri_flanT5_5class.py \
    --model ./LLM_based/flan-t5-base \
    --use_gpu 1 --batch_size 1 \
    --impression_only 1 \
    --output_excel flanT5_impression_results.xlsx

  # ── Quick test 200 rows (impression mode) ─────────────────────────────────
  python cardiac_mri_flanT5_5class.py \
    --model ./LLM_based/flan-t5-base \
    --use_gpu 1 --batch_size 1 \
    --impression_only 1 --limit 200 \
    --output_excel flanT5_impression_test200.xlsx

  # ── Full report mode (causes collapse on base/large — negative control) ───
  python cardiac_mri_flanT5_5class.py \
    --model ./LLM_based/flan-t5-base \
    --use_gpu 1 --batch_size 1 \
    --impression_only 0 \
    --output_excel flanT5_fullreport_results.xlsx

  # ── Eval against NLP ground truth (after run) ─────────────────────────────
  python cardiac_mri_flanT5_5class.py \
    --eval_only \
    --pred_excel flanT5_impression_results.xlsx
"""

import os
import re
import argparse
import warnings
from typing import Dict, Any, List, Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, TimestampType
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.environ.get("CMR_DATA_DIR", "data")   # folder with the report and annotation files (not distributed)
SOURCE_EXCEL = os.path.join(
    BASE_DIR,
    "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx"
)
NLP_EXCEL    = os.path.join(BASE_DIR, "rowlevel_report_classification_5class.xlsx")
OUT_EXCEL    = os.path.join(BASE_DIR, "flanT5_5class_results.xlsx")

HF_CACHE = os.path.join(BASE_DIR, ".hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)

os.environ["HF_HOME"]                  = HF_CACHE
os.environ["TRANSFORMERS_CACHE"]       = HF_CACHE
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

APP_NAME = "CardiacMRI_FlanT5_5Class"

# ─────────────────────────────────────────────────────────────────────────────
# 5-class taxonomy
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
#
# FULL_PROMPT   — identical to the LLaMA pipeline (~490 tokens).
#   impression_only=0.  Causes collapse on base/large (only ~70 chars of
#   report visible after 512-token truncation).
#
# MINIMAL_PROMPT — condensed 57-word version (~60 tokens).
#   impression_only=1 (default).  Leaves ~396 tokens for FINDINGS+IMPRESSION.
# ─────────────────────────────────────────────────────────────────────────────
FULL_PROMPT = """You are a cardiovascular radiologist classifying cardiac MRI reports by LGE scar pattern.

CLASSES (pick exactly one):
1. ischemic      — LGE is subendocardial OR transmural in coronary territory (LAD/RCA/LCx).
                   Triggers: subendocardial, transmural, infarct, infarction, CAD, STEMI, NSTEMI,
                   coronary distribution, LAD/RCA/LCX territory, embolic infarct, thromboembolic.
2. non-ischemic  — LGE is mid-wall, epicardial, patchy, mesocardial, nonvascular, or RV-insertion.
                   Triggers: mid-wall, midwall, epicardial, patchy, mesocardial, nonvascular,
                   mid myocardial, RV insertion, myocarditis, sarcoid, amyloid, HCM, DCM,
                   hypertrophic cardiomyopathy, dilated cardiomyopathy, infiltrative, inflammatory.
3. mixed         — BOTH ischemic AND non-ischemic LGE patterns in the SAME study.
4. no-scar       — Gadolinium given, NO enhancement found.
                   Triggers: no LGE, no delayed enhancement, no late gadolinium enhancement,
                   no obvious myocardial scar, no convincing myocardial scar, no macroscopic scar,
                   absent delayed enhancement, without LGE, blood pool artifact (not scar).
5. not-specified — LGE/scar present but pattern genuinely unclassifiable.

RULES:
- If FINDINGS says no LGE -> no-scar (even if IMPRESSION is vague).
- Blood pool artifact = no-scar (NOT ischemic or not-specified).
- Mesocardial / nonvascular / mid myocardial = non-ischemic (NEVER ischemic).
- If both ischemic AND non-ischemic evidence present -> mixed.
- not-specified only when pattern truly cannot be determined.

Return ONLY one word from: ischemic, non-ischemic, mixed, no-scar, not-specified"""

# Minimal prompt: ~57 words / ~60 tokens
# Designed for FLAN-T5-base/large: leaves ~396 tokens for FINDINGS+IMPRESSION (~250 tokens avg)
MINIMAL_PROMPT = """Classify cardiac MRI report by LGE scar pattern. Return ONLY one label:
ischemic: subendocardial/transmural LGE; CAD; infarct; coronary territory; embolic
non-ischemic: mid-wall; epicardial; patchy; mesocardial; myocarditis; sarcoid; HCM; DCM
mixed: BOTH ischemic AND non-ischemic patterns present
no-scar: no LGE; no delayed enhancement; no scar; blood pool artifact
not-specified: scar present, pattern unclassifiable

Labels: ischemic, non-ischemic, mixed, no-scar, not-specified"""


# ─────────────────────────────────────────────────────────────────────────────
# IMPRESSION + FINDINGS extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_impression_and_findings(text: str, max_chars: int = 1200) -> str:
    """
    Extract the FINDINGS and IMPRESSION sections from a CMR report.

    Rationale
    ---------
    CMR reports follow a structured format:
      PROCEDURE / INDICATION / TECHNIQUE / MEASUREMENTS -> FINDINGS -> IMPRESSION

    Only FINDINGS and IMPRESSION contain the scar pattern description.
    The preceding sections consume valuable token budget in small seq2seq models.

    On FLAN-T5-base (512-token limit), the full 490-token system prompt leaves
    only ~22 tokens (~70 chars) for the report — capturing only the
    INDICATION/TECHNIQUE header, never the scar description.

    This function extracts the relevant sections (~250 tokens average across
    41,501 reports) so the model can read and classify the actual scar pattern.

    Fallback: if neither section is found, returns the last max_chars characters
    of the report (IMPRESSION is almost always at the end of CMR reports).

    Parameters
    ----------
    text      : full report text
    max_chars : hard cap on extracted text (default 1200 chars ~ 300 tokens)

    Returns
    -------
    str : extracted FINDINGS + IMPRESSION, capped at max_chars
    """
    text = str(text)
    parts = []

    # Extract FINDINGS section
    findings_match = re.search(
        r'\bFINDINGS\s*[:\-]?\s*(.*?)(?=\bIMPRESSION\b|\bFINAL REPORT\b|\bATTENDING\b|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if findings_match:
        findings_text = findings_match.group(1).strip()
        parts.append("FINDINGS: " + findings_text[:700])

    # Extract IMPRESSION section
    impression_match = re.search(
        r'\bIMPRESSION\s*[:\-]?\s*(.*?)(?=\bFINAL REPORT\b|\bATTENDING\b|\bDATE SIGNED\b|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if impression_match:
        impression_text = impression_match.group(1).strip()
        parts.append("IMPRESSION: " + impression_text[:500])

    if parts:
        return "\n\n".join(parts)[:max_chars]

    # Fallback: last max_chars of report
    return text[-max_chars:].strip()


def build_prompt(report_text: str, impression_only: bool = True) -> str:
    """
    Build the full prompt for FLAN-T5.

    impression_only=True  (default, recommended for base/large):
        MINIMAL_PROMPT (~60 tokens) + extracted FINDINGS+IMPRESSION (~250 tokens)
        Total: ~310 tokens — fits within 512-token limit

    impression_only=False (for XL/XXL or negative-control experiments):
        FULL_PROMPT (~490 tokens) + truncated full report
        On base/large: only ~70 chars of report visible — causes prediction collapse
    """
    system = MINIMAL_PROMPT if impression_only else FULL_PROMPT
    return f"{system}\n\nREPORT:\n{report_text}\n\nClassification:"


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_ALIASES = {
    "ischemic":       "ischemic",
    "non-ischemic":   "non-ischemic",
    "non ischemic":   "non-ischemic",
    "nonischemic":    "non-ischemic",
    "non_ischemic":   "non-ischemic",
    "mixed":          "mixed",
    "no-scar":        "no-scar",
    "no scar":        "no-scar",
    "noscar":         "no-scar",
    "no_scar":        "no-scar",
    "not-specified":  "not-specified",
    "not specified":  "not-specified",
    "notspecified":   "not-specified",
    "not_specified":  "not-specified",
}


def parse_class(raw_text: str) -> str:
    txt = raw_text.strip().lower()
    if txt in _CLASS_ALIASES:
        return _CLASS_ALIASES[txt]
    for token in ["non-ischemic", "non ischemic", "nonischemic",
                  "mixed", "ischemic",
                  "no-scar", "no scar", "noscar",
                  "not-specified", "not specified"]:
        if token in txt:
            return _CLASS_ALIASES.get(token, token)
    return "not-specified"


def safe_parse(raw_text: str) -> Dict[str, Any]:
    return {
        "scar_type_llm": parse_class(raw_text),
        "llm_raw":       raw_text[:200],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Postprocess guards
# ─────────────────────────────────────────────────────────────────────────────
_DEVICE_KW = [
    "pacemaker", "pacer", "implantable cardioverter", "crt-d", "crt-p",
    "biventricular pacemaker", "device lead", "pacemaker lead", "icd lead",
    "defibrillator lead", "pacing lead", "pulse generator",
    "cardiac resynchronization",
]

def _is_device(t: str) -> bool:
    tl = t.lower()
    return any(k in tl for k in _DEVICE_KW) or bool(re.search(r'\bICD\b|\bCRT-[DP]\b', t))


_NO_CONTRAST_PHRASES = [
    "no gadolinium contrast was administered",
    "no contrast was administered",
    "without gadolinium contrast",
    "non-contrast study",
    "contrast not given",
    "no intravenous contrast",
]
_LGE_POSITIVE_HEADERS = [
    "late gadolinium enhancement", "late gadolinium", "delayed enhancement",
    "delayed contrast", "viability", "gadavist", "gadolinium-based contrast",
    "intravenous injection", "gbca",
]

def _is_no_contrast(t: str) -> bool:
    tl = t.lower()
    if any(p in tl for p in _LGE_POSITIVE_HEADERS):
        return False
    return any(p in tl for p in _NO_CONTRAST_PHRASES)


_POSITIVE_LGE_SIGNALS = [
    "subendocardial", "transmural", "mid-wall", "midwall", "epicardial",
    "mesocardial", "nonvascular", "mid myocardial", "patchy lge",
    "scar", "fibrosis", "lge present", "enhancement present",
    "delayed enhancement is present", "late gadolinium enhancement is present",
]

def postprocess(result: dict, full_text: str) -> dict:
    """Apply safety guards using the FULL report text (even in impression_only mode)."""
    if _is_device(full_text):
        result["scar_type_llm"] = None
        result["llm_raw"] = "EXCLUDED: cardiac device"
        return result

    if _is_no_contrast(full_text):
        result["scar_type_llm"] = None
        result["llm_raw"] = "EXCLUDED: no contrast"
        return result

    tl  = full_text.lower()
    cls = result.get("scar_type_llm", "not-specified")

    if cls == "no-scar":
        findings_section = re.search(
            r'findings?\s*[:\n].*?(?=impression\s*[:\n]|$)', tl, re.DOTALL
        )
        region = findings_section.group() if findings_section else tl
        if any(s in region for s in _POSITIVE_LGE_SIGNALS):
            result["scar_type_llm"] = "not-specified"
            result["llm_raw"] = "GUARD: positive LGE in FINDINGS despite no-scar prediction"

    if result.get("scar_type_llm") not in CLASSES + [None]:
        result["scar_type_llm"] = "not-specified"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Spark helpers
# ─────────────────────────────────────────────────────────────────────────────
def initialize_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName(APP_NAME)
        .master("local[*]")
        .config("spark.driver.memory", "12G")
        .config("spark.driver.maxResultSize", "4G")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def load_source_excel(spark: SparkSession, excel_path: str, limit: int = 0):
    df_pd = pd.read_excel(excel_path, header=8)
    required = ["MRN", "cardiac MRI text", "accession number", "procedure date"]
    missing = [c for c in required if c not in df_pd.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_pd = df_pd[required].copy()
    df_pd = df_pd.dropna(subset=["cardiac MRI text"]).reset_index(drop=True)

    if limit and limit > 0:
        df_pd = df_pd.head(limit).copy()

    df_pd["row_id"]           = np.arange(len(df_pd), dtype=np.int64)
    df_pd["MRN"]              = df_pd["MRN"].astype(str)
    df_pd["accession number"] = df_pd["accession number"].astype(str)
    df_pd["cardiac MRI text"] = df_pd["cardiac MRI text"].astype(str)
    df_pd["procedure date"]   = pd.to_datetime(df_pd["procedure date"], errors="coerce")

    schema = StructType([
        StructField("MRN",               StringType(),    True),
        StructField("cardiac MRI text",  StringType(),    True),
        StructField("accession number",  StringType(),    True),
        StructField("procedure date",    TimestampType(), True),
        StructField("row_id",            LongType(),      False),
    ])
    return spark.createDataFrame(df_pd, schema=schema)


# ─────────────────────────────────────────────────────────────────────────────
# Inference  (RDD mapPartitions — loads model ONCE per partition)
# ─────────────────────────────────────────────────────────────────────────────
def make_infer_fn(
    model_name:      str,
    max_new_tokens:  int,
    batch_size:      int,
    use_gpu:         int,
    max_input_chars: int,
    impression_only: bool,
):
    def infer_partition(rows: Iterable[Row]) -> Iterable[Row]:
        import os as _os
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        is_local = _os.path.isdir(model_name)
        device   = "cuda" if (torch.cuda.is_available() and use_gpu == 1) else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=HF_CACHE, local_files_only=is_local,
        )

        load_kwargs = dict(cache_dir=HF_CACHE, local_files_only=is_local)
        if device == "cuda":
            load_kwargs["dtype"] = torch.float16  # updated from deprecated torch_dtype

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs).to(device)
        model.eval()

        def _generate(prompts: List[str]) -> List[str]:
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                             do_sample=False, num_beams=1)
                else:
                    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                         do_sample=False, num_beams=1)
            return tokenizer.batch_decode(out, skip_special_tokens=True)

        buf_rows:    List[Row] = []
        buf_prompts: List[str] = []
        buf_texts:   List[str] = []

        def flush():
            nonlocal buf_rows, buf_prompts, buf_texts
            if not buf_rows:
                return
            decoded = _generate(buf_prompts)
            for row, raw, full_text in zip(buf_rows, decoded, buf_texts):
                result = safe_parse(raw)
                result = postprocess(result, full_text)
                yield Row(
                    MRN               = str(row["MRN"]) if row["MRN"] is not None else None,
                    **{
                        "cardiac MRI text": row["cardiac MRI text"],
                        "accession number": str(row["accession number"])
                                            if row["accession number"] is not None else None,
                        "procedure date"  : row["procedure date"],
                        "row_id"          : int(row["row_id"]),
                        "scar_type_flanT5": result.get("scar_type_llm"),
                        "flanT5_raw"      : result.get("llm_raw", ""),
                    }
                )
            buf_rows, buf_prompts, buf_texts = [], [], []

        for row in rows:
            full_text = row["cardiac MRI text"] if row["cardiac MRI text"] else ""

            # ── KEY DIFFERENCE: impression_only mode ─────────────────────────
            if impression_only:
                # Extract FINDINGS + IMPRESSION only (~250 tokens average)
                # Minimal prompt (~60 tokens) + this = ~310 tokens total
                report_for_model = extract_impression_and_findings(
                    full_text, max_chars=max_input_chars
                )
            else:
                # Full report, truncated to max_input_chars
                # WARNING: full system prompt (~490 tokens) + 512-token limit
                # = only ~70 chars of report visible — causes prediction collapse
                report_for_model = full_text[:max_input_chars] if max_input_chars else full_text

            prompt = build_prompt(report_for_model, impression_only=impression_only)

            buf_rows.append(row)
            buf_prompts.append(prompt)
            buf_texts.append(full_text)  # FULL text always used for postprocess guards

            if len(buf_rows) >= batch_size:
                yield from flush()

        yield from flush()

    return infer_partition


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  (vs NLP 5-class ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame,
             pred_col: str = "scar_type_flanT5",
             true_col: str = "scar_type") -> None:
    try:
        from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                     classification_report)
    except ImportError:
        print("scikit-learn not installed — skipping evaluation.")
        return

    df_eval = df.dropna(subset=[pred_col, true_col]).copy()
    df_eval = df_eval[
        df_eval[true_col].isin(CLASSES) & df_eval[pred_col].isin(CLASSES)
    ]
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
    print(f"FLAN-T5 EVALUATION  (5-class, n={len(df_eval):,})")
    print("=" * 65)
    print(f"  Accuracy  : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Macro F1  : {rep['macro avg']['f1-score']:.3f}")
    print(f"  Macro P   : {rep['macro avg']['precision']:.3f}")
    print(f"  Macro R   : {rep['macro avg']['recall']:.3f}")
    print(f"  Cohen's k : {kappa:.3f}")
    print()
    print(f"  {'Class':16s}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("  " + "-" * 46)
    for c in CLASSES:
        r = rep[c]
        print(f"  {c:16s}  {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1-score']:6.3f} {int(r['support']):5d}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────────────────────
def export_excel(df: pd.DataFrame, out_path: str, include_text: bool = False,
                 impression_only: bool = True) -> None:
    base_cols = ["row_id", "MRN", "procedure date", "accession number",
                 "scar_type_flanT5", "flanT5_raw"]
    if include_text:
        base_cols.append("cardiac MRI text")
    cols = [c for c in base_cols if c in df.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df[cols].to_excel(w, sheet_name="Row_Level", index=False)

        summary = (
            df["scar_type_flanT5"]
            .value_counts(dropna=False)
            .rename_axis("class")
            .reset_index(name="count")
        )
        summary["percent"] = summary["count"] / len(df) * 100.0
        summary.to_excel(w, sheet_name="Class_Summary", index=False)

        # Run parameters sheet — for reproducibility
        meta = pd.DataFrame([
            {"parameter": "impression_only",
             "value": str(impression_only),
             "note": "True = FINDINGS+IMPRESSION extracted; False = full report truncated"},
            {"parameter": "prompt_mode",
             "value": "MINIMAL_PROMPT" if impression_only else "FULL_PROMPT",
             "note": "MINIMAL_PROMPT recommended for FLAN-T5-base/large (512-token limit)"},
            {"parameter": "total_rows",     "value": str(len(df)), "note": ""},
            {"parameter": "classified",
             "value": str(df["scar_type_flanT5"].notna().sum()), "note": ""},
            {"parameter": "excluded",
             "value": str(df["scar_type_flanT5"].isna().sum()),
             "note": "Device or no-contrast exclusion"},
        ])
        meta.to_excel(w, sheet_name="Run_Parameters", index=False)

    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="FLAN-T5 5-class cardiac MRI scar phenotype classifier"
    )
    p.add_argument("--model", type=str, default="google/flan-t5-base")
    p.add_argument("--source_excel", default=SOURCE_EXCEL)
    p.add_argument("--output_excel", default=OUT_EXCEL)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N rows (0 = all).")
    p.add_argument("--use_gpu", type=int, default=1)
    p.add_argument("--partitions", type=int, default=0)
    p.add_argument("--max_input_chars", type=int, default=1200,
                   help=(
                       "impression_only=1: cap on extracted FINDINGS+IMPRESSION text. "
                       "impression_only=0: truncation of raw full report."
                   ))
    p.add_argument("--impression_only", type=int, default=1,
                   help=(
                       "1 (default, RECOMMENDED): extract FINDINGS+IMPRESSION + minimal prompt. "
                       "Fits within FLAN-T5-base/large 512-token limit. "
                       "0: full report + full system prompt. Causes prediction collapse on "
                       "base/large (only ~70 chars of report visible after truncation)."
                   ))
    p.add_argument("--include_text", type=int, default=0)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--pred_excel", default=OUT_EXCEL)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    impression_only = bool(args.impression_only)

    if args.eval_only:
        print(f"Loading predictions from: {args.pred_excel}")
        df_pred = pd.read_excel(args.pred_excel, sheet_name="Row_Level")
        if "scar_type" not in df_pred.columns:
            df_nlp = pd.read_excel(NLP_EXCEL, sheet_name=0)
            if "scar_type" in df_nlp.columns and "row_id" in df_nlp.columns:
                df_pred = df_pred.merge(df_nlp[["row_id", "scar_type"]],
                                        on="row_id", how="left")
        evaluate(df_pred)
        return

    partitions = args.partitions if args.partitions > 0 else (1 if args.use_gpu else 8)

    print("=" * 70)
    print("FLAN-T5 5-Class Cardiac MRI Scar Classifier")
    print(f"  Model           : {args.model}")
    print(f"  use_gpu         : {args.use_gpu}  |  partitions: {partitions}  |  batch: {args.batch_size}")
    print(f"  impression_only : {impression_only}")
    if impression_only:
        print(f"  prompt_mode     : MINIMAL_PROMPT (~60 tokens) + FINDINGS+IMPRESSION (~250 tokens)")
        print(f"  max_input_chars : {args.max_input_chars} chars (cap on extracted sections)")
    else:
        print(f"  prompt_mode     : FULL_PROMPT (~490 tokens) + full report")
        print(f"  WARNING: full-report mode leaves only ~70 chars of report visible on base/large")
    print(f"  max_new_tokens  : {args.max_new_tokens}")
    print(f"  limit           : {args.limit if args.limit else 'ALL'}")
    print(f"  output          : {args.output_excel}")
    print("=" * 70)

    spark = initialize_spark()

    try:
        df_spark = load_source_excel(spark, args.source_excel, limit=args.limit)
        n_rows   = df_spark.count()
        print(f"Rows to classify: {n_rows:,}")

        df_spark = df_spark.coalesce(partitions)

        out_schema = StructType(
            df_spark.schema.fields + [
                StructField("scar_type_flanT5", StringType(), True),
                StructField("flanT5_raw",        StringType(), True),
            ]
        )

        infer_fn = make_infer_fn(
            model_name      = args.model,
            max_new_tokens  = args.max_new_tokens,
            batch_size      = args.batch_size,
            use_gpu         = args.use_gpu,
            max_input_chars = args.max_input_chars,
            impression_only = impression_only,
        )

        rdd_out = df_spark.rdd.mapPartitions(infer_fn)
        df_out  = spark.createDataFrame(rdd_out, schema=out_schema)
        df_pd   = df_out.toPandas()

        print("\n── Classification breakdown ───────────────────────────────")
        print(df_pd["scar_type_flanT5"].value_counts(dropna=False).to_string())

        excluded   = df_pd["scar_type_flanT5"].isna().sum()
        classified = (~df_pd["scar_type_flanT5"].isna()).sum()
        print(f"\n  Excluded (device/no-contrast): {excluded:,}")
        print(f"  Classified:                    {classified:,}")

        export_excel(df_pd, args.output_excel,
                     include_text=bool(args.include_text),
                     impression_only=impression_only)

        if "scar_type" in df_pd.columns:
            evaluate(df_pd)

    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

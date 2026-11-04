#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cardiac_mri_llm_fewshot_all.py
===============================
Few-shot in-context learning for ALL LLM models in the CMR scar phenotyping
benchmark: Llama-3.1-8B, Llama-3.1-70B (via Ollama), and MedGemma-4B-IT
(via HuggingFace Transformers).

FlanT5-base is NOT included — its 512-token hard limit cannot accommodate
even one full in-context example alongside the report text. This is noted
explicitly in the output as "not applicable — architectural token limit."

Addresses Ulas Bagci review comment (April 2026):
  "Add few-shot prompting experiments for the LLMs. Testing Llama-3.1-8B
   with k=3 or k=5 examples in-context directly addresses the zero-shot-only
   criticism and gives a more complete picture of where LLMs actually land."

Design
------
  - IDENTICAL 75/25 stratified split (seed=42) as ALL other experiments.
    Loads from train_test_split_expert.xlsx if present (nlp_fewshot.py output).
  - Few-shot examples retrieved per query via TF-IDF k-NN over training set
    (same retrieval strategy as nlp_fewshot.py for methodological consistency).
  - Each example is formatted with the FULL structured JSON answer to match
    the model's expected output format exactly.
  - k=0 (zero-shot baseline), k=3, k=5 supported.
  - Same post-processing guard as zero-shot pipeline.
  - Output mirrors Table 5 structure for direct manuscript insertion.

Usage
-----
  # Ollama is already running on your server — skip "ollama serve"
  # if you see "address already in use", it just means it's running fine.

  # Smoke test first (5 reports, k=3, Llama 8B only)
  python cardiac_mri_llm_fewshot_all.py --models llama8b --k 3 --n_test 5

  # Full held-out test set (n=66), Llama 8B, k=3  (~9 min)
  python cardiac_mri_llm_fewshot_all.py --models llama8b --k 3

  # Llama 8B k=5  (~11 min)
  python cardiac_mri_llm_fewshot_all.py --models llama8b --k 5

  # Llama 70B k=3  (~35 min)
  python cardiac_mri_llm_fewshot_all.py --models llama70b --k 3

  # MedGemma k=3 (air-gapped server)
  TRANSFORMERS_OFFLINE=1 python cardiac_mri_llm_fewshot_all.py \\
      --models medgemma --k 3 \\
      --local_model_dir <path-to>/medgemma-4b-it

  # All Ollama models together, k=3
  python cardiac_mri_llm_fewshot_all.py \\
      --models llama8b llama70b --k 3 --out fewshot_ollama_k3.xlsx

  # Resume after disconnect
  python cardiac_mri_llm_fewshot_all.py --models llama8b --k 3 --resume

Requirements
------------
  pip install ollama transformers torch accelerate bitsandbytes
  pip install pandas openpyxl scikit-learn scipy tqdm
  ollama pull llama3.1:8b
  ollama pull llama3.1:70b   # optional
"""

import os, re, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              classification_report, confusion_matrix)
from scipy.stats import bootstrap
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

# Import ollama at top level — same pattern as cardiac_mri_llm_5class.py
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("WARNING: ollama package not installed. Run: pip install ollama")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.environ.get("CMR_DATA_DIR", "data")   # folder with the report and annotation files (not distributed)
EXPERT_XLSX   = os.path.join(BASE_DIR, "Two_Expert_Annotation_Comparision.xlsx")
SOURCE_XLSX   = os.path.join(BASE_DIR,
    "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx")
SPLIT_XLSX    = os.path.join(BASE_DIR, "train_test_split_expert.xlsx")

CLASSES       = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]
RANDOM_STATE  = 42
TEST_SIZE     = 0.25
MAX_CHARS     = 5000   # report truncation for query
EX_CHARS      = 1500   # report truncation per few-shot example
CHECKPOINT    = 20     # save every N predictions

# Model tags
LLAMA_8B      = "llama3.1:8b"
LLAMA_70B     = "llama3.1:70b"
MEDGEMMA_HF   = "google/medgemma-4b-it"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (identical to cardiac_mri_llm_5class.py for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a board-certified cardiovascular radiologist and cardiologist with \
15 years of CMR reporting experience.

Your task: classify a cardiac MRI report into EXACTLY ONE of 5 scar phenotype \
classes based on LGE (late gadolinium enhancement) findings.

=== 5-CLASS TAXONOMY ===
1. ischemic       — LGE in subendocardial OR transmural distribution within \
coronary territory (LAD/RCA/LCx).
                    Key: subendocardial, transmural, CAD, infarction, STEMI, \
coronary territory
2. non-ischemic   — LGE in mid-wall, epicardial, patchy, or RV-insertion \
pattern NOT following coronary territory.
                    Key: mid-wall, epicardial, myocarditis, sarcoid, HCM, \
DCM, amyloid, patchy, RV insertion
3. mixed          — BOTH ischemic AND non-ischemic LGE patterns present in \
the SAME study.
                    Requires positive evidence of BOTH patterns simultaneously.
4. no-scar        — LGE imaging performed, gadolinium given, NO enhancement found.
                    Key: "no LGE", "no delayed enhancement", "no scar", \
negative viability
5. not-specified  — LGE present, scar mentioned, but pattern cannot be \
classified. Use ONLY as last resort.

=== DECISION HIERARCHY (follow in order) ===
Step 1: Explicit "no LGE / no delayed enhancement" in FINDINGS? → no-scar
Step 2: BOTH ischemic AND non-ischemic evidence present? → mixed
Step 3: Subendocardial/transmural LGE in coronary territory? → ischemic
Step 4: Mid-wall/epicardial/non-coronary LGE? → non-ischemic
Step 5: LGE confirmed but pattern unclassifiable? → not-specified

=== CRITICAL FAILURE MODE WARNINGS ===
⚠ FINDINGS overrides IMPRESSION — always trust FINDINGS over summary.
⚠ After finding ischemic evidence, keep reading for non-ischemic pattern too.
⚠ Do NOT default to not-specified when evidence points clearly to one class.

=== OUTPUT FORMAT (strict JSON only, no preamble) ===
{
  "step1_no_lge_check": "exact quote or none",
  "step2_ischemic_evidence": "exact quote or none",
  "step2_nonischemic_evidence": "exact quote or none",
  "step3_mixed_check": "yes/no + one-line reasoning",
  "classification": "ischemic|non-ischemic|mixed|no-scar|not-specified",
  "confidence": 0.0-1.0,
  "evidence": "brief quote from report supporting classification"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Few-shot in-context LLM inference — all models")
    p.add_argument('--expert_xlsx',      default=EXPERT_XLSX)
    p.add_argument('--source_xlsx',      default=SOURCE_XLSX)
    p.add_argument('--split_xlsx',       default=SPLIT_XLSX)
    p.add_argument('--models', nargs='+',
                   default=['llama8b'],
                   choices=['llama8b', 'llama70b', 'medgemma'],
                   help='Models to run (default: llama8b)')
    p.add_argument('--k', type=int, default=3, choices=[0, 3, 5],
                   help='Few-shot examples per query (0=zero-shot baseline)')
    p.add_argument('--knn_pool', type=int, default=15,
                   help='TF-IDF retrieval pool size before class-balanced selection')
    p.add_argument('--ollama_host', default='http://localhost:11434',
                   help='Ollama host (default: http://localhost:11434)')
    p.add_argument('--local_model_dir', default=None,
                   help='Local directory for MedGemma weights (air-gapped server)')
    p.add_argument('--n_test', type=int, default=None,
                   help='Limit test set to N reports (smoke test)')
    p.add_argument('--out', default=None,
                   help='Output Excel path (auto-generated if not set)')
    p.add_argument('--resume', action='store_true',
                   help='Resume from existing output (skip already-done reports)')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--num_ctx',     type=int,   default=8192,
                   help='Ollama context window (default 8192; few-shot prompts exceed the 2,048 default)')
    p.add_argument('--retries',     type=int,   default=6,
                   help='Retries per Ollama call before recording it as missing')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — mirrors _expert_load() in cardiac_mri_nlp_5class.py exactly
# ─────────────────────────────────────────────────────────────────────────────

# Reports excluded from expert annotation (non-evaluable studies)
# Note: these six report IDs are valid expert-labeled reports that were excluded by mistake (see the paper,
# Methods and Figure 1); the list is kept unchanged so that the published 194/64 split is reproduced.
_EXPERT_EXCLUDED = {
    'RPT0006', 'RPT0042', 'RPT0089', 'RPT0134', 'RPT0198', 'RPT0231'
}

def load_expert_data(expert_xlsx, source_xlsx):
    """
    Reads Two_Expert_Annotation_Comparision.xlsx exactly as cardiac_mri_nlp_5class.py
    does via _expert_load():
      - Sheet: "Reader1_Annotation (Onur)"  or  "Reader1_Annotation"
      - Header row: index 2 (row 3 in Excel, 1-based)
      - Data starts: index 3
      - Columns used: "Report ID", "CMR Report Text", "Reader 1 Label ▼"
    Returns a DataFrame with columns [gold_label, report_text, report_id].
    """
    print(f"Loading expert annotations: {expert_xlsx}")
    xl = pd.read_excel(expert_xlsx, sheet_name=None, header=None)
    print(f"  Sheets found: {list(xl.keys())}")

    text_map, label_map = {}, {}

    for sname in ["Reader1_Annotation (Onur)", "Reader1_Annotation"]:
        if sname not in xl:
            continue
        raw = xl[sname]
        hdr = raw.iloc[2].tolist()           # row index 2 = header
        dr  = raw.iloc[3:].copy()            # row index 3+ = data
        dr.columns = hdr
        dr = dr.reset_index(drop=True)

        # Drop non-evaluable reports
        dr = dr[~dr["Report ID"].astype(str).str.strip().isin(_EXPERT_EXCLUDED)]

        for _, row in dr.iterrows():
            rpt = str(row.get("Report ID", "") or "").strip()
            if not rpt:
                continue
            # Report text
            txt = str(row.get("CMR Report Text", "") or "").strip()
            if txt and txt.lower() not in ("nan", "none", ""):
                text_map[rpt] = txt
            # Reader 1 label (try multiple possible column names)
            for col in ["Reader 1 Label ▼", "Reader 1 Label", "Reader1 Label",
                        "R1 Label", "label", "Label"]:
                v = str(row.get(col, "") or "").strip().lower()
                if v and v not in ("nan", "none", ""):
                    label_map[rpt] = v
                    break

        print(f"  Sheet '{sname}': {len(label_map)} labels, {len(text_map)} texts")
        break

    if not label_map:
        raise RuntimeError(
            "Could not load any labels from Two_Expert_Annotation_Comparision.xlsx.\n"
            "Expected sheet 'Reader1_Annotation (Onur)' with header on row 3.\n"
            f"Found sheets: {list(xl.keys())}")

    # Build DataFrame
    rows = []
    for rpt_id, label in label_map.items():
        label_clean = label.strip().lower()
        if label_clean not in CLASSES:
            continue
        txt = text_map.get(rpt_id, "")
        rows.append({'report_id': rpt_id, 'gold_label': label_clean,
                     'report_text': txt})

    df = pd.DataFrame(rows).reset_index(drop=True)
    missing = (df['report_text'] == "").sum()
    print(f"  Valid labelled reports: {len(df)}  |  missing text: {missing}")
    if missing > 0:
        print("  WARNING: Some reports have no text. Check 'CMR Report Text' column.")
    return df


def make_split(df, split_xlsx):
    """
    Load the pre-saved split from train_test_split_expert.xlsx (produced by
    nlp_fewshot.py) to guarantee IDENTICAL train/test indices across all
    experiments.  Falls back to regenerating with the same parameters if not found.

    train_test_split_expert.xlsx has sheet 'All_Reports' with columns:
      Report ID, Split, R1 Label, ...
    (NOT 'train'/'test' sheets — those don't exist in this file.)
    """
    if split_xlsx and os.path.exists(split_xlsx):
        try:
            all_df = pd.read_excel(split_xlsx, sheet_name='All_Reports')
            all_df.columns = [c.strip().lower().replace(' ', '_')
                              for c in all_df.columns]

            # Map report_id → split label
            split_col = next((c for c in all_df.columns if 'split' in c), None)
            id_col    = next((c for c in all_df.columns
                              if 'report' in c and 'id' in c), None)
            if split_col and id_col:
                train_ids = set(all_df[all_df[split_col] == 'train'][id_col]
                                .astype(str).tolist())
                test_ids  = set(all_df[all_df[split_col] == 'test'][id_col]
                                .astype(str).tolist())
                train_df = df[df['report_id'].isin(train_ids)].reset_index(drop=True)
                test_df  = df[df['report_id'].isin(test_ids)].reset_index(drop=True)
                if len(train_df) > 0 and len(test_df) > 0:
                    print(f"Loaded pre-saved split: {len(train_df)} train | "
                          f"{len(test_df)} test")
                    return train_df, test_df
        except Exception as e:
            print(f"  Could not load split ({e}), regenerating...")

    # Regenerate with identical parameters to nlp_fewshot.py
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df['gold_label'])
    print(f"Generated split (seed={RANDOM_STATE}): "
          f"{len(train_df)} train | {len(test_df)} test")
    print("  Train dist:", train_df['gold_label'].value_counts().to_dict())
    print("  Test  dist:", test_df['gold_label'].value_counts().to_dict())
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF retriever
# ─────────────────────────────────────────────────────────────────────────────
class FewShotRetriever:
    """
    Retrieves k class-balanced few-shot examples from the training set
    using TF-IDF cosine similarity (same as nlp_fewshot.py Layer 3).
    """
    def __init__(self, train_df, knn_pool=15):
        self.train_df = train_df.copy()
        self.knn_pool = knn_pool
        self.vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=2)
        self.train_vecs = self.vec.fit_transform(
            train_df['report_text'].fillna('').tolist())
        print(f"TF-IDF built: {self.train_vecs.shape[1]:,} features, "
              f"{self.train_vecs.shape[0]} training docs")

    def retrieve(self, query_text, k):
        if k == 0:
            return pd.DataFrame(columns=['gold_label', 'report_text'])
        q = self.vec.transform([str(query_text)])
        sims = cosine_similarity(q, self.train_vecs).flatten()
        top_idx = np.argsort(sims)[::-1][:self.knn_pool]
        pool = self.train_df.iloc[top_idx].copy()

        # Class-balanced selection: at most ceil(k/5) per class first
        selected, used, counts = [], set(), {c: 0 for c in CLASSES}
        quota = max(1, k // len(CLASSES))
        for i, (_, row) in enumerate(pool.iterrows()):
            if len(selected) >= k: break
            if counts[row['gold_label']] < quota:
                selected.append(row); counts[row['gold_label']] += 1; used.add(i)
        # Pad with remaining highest-similarity rows
        for i, (_, row) in enumerate(pool.iterrows()):
            if len(selected) >= k: break
            if i not in used:
                selected.append(row); used.add(i)
        return pd.DataFrame(selected[:k]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
_EVID_SENT = re.compile(r'[^.\n]*\b(enhancement|lge|scar|fibrosis|infarct)[^.\n]*', re.I)

def _example_evidence(text):
    """Quote the first LGE-related sentence of the (truncated) example report, so the
    example's evidence field shows a real quotation instead of a copyable placeholder."""
    m = _EVID_SENT.search(str(text)[:EX_CHARS])
    return m.group(0).strip()[:200] if m else "no LGE description in report"


def _make_example_json(cls, text):
    """Structured JSON answer for a few-shot example."""
    return json.dumps({
        "step1_no_lge_check":         "none" if cls != "no-scar" else "no delayed enhancement identified",
        "step2_ischemic_evidence":    "see report" if cls in ("ischemic", "mixed") else "none",
        "step2_nonischemic_evidence": "see report" if cls in ("non-ischemic", "mixed") else "none",
        "step3_mixed_check":          "yes — both patterns present" if cls == "mixed" else "no",
        "classification":             cls,
        "confidence":                 0.95,
        "evidence":                   _example_evidence(text)
    }, indent=2)


def build_few_shot_prompt(query_text, examples_df):
    """Assemble user turn: few-shot block + query report."""
    parts = []
    if len(examples_df) > 0:
        parts.append("Study these expert-annotated examples carefully:\n")
        for _, ex in examples_df.iterrows():
            parts.append(f"=== EXAMPLE (label: {ex['gold_label']}) ===")
            parts.append(f"REPORT:\n{str(ex['report_text'])[:EX_CHARS]}\n")
            parts.append(f"ANSWER:\n{_make_example_json(ex['gold_label'], ex['report_text'])}\n")
        parts.append("=== NOW CLASSIFY THE FOLLOWING REPORT ===\n")
    parts.append(f"REPORT:\n{str(query_text)[:MAX_CHARS]}\n")
    parts.append("Return ONLY valid JSON. No preamble.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing guard  (identical to zero-shot pipeline)
# ─────────────────────────────────────────────────────────────────────────────
_POS_LGE = re.compile(
    r'\b(subendocardial|transmural|mid.?wall|epicardial|patchy\s+lge|'
    r'patchy\s+enhancement|scar|fibrosis|lge\s+present|'
    r'delayed\s+enhancement\s+present)\b', re.I)

def guard(label, text):
    if label == 'no-scar':
        findings = re.search(r'findings?.*?(?=impression|conclusion|$)',
                             text, re.I | re.DOTALL)
        if findings and _POS_LGE.search(findings.group()):
            return 'not-specified', 'GUARD: positive LGE in FINDINGS'
    return label, ''


# ─────────────────────────────────────────────────────────────────────────────
# Ollama inference  (Llama-3.1-8B and 70B)
# Uses the exact same pattern as cardiac_mri_llm_5class.py (known working)
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(model_tag, system, user, temperature=0.0, retries=6, num_ctx=8192):
    """Same decision logic as cardiac_mri_llm_5class.call_ollama: failed calls, unparseable
    replies and out-of-taxonomy classes are returned as label=None (missing), never as a class."""
    if not OLLAMA_AVAILABLE:
        print("ERROR: ollama package not available. Run: pip install ollama")
        sys.exit(1)

    for attempt in range(retries + 1):
        try:
            resp = ollama.chat(
                model=model_tag,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={"temperature": temperature, "seed": 42, "num_predict": 512,
                         "num_ctx": num_ctx},
            )
            break
        except Exception as e:
            if attempt < retries:
                wait = min(2 ** (attempt + 1), 60)
                print(f"\n  Ollama error (attempt {attempt+1}/{retries+1}): {e} — retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  Ollama FAILED after {retries+1} attempts: {type(e).__name__}: {e}")
                return {"label": None, "status": "call_failed", "confidence": 0,
                        "evidence": f"ERROR: {type(e).__name__}: {e}", "parse_error": True,
                        "prompt_tokens": None, "output_tokens": None, "done_reason": None}

    diag = {"prompt_tokens": resp.get("prompt_eval_count"),
            "output_tokens": resp.get("eval_count"),
            "done_reason":   resp.get("done_reason")}
    raw = resp["message"]["content"].strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Regex fallback — same as cardiac_mri_llm_5class.py
        m = re.search(r'"classification"\s*:\s*"'
                      r'(ischemic|non-ischemic|mixed|no-scar|not-specified)"', raw, re.IGNORECASE)
        if m:
            return {"label": m.group(1).lower(), "status": "regex_fallback", "confidence": 0.5,
                    "evidence": "JSON parse failed — regex fallback", "parse_error": True, **diag}
        return {"label": None, "status": "parse_failed", "confidence": 0,
                "evidence": raw[:300], "parse_error": True, **diag}
    cls = str(parsed.get("classification", "")).strip().lower()
    if cls not in CLASSES:
        return {"label": None, "status": "invalid_class", "confidence": 0,
                "evidence": f"invalid class: {cls[:50]}", "parse_error": True, **diag}
    try:
        conf = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"label": cls, "status": "json_ok", "confidence": conf,
            "evidence": str(parsed.get("evidence", ""))[:300], "parse_error": False, **diag}


# ─────────────────────────────────────────────────────────────────────────────
# MedGemma inference
# ─────────────────────────────────────────────────────────────────────────────
class MedGemmaRunner:
    def __init__(self, model_dir=None, temperature=0.0):
        try:
            import torch
            from transformers import pipeline, BitsAndBytesConfig
        except ImportError:
            raise ImportError("pip install transformers torch accelerate bitsandbytes")

        model_id = model_dir or MEDGEMMA_HF
        print(f"Loading MedGemma: {model_id}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        quant  = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True) \
                 if device == "cuda" else None
        self.pipe = pipeline(
            "text-generation", model=model_id,
            quantization_config=quant,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True)
        self.temperature = temperature
        print("  MedGemma loaded.")

    def __call__(self, system, user):
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user',   'content': user}]
        try:
            out = self.pipe(messages, max_new_tokens=512,
                            do_sample=(self.temperature > 0),
                            temperature=max(self.temperature, 1e-6),
                            pad_token_id=self.pipe.tokenizer.eos_token_id)
            raw = out[0]['generated_text'][-1]['content'].strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            parsed = json.loads(raw)
            cls = parsed.get('classification', '').strip().lower()
            if cls not in CLASSES:
                cls = 'not-specified'
            return {'label': cls,
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'evidence': str(parsed.get('evidence', ''))[:300],
                    'parse_error': False}
        except Exception as e:
            m = re.search(
                r'"classification"\s*:\s*"(ischemic|non-ischemic|mixed|no-scar|not-specified)"',
                locals().get('raw', ''), re.I)
            cls = m.group(1).lower() if m else 'not-specified'
            return {'label': cls, 'confidence': 0.5,
                    'evidence': f'parse error: {e}', 'parse_error': True}


# ─────────────────────────────────────────────────────────────────────────────
# Per-model inference loop
# ─────────────────────────────────────────────────────────────────────────────
def run_model(model_key, k, test_df, retriever, args):
    """Run inference for one model, return list of result dicts."""
    print(f"\n{'─'*60}")
    print(f"  Model: {model_key}  |  k={k}  |  n={len(test_df)}")
    print(f"{'─'*60}")

    # ── Ollama connectivity check ─────────────────────────────────────────────
    if model_key in ('llama8b', 'llama70b'):
        if not OLLAMA_AVAILABLE:
            print("ERROR: ollama package not installed. Run: pip install ollama")
            sys.exit(1)
        model_tag = LLAMA_8B if model_key == 'llama8b' else LLAMA_70B
        print(f"  Checking Ollama connectivity ({model_tag})...")
        try:
            test_resp = ollama.chat(
                model=model_tag,
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                options={"temperature": 0, "num_predict": 10, "num_ctx": args.num_ctx})
            test_out = test_resp["message"]["content"].strip()
            print(f"  ✓ Ollama responding: '{test_out[:50]}'")
        except Exception as e:
            print(f"  ✗ Ollama connectivity FAILED: {type(e).__name__}: {e}")
            print("  Make sure Ollama is running: ollama serve &")
            print(f"  Make sure model is pulled:  ollama pull {model_tag}")
            sys.exit(1)

    # Load checkpoint if resuming (next to the output, so parallel runs don't collide)
    ckpt_path = os.path.splitext(args.out)[0] + f"_ckpt_{model_key}_k{k}.csv"
    done_idx = set()
    results = []
    if args.resume and os.path.exists(ckpt_path):
        prev = pd.read_csv(ckpt_path)
        done_idx = set(prev['idx'].tolist())
        results = prev.to_dict('records')
        print(f"  Resuming: {len(done_idx)} already done")

    # Init model runner for MedGemma (load once, reuse)
    medgemma_runner = None
    if model_key == 'medgemma':
        medgemma_runner = MedGemmaRunner(
            model_dir=args.local_model_dir, temperature=args.temperature)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df),
                         desc=f"  {model_key} k={k}"):
        if idx in done_idx:
            continue

        text = str(row['report_text'])
        gold = str(row['gold_label'])

        # Retrieve examples
        examples = retriever.retrieve(text, k) if k > 0 else pd.DataFrame()
        user_msg = build_few_shot_prompt(text, examples)

        # Inference
        t0 = time.time()
        if model_key in ('llama8b', 'llama70b'):
            out = call_ollama(LLAMA_8B if model_key == 'llama8b' else LLAMA_70B,
                              SYSTEM_PROMPT, user_msg, args.temperature,
                              retries=args.retries, num_ctx=args.num_ctx)
        elif model_key == 'medgemma':
            out = medgemma_runner(SYSTEM_PROMPT, user_msg)
            out.update(status='json_ok' if not out['parse_error'] else 'regex_fallback',
                       prompt_tokens=None, output_tokens=None, done_reason=None)
        else:
            raise ValueError(f"Unknown model: {model_key}")
        elapsed = time.time() - t0

        # Verbose output for first report so you can see it's really working
        if len(results) == 0:
            tqdm.write(f"\n  === First report debug ===")
            tqdm.write(f"  Gold: {gold}  |  Predicted: {out['label']}  "
                       f"|  elapsed: {elapsed:.1f}s  |  parse_error: {out['parse_error']}")
            tqdm.write(f"  Evidence: {out['evidence'][:120]}")
            if elapsed < 0.1:
                tqdm.write("  WARNING: elapsed <0.1s — Ollama may not have been called!")

        # Warn loudly on parse errors
        if out['parse_error'] and 'ERROR' in str(out['evidence']):
            tqdm.write(f"  !! parse_error idx={idx}: {out['evidence'][:200]}")

        # Guard (missing predictions stay missing)
        if out['label'] is None:
            label_final, guard_note = None, ''
        else:
            label_final, guard_note = guard(out['label'], text)

        results.append({
            'idx': idx, 'gold_label': gold,
            'predicted_label': label_final,
            'raw_label': out['label'],
            'guard_triggered': label_final != out['label'],
            'guard_note': guard_note,
            'status': out['status'],
            'parse_error': out['parse_error'],
            'confidence': out['confidence'],
            'evidence': out['evidence'][:300],
            'prompt_tokens': out['prompt_tokens'],
            'output_tokens': out['output_tokens'],
            'done_reason': out['done_reason'],
            'elapsed_s': round(elapsed, 2),
        })

        # Checkpoint
        if len(results) % CHECKPOINT == 0:
            pd.DataFrame(results).to_csv(ckpt_path, index=False)
            valid = [r for r in results if r['predicted_label'] in CLASSES]
            acc = accuracy_score([r['gold_label'] for r in valid],
                                 [r['predicted_label'] for r in valid]) if valid else float('nan')
            tqdm.write(f"    [{len(results)}/{len(test_df)}] running acc={acc:.3f}  "
                       f"missing={len(results) - len(valid)}")

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(results):
    # Missing predictions (failed calls / unparseable replies) are excluded and counted
    valid  = [r for r in results if r['predicted_label'] in CLASSES]
    y_true = [r['gold_label']      for r in valid]
    y_pred = [r['predicted_label'] for r in valid]
    acc    = accuracy_score(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred, labels=CLASSES)

    def kappa_fn(yt, yp):
        try:    return cohen_kappa_score(yt, yp)
        except: return np.nan
    bstrap = bootstrap((np.array(y_true), np.array(y_pred)),
                       kappa_fn, n_resamples=1000, paired=True,
                       confidence_level=0.95, random_state=RANDOM_STATE,
                       method='percentile')
    klo, khi = bstrap.confidence_interval.low, bstrap.confidence_interval.high
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return dict(accuracy=acc, kappa=kappa, kappa_ci=(klo, khi),
                report=report, cm=cm, y_true=y_true, y_pred=y_pred,
                n_valid=len(valid), n_missing=len(results) - len(valid),
                n_correct=int(sum(t == p for t, p in zip(y_true, y_pred))))


# ─────────────────────────────────────────────────────────────────────────────
# Excel output
# ─────────────────────────────────────────────────────────────────────────────
def save_excel(all_model_results, k, args, out_path):
    """
    all_model_results: dict of {model_key: {'results': [...], 'metrics': {...}}}
    """
    wb = Workbook()
    BLUE  = PatternFill("solid", fgColor="1F4E79")
    BLUE2 = PatternFill("solid", fgColor="2E75B6")
    GREEN = PatternFill("solid", fgColor="C6EFCE")
    RED   = PatternFill("solid", fgColor="FCE4D6")
    WB    = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    BOLD  = Font(bold=True, name="Arial", size=10)

    # ── Sheet 1: Summary ─────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    n_test = len(next(iter(all_model_results.values()))['results'])
    ws['A1'] = f"Few-Shot LLM Results  |  k={k}  |  n_test={n_test}  |  split=75/25 seed=42"
    ws['A1'].font = Font(bold=True, name="Arial", size=12)
    ws['A2'] = (f"Reference: Rule-Based NLP zero-shot = 92.2% (59/64), κ=0.887  |  "
                f"num_ctx={args.num_ctx}  |  missing predictions excluded and counted")
    ws['A2'].font = Font(italic=True, name="Arial", size=10, color="1F4E79")

    hdrs = ['Model', 'k', 'n valid', 'Accuracy', 'Correct/n',
            "Cohen's κ", 'κ 95% CI', 'Macro F1',
            'Ischemic F1', 'Non-Isc F1', 'Mixed F1', 'No-Scar F1', 'Not-Spec F1',
            'vs Zero-Shot Δacc', 'n missing', 'max prompt tokens']
    for ci, h in enumerate(hdrs, 1):
        c = ws.cell(row=4, column=ci, value=h)
        c.fill = BLUE; c.font = WB
        c.alignment = Alignment(horizontal='center', wrap_text=True)

    # Reference row
    ref = ['Rule-Based NLP', '0', '64', '92.2%', '59/64', '0.887',
           '[0.79–0.98]', '—', '—', '—', '—', '—', '—', '—', '0', '—']
    for ci, v in enumerate(ref, 1):
        ws.cell(row=5, column=ci, value=v).font = Font(bold=True, name="Arial",
                                                        size=10, color="1F4E79")

    # Zero-shot baselines on the same 64 reports (num_ctx=8192 re-runs) for Δ computation
    zs_acc = {'llama8b': 0.672, 'llama70b': 0.750, 'medgemma': 0.455}

    for ri, (mkey, data) in enumerate(all_model_results.items(), 6):
        m = data['metrics']
        acc, kappa = m['accuracy'], m['kappa']
        klo, khi   = m['kappa_ci']
        n          = m['n_valid']
        ptoks      = [r['prompt_tokens'] for r in data['results'] if r.get('prompt_tokens')]
        rpt        = m['report']
        mf1        = rpt.get('macro avg', {}).get('f1-score', 0)
        delta      = acc - zs_acc.get(mkey, 0)
        label_map  = {'llama8b': 'Llama-3.1-8B',
                      'llama70b': 'Llama-3.1-70B',
                      'medgemma': 'MedGemma-4B-IT'}

        vals = [label_map.get(mkey, mkey), k, n,
                f"{acc:.3f} ({acc:.1%})", f"{m['n_correct']}/{n}",
                f"{kappa:.3f}", f"[{klo:.3f}–{khi:.3f}]",
                round(mf1, 3),
                round(rpt.get('ischemic',      {}).get('f1-score', 0), 3),
                round(rpt.get('non-ischemic',  {}).get('f1-score', 0), 3),
                round(rpt.get('mixed',         {}).get('f1-score', 0), 3),
                round(rpt.get('no-scar',       {}).get('f1-score', 0), 3),
                round(rpt.get('not-specified', {}).get('f1-score', 0), 3),
                f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}",
                m['n_missing'], max(ptoks) if ptoks else None]
        for ci, v in enumerate(vals, 1):
            ws.cell(row=ri, column=ci, value=v).font = Font(name="Arial", size=10)

    for ci in range(1, len(hdrs)+1):
        ws.column_dimensions[get_column_letter(ci)].width = 16
    ws.column_dimensions['A'].width = 20
    ws.row_dimensions[4].height = 30

    # ── Per-model sheets ──────────────────────────────────────────────────────
    label_map = {'llama8b': 'Llama8B', 'llama70b': 'Llama70B', 'medgemma': 'MedGemma'}
    for mkey, data in all_model_results.items():
        m, results = data['metrics'], data['results']
        sname = f"{label_map.get(mkey, mkey)}_k{k}"

        # Predictions
        ws_p = wb.create_sheet(f"{sname}_Preds")
        hdrs2 = ['idx', 'gold_label', 'predicted_label', 'correct',
                 'guard_triggered', 'parse_error', 'confidence', 'evidence',
                 'status', 'raw_label', 'prompt_tokens', 'output_tokens', 'done_reason']
        for ci, h in enumerate(hdrs2, 1):
            c = ws_p.cell(row=1, column=ci, value=h)
            c.fill = BLUE2; c.font = WB
        for ri, r in enumerate(results, 2):
            correct = r['gold_label'] == r['predicted_label']
            vals = [r['idx'], r['gold_label'], r['predicted_label'] or 'MISSING',
                    correct, r['guard_triggered'], r['parse_error'],
                    round(r['confidence'], 3), r['evidence'][:200],
                    r.get('status'), r.get('raw_label'), r.get('prompt_tokens'),
                    r.get('output_tokens'), r.get('done_reason')]
            for ci, v in enumerate(vals, 1):
                cell = ws_p.cell(row=ri, column=ci, value=v)
                if ci == 4:
                    cell.fill = GREEN if v else RED
        ws_p.column_dimensions['H'].width = 50

        # Confusion matrix
        ws_c = wb.create_sheet(f"{sname}_CM")
        ws_c.cell(row=1, column=1, value="True \\ Pred").font = BOLD
        for ci, cls in enumerate(CLASSES, 2):
            ws_c.cell(row=1, column=ci, value=cls).font = BOLD
            ws_c.cell(row=ci, column=1, value=cls).font = BOLD
        for ri, row_vals in enumerate(m['cm'], 2):
            for ci, v in enumerate(row_vals, 2):
                cell = ws_c.cell(row=ri, column=ci, value=int(v))
                cell.fill = GREEN if ri-1 == ci-1 else (RED if v > 0 else PatternFill())

    wb.save(out_path)
    print(f"\nSaved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.out is None:
        mstr = '_'.join(args.models)
        args.out = os.path.join(BASE_DIR,
            f"fewshot_all_k{args.k}_{mstr}_results.xlsx")

    # Load data
    df = load_expert_data(args.expert_xlsx, args.source_xlsx)
    train_df, test_df = make_split(df, args.split_xlsx)
    if args.n_test:
        test_df = test_df.head(args.n_test).reset_index(drop=True)
        print(f"Smoke test: {args.n_test} reports")

    # Build retriever
    retriever = FewShotRetriever(train_df, knn_pool=args.knn_pool)

    # Run each requested model
    all_results = {}
    for mkey in args.models:
        results = run_model(mkey, args.k, test_df, retriever, args)
        metrics = compute_metrics(results)
        all_results[mkey] = {'results': results, 'metrics': metrics}

        # Per-model print
        acc, kappa = metrics['accuracy'], metrics['kappa']
        klo, khi   = metrics['kappa_ci']
        n          = metrics['n_valid']
        print(f"\n  {mkey}  k={args.k}:")
        print(f"  Accuracy : {acc:.3f}  ({metrics['n_correct']}/{n})  missing={metrics['n_missing']}")
        print(f"  κ        : {kappa:.3f}  [{klo:.3f}–{khi:.3f}]")
        print(classification_report(
            metrics['y_true'], metrics['y_pred'], labels=CLASSES, zero_division=0))

    # Save
    save_excel(all_results, args.k, args, args.out)

    # Final comparison
    zs = {'llama8b': (0.672, 0.558), 'llama70b': (0.750, 0.665),
          'medgemma': (0.455, 0.307)}
    lmap = {'llama8b': 'Llama-3.1-8B', 'llama70b': 'Llama-3.1-70B',
            'medgemma': 'MedGemma-4B-IT'}
    print(f"\n{'='*65}")
    print(f"SUMMARY  |  k={args.k}  |  n_test={len(test_df)}")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'Acc':>7} {'κ':>7}  {'Δacc vs 0-shot':>15}")
    print(f"  {'─'*52}")
    print(f"  {'NLP zero-shot (ref)':<22} {'92.2%':>7} {'0.887':>7}")
    for mkey, data in all_results.items():
        m  = data['metrics']
        zs_a, _ = zs.get(mkey, (0, 0))
        delta   = m['accuracy'] - zs_a
        print(f"  {lmap.get(mkey, mkey):<22} "
              f"{m['accuracy']:>7.1%} "
              f"{m['kappa']:>7.3f}  "
              f"  {'+' if delta>=0 else ''}{delta:.1%}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

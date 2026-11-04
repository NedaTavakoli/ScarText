#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cardiac_mri_nlp_5class.py
==========================
Row-level classification of cardiac MRI reports into 5 LGE-characterised
scar phenotype classes:

    ischemic | non-ischemic | mixed | no-scar | not-specified

Reports involving cardiac devices (pacemaker / ICD / CRT) or performed
without gadolinium contrast are EXCLUDED before classification.

Output Excel (rowlevel_report_classification_5class.xlsx):
    Sheet 1  Row_Level          — one row per CMR report
    Sheet 2  ScarType_Summary   — class counts + percentages
    Sheet 3  MRN_RowCounts      — per-patient report counts
    Sheet 4  Patient_Level      — one row per patient (highest-priority label)
    Sheet 5  PatientType_Summary— patient-level class distribution
    Sheet 6  Run_Summary        — cohort overview
"""

import os, re, sys, argparse, warnings
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StringType, FloatType, BooleanType,
    StructType, StructField, LongType, TimestampType,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.environ.get("CMR_DATA_DIR", "data")   # folder with the report and annotation files (not distributed)
EXCEL_PATH = os.path.join(BASE_DIR, "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx")
OUT_XLSX         = os.path.join(BASE_DIR, "rowlevel_report_classification_5class.xlsx")
EXPERT_OUT_XLSX  = os.path.join(BASE_DIR, "expert_annotation_results.xlsx")

# HTML gallery paths
INDEX_HTML         = os.path.join(BASE_DIR, "index_ner_gallery_5class.html")
GALLERY_ISC_HTML   = os.path.join(BASE_DIR, "gallery_ischemic.html")
GALLERY_NONISC_HTML= os.path.join(BASE_DIR, "gallery_non_ischemic.html")
GALLERY_MIXED_HTML = os.path.join(BASE_DIR, "gallery_mixed.html")
GALLERY_NOSCAR_HTML= os.path.join(BASE_DIR, "gallery_no_scar.html")
GALLERY_NOTSPEC_HTML=os.path.join(BASE_DIR, "gallery_not_specified.html")

APP_NAME = "CardiacMRI_RowLevel_5Class"

# 5 classes
CLASSES  = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]

# Patient-level priority (highest first)
PRIORITY = {c: i for i, c in enumerate(
    ["mixed", "ischemic", "non-ischemic", "no-scar", "not-specified"]
)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="5-class cardiac MRI NLP + HTML gallery")
    # PHI redaction
    p.add_argument("--redact_mrn",            action="store_true")
    p.add_argument("--redact_accession",      action="store_true")
    p.add_argument("--redact_procedure_date", action="store_true")
    p.add_argument("--redact_all",            action="store_true",
                   help="Shortcut: redact MRN + accession + date")
    # NLP gallery options
    p.add_argument("--gallery_n",             type=int, default=60,
                   help="Max reports per NLP gallery class (default 60)")
    p.add_argument("--require_lge_for_gallery", action="store_true",
                   help="Only show LGE-positive reports in ischemic/non-ischemic/mixed galleries")
    p.add_argument("--random_gallery",        action="store_true",
                   help="Sample gallery reports randomly instead of by keyword score")
    p.add_argument("--random_seed",           type=int, default=42,
                   help="Random seed for --random_gallery (default 42)")
    # Expert annotation galleries (optional)
    p.add_argument("--expert_galleries",      action="store_true",
                   help="Also generate expert annotation HTML galleries")
    p.add_argument("--two_expert_xlsx",       type=str,
                   default=os.path.join(BASE_DIR, "Two_Expert_Annotation_Comparision.xlsx"),
                   help="Path to Two_Expert_Annotation_Comparision.xlsx — contains report text, "
                        "Reader 1 labels, Reader 2 labels, accession numbers, and procedure dates. "
                        "This is the only file needed for expert galleries.")
    args = p.parse_args()
    if args.redact_all:
        args.redact_mrn = args.redact_accession = args.redact_procedure_date = True
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Spark
# ─────────────────────────────────────────────────────────────────────────────
def initialize_spark() -> SparkSession:
    print("Initializing Spark...")
    spark = (
        SparkSession.builder.appName(APP_NAME)
        .master("local[*]")
        .config("spark.driver.memory", "8G")
        .config("spark.driver.maxResultSize", "2G")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "1000M")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version}  |  Python {sys.version.split()[0]}\n")
    return spark


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).lower()


def extract_lvef(text) -> Optional[float]:
    t = str(text) if text else ""
    patterns = [
        r"\bLVEF[:\s=]*\(?\s*(\d{1,3})\s*%?\s*\)?\b",
        r"\bleft\s+ventric(?:le|ular)\s+ejection\s+fraction[:\s=]*\s*(\d{1,3})\s*%?\b",
        r"\bejection\s+fraction[:\s=]*\s*(\d{1,3})\s*%?\b",
        r"\bEF[:\s=]*\s*(\d{1,3})\s*%?\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if 10 <= v <= 100:
                    return v
            except Exception:
                pass
    return None


def _is_device(text) -> bool:
    t = _safe(text)
    keywords = [
        "pacemaker", "pacer", "implantable cardioverter",
        "crt-d", "crt-p", "crt device", "cardiac resynchronization",
        "biventricular pacemaker", "biventricular icd",
        "device lead", "pacemaker lead", "icd lead",
        "defibrillator lead", "pacing lead", "pulse generator",
    ]
    if any(kw in t for kw in keywords):
        return True
    if re.search(r"\bICD\b", t):
        return True
    if re.search(r"\bCRT[-\s]?[DP]\b", t):
        return True
    return False


def _is_no_contrast(text) -> bool:
    t = _safe(text)

    # Precise no-contrast statements — specific enough to not false-fire
    no_contrast_phrases = [
        "no gadolinium contrast was administered",
        "no gadolinium was administered",
        "gadolinium was not administered",
        "no contrast was administered",
        "without gadolinium contrast",       # specific — not "with and without contrast"
        "non-contrast study",
        "contrast not given",
        "no intravenous contrast",
        "no intravenous gadolinium",
        # NOTE: "without contrast" intentionally removed — it is far too broad;
        # it matches "CT chest without contrast" in comparison lines and
        # "MRI with and without contrast" in procedure names, wrongly excluding
        # 35,000+ legitimate LGE reports from the corpus.
    ]
    if any(p in t for p in no_contrast_phrases):
        return True

    # Noncardiac overread / incidental reports — these are not primary CMR reads
    if "noncardiac review" in t or re.search(r"\boverread\b", t):
        return True

    # LGE section headers — if ANY present, contrast was given and LGE was performed
    lge_headers = [
        "late gadolinium enhancement",
        "delayed enhancement",
        "delayed enhanced",          # covers "delayed enhanced imaging", "delayed-enhanced flash"
        "delayed-enhanced",
        "viability",
        "delayed imaging",
        "delayed contrast",
        "delayed contrast-enhanced",
        "gbca",                      # gadolinium-based contrast agent — explicit dosing line
        "gadolinium kinetics",
        "psir",                      # phase-sensitive inversion recovery (LGE sequence)
        "inversion recovery",
        "viability sequences",
        "viability imaging",
        "scar imaging",
        "lge imaging",
        "lge sequences",
    ]
    return not any(h in t for h in lge_headers)


# ─────────────────────────────────────────────────────────────────────────────
# Contextual scar detection
# ─────────────────────────────────────────────────────────────────────────────

# Organs/contexts that are NOT the myocardium — scar/fibrosis mentions here
# must NOT trigger has_scar=True for cardiac classification
_NON_CARDIAC_SCAR_CTX = re.compile(
    r'\b(?:'
    r'lung|pulmonary|pleural|pleur\w*|lobe|lobar|'
    r'right\s+upper|right\s+middle|right\s+lower|left\s+upper|left\s+lower|'
    r'apical\s+lung|right\s+apical|left\s+apical|lung\s+apex|'
    r'parenchymal|parenchyma|'
    r'surgical|post.surgical|post.op|post\s+operative|'
    r'breast|axillary|shoulder|renal|kidney|cortical|hepatic|liver|'
    r'pericardial|pericardium|'
    r'tuberculous|tuberculosis|'
    r'radiation|post.radiation'
    r')\b',
    re.I
)

# Sentence-level negations — extended window + "no X to suggest scar" construction
_NEGATED_SCAR_PATTERNS = re.compile(
    # "no/without/absent ... scar/fibrosis" (window up to 80 chars)
    r'\b(?:no|not|without|absent|unremarkable|normal)\b'
    r'.{0,80}'
    r'\b(?:scar\w*|fibrosis|scarring)\b'
    r'|'
    # "no [areas of/evidence of] ... to suggest scar/fibrosis" (long-range)
    r'\bno\b.{0,120}\bto\s+suggest\b.{0,50}\b(?:scar\w*|fibrosis|fibrous)\b'
    r'|'
    # "scar/fibrosis ... not identified/seen/detected"
    r'\b(?:scar\w*|fibrosis|scarring)\b'
    r'.{0,40}'
    r'\b(?:not\s+(?:identified|seen|detected|present|demonstrated|evident)|'
    r'is\s+(?:not|absent)|cannot\s+be\s+(?:assessed|evaluated))\b',
    re.I
)

# History/indication lines — scar here is a clinical question, not a finding
_HISTORY_INDICATION_CTX = re.compile(
    r'^(?:history|indication|clinical\s+indication|hx|reason\s+for\s+exam|'
    r'reason\s+for\s+study|evaluate\s+for|assess\s+for|rule\s+out|'
    r'clinical\s+question|procedure)[:\s]',
    re.I | re.MULTILINE
)

# Technique section header — content after this should not be used for classification
_TECHNIQUE_SECTION_RE = re.compile(
    r'\b(?:technique|techniques|pulse\s+sequence|protocol|sequences\s+performed|'
    r'sequences\s+obtained|imaging\s+protocol)\b',
    re.I
)

def _strip_non_findings_sections(text: str) -> str:
    """
    Remove HISTORY, INDICATION, and TECHNIQUE sections from text before
    pattern matching. Handles both line-start headers and inline headers
    within the same line (e.g. "HISTORY: CAD\nTECHNIQUE: ...").

    Sections stripped:  HISTORY / INDICATION / CLINICAL INDICATION / HX /
                        REASON FOR EXAM / REASON FOR STUDY / TECHNIQUE /
                        PROTOCOL / PULSE SEQUENCE / SEQUENCES PERFORMED

    Sections that resume classification:
                        FINDINGS / IMPRESSION / CONCLUSION / LGE /
                        LATE GADOLINIUM / DELAYED ENHANCEMENT / VIABILITY /
                        LEFT VENTRICLE / RIGHT VENTRICLE / MYOCARDIUM /
                        ASSESSMENT / POST PROCESSING / COMPARISON /
                        LEFT VENTRICULAR / RIGHT VENTRICULAR
    """
    # Section header patterns — these STOP classification
    _SKIP_HDR = re.compile(
        r'(?:^|\n)\s*(?:history|indication|clinical\s+indication|hx|'
        r'reason\s+for\s+(?:exam|study|referral)|clinical\s+question|'
        r'technique|techniques|pulse\s+sequence|protocol|'
        r'sequences?\s+(?:performed|obtained)|imaging\s+protocol|'
        r'clinical\s+history|referring\s+physician|ordering\s+physician)\s*[:\-]',
        re.I
    )
    # Section header patterns — these RESUME classification
    _RESUME_HDR = re.compile(
        r'(?:^|\n)\s*(?:findings|impression|conclusion|results|'
        r'late\s+gadolinium|delayed\s+(?:contrast\s+)?enhancement|'
        r'viability|lge\s+(?:imaging|sequences|findings)|'
        r'left\s+ventricle|right\s+ventricle|left\s+ventricular\s+(?!ejection)|'
        r'right\s+ventricular|myocardium|assessment|'
        r'cardiac\s+structure|left\s+atrium|right\s+atrium|'
        r'post\s+processing|post-processing)\s*[:\-]?',
        re.I
    )

    # Split on any section boundary (skip or resume)
    # Strategy: find all header positions, determine skip/resume state,
    # and collect only the resume-state segments.
    all_hdrs = []
    for m in _SKIP_HDR.finditer(text):
        all_hdrs.append((m.start(), 'skip'))
    for m in _RESUME_HDR.finditer(text):
        all_hdrs.append((m.start(), 'resume'))
    all_hdrs.sort(key=lambda x: x[0])

    if not all_hdrs:
        return text  # No recognisable headers — return as-is

    segments = []
    # Text before the first header is included (pre-amble / procedure line)
    prev_pos = 0
    state = 'resume'  # default: include

    for pos, hdr_type in all_hdrs:
        if state == 'resume':
            segments.append(text[prev_pos:pos])
        state = hdr_type
        prev_pos = pos

    # Last segment
    if state == 'resume':
        segments.append(text[prev_pos:])

    result = ''.join(segments).strip()
    return result if result else text

def _has_myocardial_scar(t: str) -> bool:
    """
    Return True only if text contains an AFFIRMED myocardial scar/fibrosis
    finding in a cardiac context. Returns False for:
      - Non-cardiac contexts (lung, pleural, pericardial, surgical scarring)
      - Negated scar sentences ("no scarring", "no evidence of scar",
        "no areas of abnormal enhancement to suggest scar")
      - History/indication lines (clinical question, not a finding)

    NOTE: Enhancement pattern terms alone (mesocardial, mid-myocardial, etc.)
    are NOT sufficient — the radiologist must explicitly state scar or fibrosis.
    Enhancement is the detection method; the conclusion requires explicit language.
    """
    if not re.search(r'\b(?:scar\w*|fibrosis)\b', t, re.I):
        return False

    for sent in re.split(r'(?<=[.!?])\s+|\n', t):
        s = sent.strip()
        if not s or not re.search(r'\b(?:scar\w*|fibrosis)\b', s, re.I):
            continue
        if _NON_CARDIAC_SCAR_CTX.search(s):
            continue
        if _HISTORY_INDICATION_CTX.match(s):
            continue
        if _NEGATED_SCAR_PATTERNS.search(s):
            continue
        return True
    return False


# Expand no-LGE exact phrases with all variants found in the corpus
_NO_LGE_EXACT = [
    # Standard LGE negative
    "no late gadolinium enhancement",
    "no delayed enhancement",
    "no delayed myocardial enhancement",
    "no abnormal delayed myocardial enhancement",
    "no abnormal delayed myocardial contrast enhancement",
    "no delayed contrast enhancement",
    "no abnormal lge",
    "no lge",
    "no lge identified", "no lge detected", "no lge seen",
    "no lge is identified", "no lge is seen",
    "lge negative",
    "absent delayed enhancement",
    "absent late gadolinium enhancement",
    # No enhancement to suggest scar — KEY corpus variants
    "no abnormal enhancement to suggest myocardial scar",
    "no enhancement to suggest myocardial scar",
    "no abnormal enhancement is present",
    "no enhancing myocardial scar",
    "no hyperenhancing myocardial scar",
    "no hyperenhancement to suggest myocardial scar",
    "no hyperenhancement to indicate myocardial scar",
    "no delayed hyperenhancement is seen to suggest myocardial",
    "no delayed hyperenhancement to suggest",
    "no areas of abnormal delayed myocardial contrast enhancement",
    "no areas of hyperenhancement to suggest myocardial",
    "no areas of abnormal delayed myocardial",
    "no intracardiac mass or thrombus.  viability sequences show no evidence",
    # Viability-section specific negations
    "no abnormal enhancement",                          # in viability context
    "no macroscopic scar",
    "no macroscopic myocardial scar",
    "no macroscopic or microscopic myocardial scar",
    "no macroscopic scar by delayed enhanced imaging",
    "no macroscopic scar by delayed enhancement",
    "no interval myocardial scar",
    "no interval scar",
    "no myocardial macroscopic scar",
    "no scarring on the viability sequences",
    "no evidence of any scarring on the delayed viability",
    "no evidence of any myocardial scarring",
    "no evidence of myocardial scarring",
    "no evidence of any scar on the delayed viability",
    "no evidence of any myocardial enhancement",
    "no evidence for myocardial scar",
    "no evidence for myocardial inflammation or scar",
    "no evidence of delayed enhancement to suggest scar",
    "no evidence for delayed enhancement to suggest scar",
    "no evidence of abnormal enhancement on the delayed viability",
    "no chamber filling defects or delayed enhancement",
    "no chamber filling defects or myocardial delayed enhancement",
    # Scar/fibrosis negations
    "no myocardial scar",
    "no scar identified", "no scar is identified",
    "no evidence of scar",
    "no evidence of myocardial scar",
    "no evidence of myocardial fibrosis",
    "no myocardial fibrosis",
    "no intracardiac thrombus or scarring",
    "viability sequences show no evidence for myocardial scar",
    # Without constructs
    "without late gadolinium enhancement",
    "without delayed enhancement",
    "without myocardial scar",
    # "no gross delayed enhancement" = no scar
    "no gross delayed enhancement",
    "no gross myocardial delayed enhancement",
    "no gross late gadolinium enhancement",
    # "no evidence of myocarditis" / "no evidence of subendocardial" = no scar
    "no evidence of myocarditis",
    "no evidence of subendocardial",
    "no evidence of subendocardial enhancement",
    "no evidence of subendocardial delayed enhancement",
    "no subendocardial enhancement",
    "no subendocardial delayed enhancement",
    "no subendocardial lge",
    # "unable to comment on the presence of infarction" → not-specified (handled separately)
]

_MOD = r"(?:any\s+|focal\s+|significant\s+|appreciable\s+|definite\s+|abnormal\s+|myocardial\s+|pathologic(?:al)?\s+|obvious\s+|overt\s+)*"

_NO_LGE_REGEX = [
    # "no evidence of [mod] delayed/late enhancement"
    re.compile(r"no\s+evidence\s+of\s+" + _MOD + r"(?:delayed\s+enhancement|late\s+gadolinium\s+enhancement|lge)", re.I),
    re.compile(r"without\s+evidence\s+of\s+" + _MOD + r"(?:delayed\s+enhancement|late\s+gadolinium\s+enhancement|lge)", re.I),
    re.compile(r"without\s+" + _MOD + r"(?:delayed\s+enhancement|late\s+gadolinium\s+enhancement)", re.I),
    re.compile(r"no\s+" + _MOD + r"(?:delayed\s+enhancement|late\s+gadolinium\s+enhancement)", re.I),
    # "no [obvious/abnormal] delayed enhancement in the [LV/myocardium] to suggest [scar/fibrosis]"
    re.compile(r"no\s+(?:obvious\s+|abnormal\s+)?(?:evidence\s+of\s+)?(?:abnormal\s+)?delayed\s+enhancement\s+.{0,40}to\s+suggest\s+(?:fibrous|scar|fibrosis)", re.I),
    # "no hyperenhancement to suggest/indicate myocardial scar/scarring"
    re.compile(r"no\s+(?:delayed\s+)?hyperenhancement\s+(?:is\s+seen\s+)?to\s+(?:suggest|indicate)\s+myocardial\s+\w+", re.I),
    # "no enhancement to suggest myocardial scar"
    re.compile(r"no\s+(?:abnormal\s+)?enhancement\s+(?:is\s+)?(?:evident\s+|present\s+)?to\s+suggest\s+(?:interval\s+)?myocardial\s+scar", re.I),
    # "no areas of abnormal delayed myocardial contrast enhancement to suggest [scar/MI]"
    re.compile(r"no\s+areas\s+of\s+abnormal\s+delayed\s+myocardial\s+contrast\s+enhancement", re.I),
    # "there is no evidence of any delayed enhancement on the viability sequences"
    re.compile(r"no\s+evidence\s+of\s+" + _MOD + r"delayed\s+enhancement", re.I),
    # "no scarring/fibrosis on the viability/delayed sequences"  
    re.compile(r"no\s+(?:evidence\s+of\s+)?(?:any\s+)?(?:myocardial\s+)?scarring\s+(?:on|in|at|present|identified)", re.I),
    # "no gross [myocardial] delayed/late enhancement"
    re.compile(r"no\s+gross\s+(?:myocardial\s+)?(?:delayed|late)\s+(?:gadolinium\s+)?enhancement", re.I),
    # "no evidence of myocarditis" — no scar finding
    re.compile(r"no\s+evidence\s+of\s+myocarditis", re.I),
    # "no evidence of subendocardial [enhancement/delayed enhancement]"
    re.compile(r"no\s+(?:evidence\s+of\s+)?sub[\s\-]?endocardial\s+(?:delayed\s+|late\s+)?(?:enhancement|lge|gadolinium)", re.I),
]


def _is_no_lge(t: str) -> bool:
    """Return True if the lowercased text expresses absence of LGE / scar."""
    if any(p in t for p in _NO_LGE_EXACT):
        return True
    return any(rx.search(t) for rx in _NO_LGE_REGEX)


def _extract_patterns(text) -> Dict[str, Any]:
    t = _safe(text)
    # Strip indication/history/technique sections before pattern matching
    t_findings = _safe(_strip_non_findings_sections(text))

    ischemic_kw = [
        # endocardial scar = ischemic (subendocardial distribution)
        # Use word-boundary regex to avoid matching inside "subendocardial"
        r"\bendocardial\s+scar\b",
        r"\bendocardial\s+delayed\s+enhancement\b",
        r"\bendocardial\s+late\s+gadolinium\b",
        r"\bendocardial\s+lge\b",
        # myocardial infarction / infarct — BUT NOT pulmonary infarction
        "myocardial infarction", "myocardial infarct",
        # coronary distribution (specific) — ischemic
        "coronary distribution",
        r"\blad\s+territory\b", r"\brca\s+territory\b", r"\blcx\s+territory\b",
        r"\bstemi\b", r"\bnstemi\b",
        # embolic MYOCARDIAL infarction only
        "embolic infarct", "embolic myocardial",
        "thromboembolic",
    ]
    nonischemic_kw = [
        "mid-wall", "midwall", "mid wall",
        # mid-myocardial variants (all spellings)
        "mid-myocardial", "midmyocardial", "mid myocardial",
        # mid-wall / midwall (already above)
        # mesocardial / mesocardium variants
        "mesocardial", "mesocardium", "mesocardial scar",
        "hazy mesocardial", "hazy nonvascular mesocardial",
        "atypical mesocardial",
        # subepicardial — non-ischemic
        "subepicardial", "sub-epicardial",
        # epicardial — also match partial prefix "epicard"
        "epicardial", "epicardium", "epicard",
        # patchy (non-ischemic)
        "patchy",
        # linear area of late/delayed enhancement = non-ischemic fibrosis stripe
        "linear area of late enhancement",
        "linear area of delayed enhancement",
        "linear area of late gadolinium",
        "linear area of lge",
        "linear late enhancement",
        "linear delayed enhancement",
        "linear lge",
        "linear enhancement",          # broad catch — linear is the key
        r"linear\s+\w*\s*(?:delayed|late|gadolinium)\s+enhancement",
        # RV insertion / insertion point
        r"\brv\s+insertion\b", "insertion point",
        # disease labels
        "myocarditis", "sarcoidosis", r"\bsarcoid\b",
        "amyloidosis", r"\bamyloid\b", r"\bfabry\b",
        "hypertrophic cardiomyopathy", r"\bhcm\b",
        "dilated cardiomyopathy", r"\bdcm\b",
        "infiltrative", "inflammatory",
    ]

    # Special-case regex patterns for non-ischemic — matched on t_findings
    _NONI_SPECIAL_RE = [
        # "stable focus of delayed enhancement at the basal inferoseptum suggestive of RV strain" → non-ischemic
        re.compile(
            r'(?:stable\s+focus|focus)\s+of\s+(?:delayed|late)\s+enhancement'
            r'.{0,80}(?:rv\s+strain|right\s+ventricular\s+strain)',
            re.I
        ),
        # "focal stripe of myocardial delayed enhancement along the basal septum likely a sequela of prior valve disease"
        re.compile(
            r'(?:focal\s+)?stripe\s+of\s+(?:myocardial\s+)?(?:delayed|late)\s+enhancement'
            r'.{0,120}(?:valve\s+diseas|valvular|septal)',
            re.I
        ),
        # "redemonstration of focal delayed enhancement...linear enhancement at the mid infero-lateral wall"
        re.compile(
            r'(?:redemonstration|re-?demonstration)\s+of\s+(?:focal\s+)?(?:delayed|late)\s+enhancement'
            r'.{0,200}linear\s+enhancement',
            re.I
        ),
        # Removed: too broad, fired on ischemic mid-inferior infarcts

        # Sentence-aware nonischemic/nonvascular/atypical enhancement patterns.
        # These require a POSITIVE (non-negated) sentence context.
        # Implemented as a single callable that scans sentences.
        # (Placeholder regex — actual logic handled in _match_nonischemic_extra below)
        re.compile(r'PLACEHOLDER_NEVER_MATCHES_XYZ123', re.I),
    ]

    # Special-case regex patterns for ischemic — matched on t_findings
    # plain "infarct" / "infarction" but NOT "pulmonary infarction"
    _ISCH_INFARCT_RE = re.compile(
        r'(?<!\bpulmonary\s)(?<!\blung\s)(?<!\bpleural\s)\b(?:infarct|infarction)\b',
        re.I
    )
    # "unable to comment on the presence of infarction" → not-specified (not ischemic)
    _UNABLE_INFARCT_RE = re.compile(
        r'unable\s+to\s+comment\s+on\s+(?:the\s+)?presence\s+of\s+(?:infarction|infarct)',
        re.I
    )

    # Non-vascular: do NOT use this term alone to catch non-ischemic
    # (it is a qualifier, not a standalone classifier — remove from plain keyword matching
    #  but keep it as a qualifier when combined with other non-ischemic terms)

    # ── Negation guards for ischemic triggers ─────────────────────────────────
    # "no evidence of subendocardial infarction / enhancement"
    _NO_SUBENDO_RE = re.compile(
        r'no\s+evidence\s+of\s+sub[-\s]?endocardial\s+(?:infarction|enhancement|lge)',
        re.I
    )
    # non-ischemic qualifiers that override subendocardial:
    # "nonvascular pattern of subendocardial", "subendocardial consistent with amyloidosis/myocarditis",
    # "sequelae of remote myocarditis", "mesocardial to subendocardial", "subendocardial to mid myocardial"
    _SUBENDO_NONI_OVERRIDE_RE = re.compile(
        r'(?:'
        # nonvascular pattern of subendocardial
        r'non[-\s]?vascular\s+(?:pattern\s+)?(?:of\s+)?(?:the\s+)?sub[-\s]?endocardial'
        r'|non[-\s]?vascular.{0,60}sub[-\s]?endocardial'
        # subendocardial consistent with known non-ischemic disease
        r'|sub[-\s]?endocardial.{0,300}(?:amyloid|myocarditis|sarcoid|infiltrative|'
        r'cardiomyopathy\s+(?:pattern|consistent|most))'
        r'|(?:amyloid|myocarditis|sarcoid|infiltrative).{0,300}sub[-\s]?endocardial'
        # mesocardial/mid-myocardial to subendocardial — non-ischemic distribution
        r'|mesocardial\s+to\s+sub[-\s]?endocardial'
        r'|sub[-\s]?endocardial\s+to\s+mid[-\s]?myocardial'
        # sequelae of myocarditis
        r'|sequelae\s+of\s+(?:remote\s+)?myocarditis'
        r'|sub[-\s]?endocardial.{0,150}sequelae'
        # without perfusion defect / non-ischemic explicitly stated
        r'|sub[-\s]?endocardial.{0,100}without\s+perfusion\s+defect'
        r'|sub[-\s]?endocardial.{0,60}non[-\s]?ischemic'
        # perfusion artifact context
        r'|(?:perfusion|processing)\s+artifact.{0,80}sub[-\s]?endocardial'
        r'|sub[-\s]?endocardial.{0,120}(?:artifact|processing\s+artifact)'
        r'|matched.{0,80}sub[-\s]?endocardial\s+defect.{0,80}artifact'
        # subendocardial STRESS PERFUSION defect — not an LGE scar finding
        r'|sub[-\s]?endocardial\s+(?:inducible\s+)?perfusion\s+defect'
        r'|sub[-\s]?endocardial.{0,60}(?:stress|at\s+stress|inducible)'
        # microvascular disease context
        r'|microvascular.{0,200}sub[-\s]?endocardial'
        r'|sub[-\s]?endocardial.{0,200}microvascular'
        # infiltrative pattern
        r'|infiltrative.{0,120}sub[-\s]?endocardial'
        r'|sub[-\s]?endocardial.{0,120}infiltrative'
        # subendocardial co-occurring with subepicardial → non-ischemic pattern
        r'|sub[-\s]?endocardial.{0,200}sub[-\s]?epicardial'
        r'|sub[-\s]?epicardial.{0,200}sub[-\s]?endocardial'
        r')',
        re.I | re.S
    )
    # "rather than infarction", "infarction is not" etc — differential/negated
    _INFARCT_NEGATED_RE = re.compile(
        r'(?:rather\s+than|not\s+(?:consistent\s+with|suggestive\s+of)|'
        r'no\s+evidence\s+of|rules?\s+out|unlikely\s+(?:to\s+represent\s+)?|'
        r'cannot\s+exclude|consider(?:ed)?\s+(?:less\s+likely|unlikely))\s+'
        r'.{0,40}\binfarct(?:ion)?\b'
        r'|\binfarct(?:ion)?\b.{0,40}'
        r'(?:not\s+(?:identified|seen|present|demonstrated)|is\s+(?:not|absent|excluded))',
        re.I
    )
    # CAD without LGE confirmation: only ischemic if paired with explicit scar/infarct/enhancement
    _CAD_WITH_LGE_RE = re.compile(
        r'(?:coronary\s+artery\s+disease|\bcad\b)\b.{0,300}'
        r'(?:delayed\s+enhancement|late\s+gadolinium|\blge\b|infarct|scar)',
        re.I | re.S
    )
    # ischemic cardiomyopathy: only ischemic if in findings (not just history) AND not ruled out
    _ISCH_CM_RULED_OUT_RE = re.compile(
        r'(?:non[-\s]?ischemic\s+cardiomyopathy|not\s+ischemic|'
        r'(?:rather\s+than|no\s+evidence\s+of)\s+ischemic)',
        re.I
    )

    def _match_ischemic():
        for kw in ischemic_kw:
            if kw.startswith(r"\b"):
                if re.search(kw, t_findings): return True
            else:
                if kw in t_findings: return True

        # subendocardial — with extensive context guards
        if re.search(r'\bsub[-]?endocardial\b', t_findings):
            if not _NO_SUBENDO_RE.search(t_findings):          # not negated
                if not _SUBENDO_NONI_OVERRIDE_RE.search(t_findings):  # not non-ischemic pattern
                    return True

        # CAD/coronary artery disease — only ischemic when paired with LGE/scar findings
        if re.search(r'\b(?:coronary\s+artery\s+disease|cad)\b', t_findings):
            if _CAD_WITH_LGE_RE.search(t_findings):
                return True

        # ischemic cardiomyopathy in findings — only if not ruled out
        if 'ischemic cardiomyopathy' in t_findings:
            if not _ISCH_CM_RULED_OUT_RE.search(t_findings):
                return True

        # emboli/embolism — only cardiac/myocardial, not pulmonary/DVT
        _CARDIAC_EMBOL_RE = re.compile(
            r'(?:cardiac|myocardial|coronary|arterial)\s+embol\w+'
            r'|embol\w+\s+(?:myocardial|infarct|event)',
            re.I
        )
        if _CARDIAC_EMBOL_RE.search(t_findings):
            return True

        # plain infarct/infarction (not pulmonary, not negated, not differential)
        if _ISCH_INFARCT_RE.search(t_findings):
            if not _UNABLE_INFARCT_RE.search(t_findings):
                if not _INFARCT_NEGATED_RE.search(t_findings):
                    # Exclude purely pulmonary infarct context
                    # Exclude purely pulmonary infarct context
                    _PULM_INFARCT_ONLY_RE = re.compile(
                        r'pulmonary\s+infarct|lung\s+infarct|lobar\s+infarct'
                        r'|right\s+(?:lower|upper|middle)\s+lobe\s+.{0,40}infarct'
                        r'|infarct.{0,40}(?:lung|lobe|pulmonary)'
                        r'|consolidation.{0,80}infarct',
                        re.I
                    )
                    # Cardiac infarct: specific wall/territory mention
                    _CARDIAC_INFARCT_RE = re.compile(
                        r'myocardial\s+infarct'
                        r'|infarct.{0,60}(?:wall|septum|apex|inferior|anterior|lateral|basal|lv|left\s+ventricl)'
                        r'|(?:inferior|anterior|lateral|septal|apical|basal).{0,60}infarct'
                        r'|\blad\b|\brca\b|\blcx\b',
                        re.I
                    )
                    # Differential/vague infarct — not definitive
                    _INFARCT_DIFFERENTIAL_RE = re.compile(
                        r'(?:could\s+(?:also\s+)?be\s+considered|may\s+represent|'
                        r'cannot\s+exclude|possible\s+|unlikely)\s*.{0,60}\binfarct(?:ion)?\b'
                        r'|\binfarct(?:ion)?\b\s*\.\s*$',  # bare 'an infarct.' fragment
                        re.I | re.M
                    )
                    t_infarct_hits = list(re.finditer(r'\binfarct(?:ion)?\b', t_findings, re.I))
                    all_pulmonary = all(
                        _PULM_INFARCT_ONLY_RE.search(
                            t_findings[max(0, m.start()-80):m.end()+80]
                        )
                        for m in t_infarct_hits
                    )
                    if all_pulmonary:
                        pass  # purely pulmonary — not cardiac ischemic
                    elif _INFARCT_DIFFERENTIAL_RE.search(t_findings) and not _CARDIAC_INFARCT_RE.search(t_findings):
                        pass  # vague differential only, no specific cardiac territory
                    elif _CARDIAC_INFARCT_RE.search(t_findings):
                        return True
                    else:
                        return True
        return False

    # ── Non-ischemic false-trigger guards ────────────────────────────────────────
    # epicardial FAT or epicardial CORONARY — not the myocardial layer
    _EPICARD_FALSE_RE = re.compile(
        r'epicardial\s+(?:fat|adipose|coronary|vessel|artery|arteries|disease|'
        r'lead|electrode|pacing|defibrillator)',
        re.I
    )
    # inflammatory/infiltrative pericarditis — pericardial, not myocardial
    _PERICARD_INFLAM_RE = re.compile(
        r'(?:pericarditis|pericardial)\s+.{0,60}(?:inflammatory|infiltrative)'
        r'|(?:inflammatory|infiltrative)\s+.{0,60}pericarditis',
        re.I
    )
    # "rv insertion" in ischemic coronary territory context → NOT non-ischemic
    _RV_INSERT_ISCH_CTX_RE = re.compile(
        r'rv\s+insertion.{0,150}(?:lad|left\s+anterior\s+descending|embolic|coronary|territory|ischemic)'
        r'|(?:lad|embolic|coronary|territory).{0,150}rv\s+insertion',
        re.I | re.S
    )
    # "infiltrative process" as qualifier on ischemic pattern — not standalone infiltrative disease
    _INFILTRATIVE_SUPERIMPOSED_RE = re.compile(
        r'infiltrative\s+process'
        r'|superimposed\s+infiltrative'
        r'|infiltrative\s+(?:component|superimposed)',
        re.I
    )
    # myocarditis/sarcoid only in history/indication (already stripped but as extra guard)
    _MYOCARD_HISTORY_ONLY_RE = re.compile(
        r'(?:history|evaluate\s+for|evaluation\s+of|rule\s+out|concern\s+for)\s+.{0,60}myocarditis',
        re.I
    )

    def _match_nonischemic():
        for kw in nonischemic_kw:
            if kw in ("nonvascular", "non-vascular", "non vascular"):
                continue  # non-vascular alone does not trigger

            if kw.startswith(r"\b"):
                if not re.search(kw, t_findings):
                    continue
            else:
                if kw not in t_findings:
                    continue

            # Matched — now check for false-trigger guards

            # midwall: skip if it appears only as "subendocardial/midwall" or
            # "subendocardial to midwall" — ischemic-dominant pattern
            if kw in ("mid-wall", "midwall", "mid wall"):
                _SUBENDO_MIDWALL_RE = re.compile(
                    r'sub[-]?endocardial[/\s](?:to\s+)?mid[-]?wall'
                    r'|sub[-]?endocardial(?:\s+to\s+|/)mid[-\s]?myocardial',
                    re.I
                )
                # Only skip if ALL midwall mentions are in subendocardial/ context
                midwall_hits = list(re.finditer(r'\bmid[-\s]?wall\b', t_findings, re.I))
                all_slash = all(
                    _SUBENDO_MIDWALL_RE.search(t_findings[max(0,m.start()-30):m.end()+30])
                    for m in midwall_hits
                )
                if all_slash:
                    continue

            # epicardial: skip if it refers to fat or coronary arteries
            if kw in ("epicardial", "epicardium", "epicard"):
                if _EPICARD_FALSE_RE.search(t_findings):
                    # Only skip if ALL epicard mentions are fat/coronary context
                    genuine = re.search(
                        r'epicardial\s+(?:delayed\s+enhancement|lge|scar|fibrosis|enhancement)',
                        t_findings, re.I
                    )
                    if not genuine:
                        continue

            # inflammatory/infiltrative: skip if it is pericarditis context
            if kw in ("inflammatory", "infiltrative"):
                if _PERICARD_INFLAM_RE.search(t_findings):
                    continue
                if _INFILTRATIVE_SUPERIMPOSED_RE.search(t_findings) and kw == "infiltrative":
                    continue  # "infiltrative process" superimposed on ischemic — not standalone

            # myocarditis: skip if only in history/indication line
            if kw == "myocarditis":
                if _MYOCARD_HISTORY_ONLY_RE.search(t_findings):
                    # Check if there's also a findings-level mention
                    genuine = re.search(
                        r'(?:consistent\s+with|findings?\s+(?:of|suggest)|'
                        r'evidence\s+of|pattern\s+of)\s+myocarditis',
                        t_findings, re.I
                    )
                    if not genuine:
                        continue

            # rv insertion: skip if it is in coronary/LAD ischemic territory context
            if kw in (r"\brv\s+insertion\b", "insertion point"):
                if _RV_INSERT_ISCH_CTX_RE.search(t_findings):
                    continue

            return True

        # special-case regex patterns
        for rx in _NONI_SPECIAL_RE:
            if rx.search(t_findings): return True

        # Sentence-aware check for nonischemic/nonvascular/atypical enhancement
        # Only fires in sentences that are NOT globally negated
        _NONI_SENT_PATS = [
            re.compile(r'(?:atypical\s+)?non[-\s]?ischemic\s+(?:type\s+|pattern\s+of\s+)?enhancement', re.I),
            re.compile(r'non[-\s]?vascular\s+pattern\s+of\s+(?:delayed\s+|late\s+)?enhancement', re.I),
            re.compile(r'(?:delayed|late)\s+(?:myocardial\s+)?(?:gadolinium\s+)?enhancement\b.{0,120}\b(?:atypical|non[-\s]?vascular)\b', re.I | re.S),
            re.compile(r'\bpattern\s+of\s+enhancement\s+is\s+non[-\s]?ischemic', re.I),
        ]
        _SENT_NEG = re.compile(r'\b(?:no|not|without|absent|no\s+evidence\s+of|no\s+signs?\s+of|no\s+obvious)\b', re.I)
        for sent in re.split(r'(?<=[.!?])\s+|\n', t_findings):
            s = sent.strip()
            if not s: continue
            # Skip sentences that start with or contain negation near the match
            for pat in _NONI_SENT_PATS:
                m = pat.search(s)
                if m:
                    # Check: is this negated? Look for negation within 50 chars before match
                    before = s[:m.start()].strip()
                    if not _SENT_NEG.search(before[-60:] if len(before) > 60 else before):
                        return True
        return False

    has_isch = _match_ischemic()
    has_noni = _match_nonischemic()
    has_scar = _has_myocardial_scar(t)

    loc_kw = [
        "inferolateral", "anterolateral", "inferoseptal", "anteroseptal",
        "anterior", "inferior", "lateral", "septal",
        "apex", "apical", "basal", "mid", "base", "circumferential",
    ]
    locs = [x for x in loc_kw if x in t]
    loc_str = ", ".join(dict.fromkeys(locs))

    if has_isch and has_noni:
        desc = "mixed (ischemic + non-ischemic)"
    elif has_isch:
        if "subendocardial" in t_findings or "sub-endocardial" in t_findings:
                                                                       desc = "subendocardial (ischemic)"
        elif "endocardial" in t_findings:                              desc = "endocardial (ischemic)"
        elif re.search(r"\bembol\w+\b", t_findings):                   desc = "embolic infarct (ischemic)"
        elif re.search(r"\b(?:infarct|infarction)\b", t_findings):     desc = "infarction (ischemic)"
        else:                                                           desc = "ischemic pattern"
    elif has_noni:
        if "mid-myocardial" in t_findings or "midmyocardial" in t_findings or "mid myocardial" in t_findings:
                                                                       desc = "mid-myocardial (non-ischemic)"
        elif "mid-wall" in t_findings or "midwall" in t_findings:      desc = "mid-wall (non-ischemic)"
        elif "mesocardium" in t_findings or "mesocardial" in t_findings:
                                                                       desc = "mesocardial (non-ischemic)"
        elif re.search(r"linear\s+\w*\s*(?:delayed|late|gadolinium|lge)?\s*enhancement", t_findings):
                                                                       desc = "linear enhancement (non-ischemic)"
        elif "linear enhancement" in t_findings or "linear lge" in t_findings:
                                                                       desc = "linear enhancement (non-ischemic)"
        elif "subepicardial" in t_findings or "sub-epicardial" in t_findings:
                                                                       desc = "subepicardial (non-ischemic)"
        elif "hazy mesocardial" in t_findings:                         desc = "hazy mesocardial (non-ischemic)"
        elif "epicard" in t_findings:                                  desc = "epicardial (non-ischemic)"
        elif "patchy" in t_findings:                                   desc = "patchy (non-ischemic)"
        elif re.search(r"\brv\s+insertion\b", t_findings) or "insertion point" in t_findings:
                                                                       desc = "RV insertion (non-ischemic)"
        else:                                                          desc = "non-ischemic pattern"
    else:
        desc = ""

    return {
        "has_ischemic":    has_isch,
        "has_nonischemic": has_noni,
        "has_scar":        has_scar,
        "pattern":         desc,
        "location":        loc_str,
    }


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 5-class classifier
# ─────────────────────────────────────────────────────────────────────────────
# Regex to detect POSITIVE (non-negated) myocardial LGE in findings.
# Used to override _is_no_lge when a report describes both enhancement
# in one territory AND no-enhancement in another (local vs global statement).
# Must be tightly scoped to avoid false positives on negated phrases.
_EXPLICIT_LGE_PRESENT_RE = re.compile(
    # "subendocardial delayed enhancement", "transmural delayed enhancement",
    # "mid-myocardial/epicardial/mesocardial delayed enhancement"
    r'\b(?:sub[-]?endocardial|transmural|mid[-\s]?myocardial|epicardial|mesocardial|'
    r'mid[-\s]?wall|linear|patchy)\s+(?:delayed\s+|late\s+)?(?:myocardial\s+)?'
    r'(?:gadolinium\s+)?enhancement\b'
    r'|'
    # "there is [subendocardial/focal/area of] delayed/late enhancement [involving/at/in the wall]"
    r'\bthere\s+is\b.{0,40}'
    r'\b(?:delayed|late)\s+(?:myocardial\s+)?(?:gadolinium\s+)?enhancement\b'
    r'.{0,60}'
    r'\b(?:involving|at\s+the|in\s+the|within\s+the|throughout\s+the|of\s+the)\b'
    r'|'
    # "delayed/late enhancement in/at/involving the [wall segment]"
    r'\b(?:delayed|late)\s+(?:myocardial\s+)?(?:gadolinium\s+)?enhancement\b'
    r'.{0,60}'
    r'\b(?:wall|segment|septum|apex|inferior|anterior|lateral|basal|mid\b|inferolateral|'
    r'anterolateral|inferoseptal|anteroseptal|inferoseptum)\b',
    re.I | re.S
)

def classify_5class(text) -> Optional[str]:
    t = _safe(text)
    if not t: return None
    if _is_device(text): return None
    if _is_no_contrast(text): return None

    # "unable to comment on the presence of infarction" → not-specified
    if re.search(
        r'unable\s+to\s+comment\s+on\s+(?:the\s+)?presence\s+of\s+(?:infarction|infarct)',
        t, re.I
    ):
        return "not-specified"

    # _is_no_lge check — but only return no-scar if there is NO explicit LGE
    # described elsewhere in findings. Prevents "without delayed enhancement"
    # in one territory from masking enhancement described in another.
    if _is_no_lge(t):
        t_findings = _safe(_strip_non_findings_sections(text))
        if not _EXPLICIT_LGE_PRESENT_RE.search(t_findings):
            return "no-scar"
        # Explicit LGE is present elsewhere → fall through to pattern matching

    p = _extract_patterns(text)

    # If scar exists (has_scar=True) but pattern type is unknown, classify as not-specified
    # (scar visualized but cannot determine ischemic vs non-ischemic)
    if p["has_ischemic"] and p["has_nonischemic"]: return "mixed"
    if p["has_ischemic"]: return "ischemic"
    if p["has_nonischemic"]: return "non-ischemic"
    if p["has_scar"]: return "not-specified"  # scar present but type unclear
    return "no-scar"


def extract_lge_info(text) -> Dict[str, Any]:
    if _is_device(text) or _is_no_contrast(text):
        return {"has_lge": False, "has_scar": False, "location": "", "pattern": ""}
    no_lge = _is_no_lge(_safe(text))
    p = _extract_patterns(text)
    return {
        "has_lge":  not no_lge,
        "has_scar": p["has_scar"],
        "location": p["location"],
        "pattern":  p["pattern"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spark pipeline
# ─────────────────────────────────────────────────────────────────────────────
def load_excel_to_spark(spark: SparkSession, excel_path: str):
    print("Loading Excel with pandas...")
    df_pd = pd.read_excel(excel_path, header=8)
    required = ["MRN", "cardiac MRI text", "accession number", "procedure date"]
    missing = [c for c in required if c not in df_pd.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    
    df_pd = df_pd[required].dropna(subset=["cardiac MRI text"]).reset_index(drop=True)
    df_pd["row_id"]          = np.arange(len(df_pd), dtype=np.int64)
    df_pd["MRN"]             = df_pd["MRN"].astype(str)
    df_pd["accession number"]= df_pd["accession number"].astype(str)
    df_pd["cardiac MRI text"]= df_pd["cardiac MRI text"].astype(str)
    df_pd["procedure date"]  = pd.to_datetime(df_pd["procedure date"], errors="coerce")

    # Keep a copy of the full raw cohort for the Exclusion_Summary sheet
    df_raw = df_pd[["MRN", "cardiac MRI text"]].copy()

    print(f"Loaded {len(df_pd):,} rows\n")
    schema = StructType([
        StructField("MRN",              StringType(),    True),
        StructField("cardiac MRI text", StringType(),    True),
        StructField("accession number", StringType(),    True),
        StructField("procedure date",   TimestampType(), True),
        StructField("row_id",           LongType(),      False),
    ])
    # Repartitioning here helps with the "large task size" warning
    return spark.createDataFrame(df_pd, schema=schema).repartition(20), df_raw


def process_row_level(df_spark):
    print("Running classification in Spark (UDFs)...")

    @F.udf(FloatType())
    def _lvef_udf(t): return extract_lvef(t)

    @F.udf(StringType())
    def _cls_udf(t):  return classify_5class(t)

    df = (df_spark
          .withColumn("lvef",      _lvef_udf(F.col("cardiac MRI text")))
          .withColumn("scar_type", _cls_udf( F.col("cardiac MRI text"))))

    n_before = df.count()
    df = df.filter(F.col("scar_type").isNotNull())
    n_after  = df.count()
    print(f"  Excluded {n_before - n_after:,} device/no-contrast reports → {n_after:,} remain\n")

    df_pd = df.toPandas()
    print("Extracting LGE metadata in pandas...")
    lge = df_pd["cardiac MRI text"].apply(extract_lge_info)
    df_pd["has_lge"]      = lge.apply(lambda x: bool(x["has_lge"]))
    df_pd["has_scar"]     = lge.apply(lambda x: bool(x["has_scar"]))
    df_pd["lge_location"] = lge.apply(lambda x: x["location"])
    df_pd["lge_pattern"]  = lge.apply(lambda x: x["pattern"])
    print("Done.\n")
    return df_pd


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level aggregation
# ─────────────────────────────────────────────────────────────────────────────
def derive_patient_level(df_row: pd.DataFrame) -> pd.DataFrame:
    def best_label(types: list) -> str:
        present = [t for t in types if t in PRIORITY]
        if not present: return "not-specified"
        return min(present, key=lambda x: PRIORITY[x])

    # FIX: Changed "list" (string) to list (builtin function)
    g = df_row.groupby("MRN").agg(
        n_reports =("row_id",   "count"),
        any_lge   =("has_lge",  "max"),
        lvef_any  =("lvef",     lambda x: float(np.nanmean(x)) if pd.notna(x).any() else np.nan),
        pct_lge   =("has_lge",  "mean"),
        types     =("scar_type", list), 
    ).reset_index()
    
    g["patient_scar_type"] = g["types"].apply(best_label)
    g["pct_lge"]           = g["pct_lge"] * 100.0
    g["any_lge"]           = g["any_lge"].astype(bool)
    return g.drop(columns=["types"])


# ─────────────────────────────────────────────────────────────────────────────
# Exclusion summary sheet builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_exclusion_summary_sheet(ws, df_raw: pd.DataFrame, df_classified: pd.DataFrame):
    """
    Write a formatted Exclusion_Summary sheet onto an openpyxl worksheet.

    df_raw        — full cohort (all 41 k rows, before any exclusion)
    df_classified — rows that passed exclusions and were classified
    """
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    # ── compute counts ────────────────────────────────────────────────────────
    df_raw = df_raw.copy()
    df_raw["_dev"] = df_raw["cardiac MRI text"].apply(_is_device)
    df_raw["_nc"]  = df_raw["cardiac MRI text"].apply(_is_no_contrast)

    total              = len(df_raw)
    n_device           = int(df_raw["_dev"].sum())
    n_no_contrast      = int(df_raw["_nc"].sum())
    n_both             = int((df_raw["_dev"] & df_raw["_nc"]).sum())
    n_device_only      = int((df_raw["_dev"] & ~df_raw["_nc"]).sum())
    n_no_contrast_only = int((df_raw["_nc"]  & ~df_raw["_dev"]).sum())
    n_excluded         = int((df_raw["_dev"] | df_raw["_nc"]).sum())
    n_classifiable     = total - n_excluded

    mrn_total        = df_raw["MRN"].nunique()
    mrn_device       = df_raw[df_raw["_dev"]]["MRN"].nunique()
    mrn_no_contrast  = df_raw[df_raw["_nc"]]["MRN"].nunique()
    mrn_both         = df_raw[df_raw["_dev"] & df_raw["_nc"]]["MRN"].nunique()
    mrn_classifiable = df_raw[~(df_raw["_dev"] | df_raw["_nc"])]["MRN"].nunique()

    cls_counts = df_classified["scar_type"].value_counts()
    cls_mrn    = df_classified.groupby("scar_type")["MRN"].nunique()
    total_cls  = len(df_classified)
    total_cls_mrn = df_classified["MRN"].nunique()

    # ── style helpers ─────────────────────────────────────────────────────────
    def argb(h): return "FF" + h.upper()

    PURPLE_DARK  = argb("4A1D6E")
    PURPLE_MID   = argb("7B3FA0")
    PURPLE_LIGHT = argb("EDE0F7")
    TEAL         = argb("1A7A6E")
    TEAL_LIGHT   = argb("D6F0ED")
    RED_LIGHT    = argb("FCE8E8")
    AMBER_LIGHT  = argb("FFF3CD")
    GREEN_LIGHT  = argb("E8F5E9")
    PINK_LIGHT   = argb("FDE8F5")
    GREY_LIGHT   = argb("F5F5F5")
    WHITE        = argb("FFFFFF")

    def fill(c):
        return PatternFill("solid", fgColor=c)

    def bdr():
        s = Side(style="thin", color="CCCCCC")
        return Border(left=s, right=s, top=s, bottom=s)

    def ctr():
        return Alignment(horizontal="center", vertical="center", wrap_text=True)

    def lft():
        return Alignment(horizontal="left", vertical="center", wrap_text=True)

    def af(row, col, value, bg=WHITE, bold=False, sz=9, color="FF000000",
           num_fmt=None, align=None):
        c = ws.cell(row, col, value)
        c.font      = Font(name="Arial", size=sz, bold=bold, color=color)
        c.fill      = fill(bg)
        c.alignment = align or ctr()
        c.border    = bdr()
        if num_fmt:
            c.number_format = num_fmt
        return c

    def title_row(r, text, bg, sz=12):
        ws.merge_cells(f"A{r}:F{r}")
        af(r, 1, text, bg=bg, bold=True, sz=sz, color="FFFFFFFF", align=ctr())
        ws.row_dimensions[r].height = 30 if sz >= 12 else 22

    def sec_hdr(r, text, bg):
        ws.merge_cells(f"A{r}:F{r}")
        af(r, 1, text, bg=bg, bold=True, sz=10, color="FFFFFFFF", align=lft())
        ws.row_dimensions[r].height = 22

    def col_hdr(r, labels, bg):
        for i, lbl in enumerate(labels, 1):
            af(r, i, lbl, bg=bg, bold=True, sz=9, align=ctr())
        ws.row_dimensions[r].height = 34

    def data_row(r, label, n_rep, pct_rep_formula, n_pat, pct_pat_formula, note, rf):
        af(r, 1, label,  bg=rf, bold=True, sz=9, align=lft())
        af(r, 2, n_rep,  bg=rf, bold=True, sz=9)
        if pct_rep_formula:
            af(r, 3, pct_rep_formula, bg=rf, sz=9, num_fmt="0.0%")
        else:
            af(r, 3, "—", bg=rf, sz=9)
        if isinstance(n_pat, int):
            af(r, 4, n_pat, bg=rf, bold=True, sz=9)
        else:
            af(r, 4, "—", bg=rf, sz=9)
        if pct_pat_formula:
            af(r, 5, pct_pat_formula, bg=rf, sz=9, num_fmt="0.0%")
        else:
            af(r, 5, "—", bg=rf, sz=9)
        af(r, 6, note, bg=rf, sz=9, align=lft())
        ws.row_dimensions[r].height = 20

    ws.sheet_view.showGridLines = False

    # ── Title block ───────────────────────────────────────────────────────────
    title_row(1, "Cardiac MRI Report Cohort — Exclusion Summary", PURPLE_DARK, sz=13)
    title_row(2, "Northwestern University  ·  Radiology  ·  ScarNet / SCMR Aim 2", PURPLE_MID, sz=9)
    ws.row_dimensions[3].height = 6

    # ── Block 1: Report-level exclusion counts ────────────────────────────────
    sec_hdr(4, "   REPORT-LEVEL EXCLUSION COUNTS", TEAL)
    col_hdr(5, ["Category", "Reports (n)", "Reports (%)", "Unique Patients (n)",
                "Unique Patients (%)", "Notes"], TEAL_LIGHT)

    block1 = [
        (6,  "Total cohort",                    total,              None,          mrn_total,        None,          "All CMR reports in dataset",                       WHITE),
        (7,  "Device (any)",                     n_device,           "=B7/B$6",    mrn_device,       "=D7/D$6",     "ICD, pacemaker, AICD, CRT — any device lead",      RED_LIGHT),
        (8,  "  Device only (no overlap)",        n_device_only,      "=B8/B$6",    None,             None,          "Device, NOT also no-contrast",                     RED_LIGHT),
        (9,  "No contrast (any)",                 n_no_contrast,      "=B9/B$6",    mrn_no_contrast,  "=D9/D$6",     "No gadolinium administered",                       AMBER_LIGHT),
        (10, "  No contrast only",               n_no_contrast_only, "=B10/B$6",   None,             None,          "No contrast, NOT also device",                     AMBER_LIGHT),
        (11, "Both device + no contrast",         n_both,             "=B11/B$6",   mrn_both,         "=D11/D$6",    "Reports meeting both exclusion criteria",           AMBER_LIGHT),
        (12, "Total excluded",                    n_excluded,         "=B12/B$6",   None,             None,          "Union: device OR no-contrast (no double-counting)", RED_LIGHT),
        (13, "Classifiable (neither criterion)",  n_classifiable,     "=B13/B$6",   mrn_classifiable, "=D13/D$6",    "Proceeded to 5-class scar classification",          GREEN_LIGHT),
    ]
    for r, lbl, nr, pr, np_, pp, note, rf in block1:
        data_row(r, lbl, nr, pr, np_, pp, note, rf)

    ws.row_dimensions[14].height = 8

    # ── Block 2: 5-class classification breakdown ─────────────────────────────
    sec_hdr(15, "   5-CLASS CLASSIFICATION  (classifiable reports only)", PURPLE_DARK)
    col_hdr(16, ["Scar Phenotype", "Reports (n)", "Reports (%)", "Unique Patients (n)",
                 "Unique Patients (%)", "Description"], PURPLE_LIGHT)

    cls_order = ["no-scar", "ischemic", "non-ischemic", "mixed", "not-specified"]
    cls_fills = [GREEN_LIGHT, RED_LIGHT, AMBER_LIGHT, PINK_LIGHT, GREY_LIGHT]
    cls_desc  = {
        "no-scar":       "No myocardial LGE / scar on viability sequences",
        "ischemic":      "Subendocardial LGE in coronary artery distribution",
        "non-ischemic":  "Mid-wall, epicardial, insertion point, infiltrative pattern",
        "mixed":         "Both ischemic and non-ischemic patterns present",
        "not-specified": "Scar present; pattern cannot be rule-classified",
    }
    total_cls_row = 22  # rows 17-21 = 5 classes, row 22 = total

    for i, (cls, rf) in enumerate(zip(cls_order, cls_fills)):
        r   = 17 + i
        n   = int(cls_counts.get(cls, 0))
        np_ = int(cls_mrn.get(cls, 0))
        data_row(r, cls, n, f"=B{r}/B${total_cls_row}", np_,
                 f"=D{r}/D${total_cls_row}", cls_desc[cls], rf)

    # Total row
    data_row(total_cls_row, "TOTAL CLASSIFIABLE", total_cls, "=B22/B22",
             total_cls_mrn, "=D22/D22", "", TEAL_LIGHT)
    ws.cell(total_cls_row, 1).font = Font(name="Arial", size=9, bold=True, color="FF000000")
    ws.row_dimensions[23].height = 8

    # ── Block 3: key metrics panel ────────────────────────────────────────────
    sec_hdr(24, "   KEY METRICS", PURPLE_MID)
    metrics = [
        ("Total CMR reports in dataset",            f"{total:,}"),
        ("Total unique patients (MRN)",             f"{mrn_total:,}"),
        ("Device exclusion — reports",              f"{n_device:,}  ({n_device/total*100:.1f}%)"),
        ("Device exclusion — unique patients",      f"{mrn_device:,}  ({mrn_device/mrn_total*100:.1f}%)"),
        ("No-contrast exclusion — reports",         f"{n_no_contrast:,}  ({n_no_contrast/total*100:.1f}%)"),
        ("No-contrast exclusion — unique patients", f"{mrn_no_contrast:,}  ({mrn_no_contrast/mrn_total*100:.1f}%)"),
        ("Overlap (device AND no-contrast)",        f"{n_both:,} reports  /  {mrn_both:,} patients"),
        ("Total excluded — reports",                f"{n_excluded:,}  ({n_excluded/total*100:.1f}%)"),
        ("Classifiable — reports",                  f"{n_classifiable:,}  ({n_classifiable/total*100:.1f}%)"),
        ("Classifiable — unique patients",          f"{mrn_classifiable:,}  ({mrn_classifiable/mrn_total*100:.1f}%)"),
    ]
    for i, (lbl, val) in enumerate(metrics):
        r = 25 + i
        af(r, 1, lbl, bg=GREY_LIGHT, bold=True, sz=9, align=lft())
        ws.merge_cells(f"B{r}:F{r}")
        af(r, 2, val, bg=WHITE, sz=9, align=lft())
        ws.row_dimensions[r].height = 20

    # ── column widths ─────────────────────────────────────────────────────────
    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 15
    ws.column_dimensions["C"].width = 13
    ws.column_dimensions["D"].width = 22
    ws.column_dimensions["E"].width = 22
    ws.column_dimensions["F"].width = 50


# ─────────────────────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────────────────────
def export_excel(df: pd.DataFrame, out_path: str, df_raw: pd.DataFrame = None):
    print("Exporting Excel...")
    # "cardiac MRI text" is excluded from the Excel output — it inflates the
    # file enormously (avg 4 KB/report × 38k rows ≈ 150 MB uncompressed).
    # Full text is available in the source xlsx and rendered in the HTML galleries.
    cols = ["row_id", "MRN", "procedure date", "accession number",
            "scar_type", "lvef", "has_lge", "has_scar",
            "lge_pattern", "lge_location"]
    df_row = df[[c for c in cols if c in df.columns]].copy()

    row_sum = (df_row["scar_type"].value_counts()
               .rename_axis("scar_type").reset_index(name="count"))
    row_sum["percent"] = row_sum["count"] / len(df_row) * 100.0

    mrn_counts = (df_row.groupby("MRN")
                  .agg(n_rows=("row_id","count"),
                       n_lge =("has_lge","sum"),
                       pct_lge=("has_lge","mean"))
                  .reset_index())
    mrn_counts["pct_lge"] *= 100.0

    pat_df  = derive_patient_level(df_row)
    pat_sum = (pat_df["patient_scar_type"].value_counts()
               .rename_axis("patient_scar_type").reset_index(name="count"))
    pat_sum["percent"] = pat_sum["count"] / len(pat_df) * 100.0

    run_sum = pd.DataFrame([{
        "n_reports_rows":        len(df_row),
        "n_unique_patients_mrn": df_row["MRN"].nunique(),
        "n_rows_with_lge":       int(df_row["has_lge"].sum()),
        "pct_rows_with_lge":     round(df_row["has_lge"].mean()*100, 2),
        "scope":                 "5-class (excluded-device + no-contrast removed)",
    }])

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_row    .to_excel(w, sheet_name="Row_Level",          index=False)
        row_sum   .to_excel(w, sheet_name="ScarType_Summary",   index=False)
        mrn_counts.to_excel(w, sheet_name="MRN_RowCounts",      index=False)
        pat_df    .to_excel(w, sheet_name="Patient_Level",      index=False)
        pat_sum   .to_excel(w, sheet_name="PatientType_Summary",index=False)
        run_sum   .to_excel(w, sheet_name="Run_Summary",        index=False)

    # Add Exclusion_Summary as a formatted sheet (requires openpyxl direct access)
    if df_raw is not None:
        from openpyxl import load_workbook
        wb = load_workbook(out_path)
        ws = wb.create_sheet("Exclusion_Summary")
        _build_exclusion_summary_sheet(ws, df_raw, df_row)
        # Move Exclusion_Summary to first position
        wb.move_sheet("Exclusion_Summary", offset=-(wb.sheetnames.index("Exclusion_Summary")))
        wb.save(out_path)
        print("  Exclusion_Summary sheet added.\n")

    print(f"Saved: {out_path}\n")
    print("=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"5-class reports : {len(df_row):,}")
    print(f"Unique patients : {df_row['MRN'].nunique():,}")
    print(f"LGE-positive    : {int(df_row['has_lge'].sum()):,} ({df_row['has_lge'].mean()*100:.1f}%)")
    print("\nReport distribution:")
    print(df_row["scar_type"].value_counts().to_string())
    print("\nPatient distribution:")
    print(pat_df["patient_scar_type"].value_counts().to_string())
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# HTML gallery logic (unchanged from original snippets)
# ─────────────────────────────────────────────────────────────────────────────
def _escape(s):
    s = "" if s is None else str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#039;"))

# Alias for clarity
_escape_html = _escape


def _pill_plain(k: str, v: str) -> str:
    return f'<span class="pill"><b>{_escape(k)}:</b> {_escape(v)}</span>'


def _pill_redactable(k: str, raw_value: str, redact: bool, toggle_fn: str) -> str:
    raw_value = "" if raw_value is None else str(raw_value)
    if not redact:
        return _pill_plain(k, raw_value)
    return (f'<span class="pill"><b>{_escape(k)}:</b>'
            f'<span class="redacted" data-raw="{_escape(raw_value)}">REDACTED</span>'
            f'<button class="toggle-btn" type="button" onclick="{toggle_fn}(this)">Show</button></span>')


def _meta_pills(row: pd.Series, redact_mrn: bool, redact_acc: bool, redact_pd: bool) -> str:
    row_id   = row.get("row_id", "")
    mrn      = str(row.get("MRN", ""))
    acc      = str(row.get("accession number", ""))
    pdate    = row.get("procedure date", "")
    scar_type= str(row.get("scar_type", ""))
    lvef_val = row.get("lvef", float("nan"))
    lvef_str = "" if pd.isna(lvef_val) else f"{float(lvef_val):.0f}%"
    has_lge  = bool(row.get("has_lge", False))
    has_scar = bool(row.get("has_scar", False))
    patt     = str(row.get("lge_pattern", "") or "")
    loc      = str(row.get("lge_location", "") or "")

    pills = [
        _pill_plain("row_id", str(row_id)),
        _pill_redactable("MRN",            mrn,        redact_mrn,  "toggleMRN"),
        _pill_redactable("procedure_date", str(pdate), redact_pd,   "togglePDate"),
        _pill_redactable("accession",      acc,        redact_acc,  "toggleAcc"),
        _pill_plain("scar_type", scar_type),
        _pill_plain("LVEF",      lvef_str),
        _pill_plain("has_LGE",   "YES" if has_lge  else "NO"),
        _pill_plain("has_scar",  "YES" if has_scar else "NO"),
    ]
    if patt:
        pills.append(_pill_plain("LGE_pattern",   patt))
    if loc:
        pills.append(_pill_plain("LGE_location",  loc))
    return "\n".join(pills)



def _hl_rules():
    rules = []
    rules += [(r"\bLVEF[:\s=]*\(?\s*(\d{1,3})\s*%?\)?\b","lvef"),
              (r"\bejection\s+fraction[:\s=]*\s*\d{1,3}\s*%?\b","lvef")]
    rules += [(r"\blate\s+gadolinium\s+enhancement\b","lge"),(r"\bdelayed\s+(contrast\s+)?enhancement\b","lge"),
              (r"\bLGE\b","lge"),
              # ischemic patterns
              (r"\bsub[-]?endocardial\b","lge"),
              (r"\bendocardial\s+(?:scar|lge|delayed|late)\b","lge"),
              (r"\bembol\w+\b","lge"),
              # non-ischemic patterns
              (r"\bmid[-\s]?wall\b|\bmid[-\s]?myocardial\b|\bmidmyocardial\b","lge"),
              (r"\bmesocardial\b|\bmesocardium\b","lge"),
              (r"\bsub[-]?epicardial\b|\bepicardial\b|\bepicardium\b|\bepicard\b","lge"),
              (r"\bpatchy\b","lge"),
              (r"\bnonvascular\b|\bnon[-\s]vascular\b","lge"),
              (r"\bhazy\s+(?:nonvascular\s+)?mesocardial\b","lge"),
              # linear enhancement — the key non-ischemic pattern
              (r"\blinear\s+\w*\s*(?:area\s+of\s+)?(?:late|delayed|gadolinium|lge)?\s*enhancement\b","lge"),
              (r"\blinear\s+lge\b|\blinear\s+delayed\b|\blinear\s+late\b","lge")]
    rules += [(r"\bno\s+(?:late\s+gadolinium|delayed)\s+enhancement\b","nocontrast"),
              (r"\bno\s+lge\b","nocontrast")]
    rules += [(r"\bLAD\b|\bleft\s+anterior\s+descending\b","territory"),
              (r"\bRCA\b|\bright\s+coronary\b","territory"),
              (r"\bLCX?\b|\bleft\s+circumflex\b","territory")]
    rules += [(r"\bischemic\b|\binfarct(?:ion)?\b|\bcoronary\s+artery\s+disease\b","key"),
              (r"\bnon[-\s]?ischemic\b|\bmyocarditis\b|\bsarcoid(?:osis)?\b","key"),
              (r"\bamyloid(?:osis)?\b|\bhcm\b|\bdcm\b|\bfabry\b","key")]
    rules += [(r"\binferolateral\b|\banterolateral\b|\binferior\b|\banterior\b","loc"),
              (r"\blateral\b|\bseptal\b|\bapical\b|\bbasal\b","loc")]
    return rules

def _spans(text, rules):
    occ, spans = [False]*(len(text)+1), []
    for pat, lbl in rules:
        for m in re.finditer(pat, text, re.IGNORECASE):
            a, b = m.start(), m.end()
            if any(occ[a:b]): continue
            for i in range(a,b): occ[i]=True
            spans.append((a,b,lbl))
    return sorted(spans)

def _render(text, spans):
    out, last = [], 0
    for a, b, lbl in spans:
        out.append(_escape(text[last:a]))
        out.append(f'<span class="hl hl-{lbl}">{_escape(text[a:b])}</span>')
        last = b
    out.append(_escape(text[last:]))
    return "".join(out)

_CSS = """body{font-family:-apple-system,"Segoe UI",Arial,sans-serif;background:#f5f5f5;margin:0;padding:20px;max-width:1600px;margin:auto}
.top{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:16px 18px;border-radius:14px;box-shadow:0 4px 10px rgba(0,0,0,.12);margin-bottom:12px}
.top h1{margin:0;font-size:18px;font-weight:900}
.legend{background:#fff;border-radius:14px;padding:12px 14px;box-shadow:0 2px 8px rgba(0,0,0,.1);margin-bottom:12px;display:flex;gap:10px;flex-wrap:wrap}
.tag{padding:6px 10px;border-radius:999px;font-weight:900;font-size:12px;border:2px solid transparent}
.tag-lvef{background:#ffe8e8;border-color:#ff4d4d}.tag-key{background:#fff4e5;border-color:#ff8c1a}
.tag-lge{background:#e6f2ff;border-color:#3399ff}.tag-ter{background:#e9fff3;border-color:#00b894}
.tag-loc{background:#fffbe6;border-color:#f1c40f}.tag-nc{background:#f3e5ff;border-color:#9b59b6}
.card{background:#fff;border-radius:14px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,.1);margin-bottom:14px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.pill{display:inline-flex;gap:8px;align-items:baseline;padding:7px 10px;background:#f2f4ff;border-radius:999px;border:2px solid rgba(102,126,234,.45);font-size:13px;font-weight:800;color:#2c3e50}
.report{background:#fff;border-radius:12px;padding:14px;line-height:1.85;font-size:14px;white-space:pre-wrap;border:1px solid rgba(0,0,0,.06)}
.hl{padding:2px 6px;border-radius:8px;border:2px solid transparent;font-weight:900}
.hl-lvef{background:#ffe8e8;border-color:#ff4d4d}.hl-key{background:#fff4e5;border-color:#ff8c1a}
.hl-lge{background:#e6f2ff;border-color:#3399ff}.hl-territory{background:#e9fff3;border-color:#00b894}
.hl-loc{background:#fffbe6;border-color:#f1c40f}.hl-nocontrast{background:#f3e5ff;border-color:#9b59b6}
.toggle-btn{margin-left:8px;padding:4px 10px;border-radius:999px;border:2px solid rgba(102,126,234,.6);background:#fff;font-weight:900;cursor:pointer;font-size:12px}
.toggle-btn:hover{background:#f2f4ff}
.redacted{display:inline-block;min-width:90px;text-align:center;letter-spacing:1px}"""

def write_gallery(df_cls, title, out_path, redact_mrn, redact_acc, redact_pd):
    rules = _hl_rules()
    cards = []
    for _, row in df_cls.iterrows():
        text  = str(row["cardiac MRI text"])
        body  = _render(text, _spans(text, rules))
        pills = _meta_pills(row, redact_mrn, redact_acc, redact_pd)
        cards.append(f'<div class="card"><div class="meta">{pills}</div><div class="report">{body}</div></div>')

    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8"/><title>{_escape(title)}</title>'
            f'<style>{_CSS}</style></head><body>'
            f'<div class="top"><h1>{_escape(title)}</h1></div>'
            f'<div class="legend"><b>Legend:</b>'
            f'<span class="tag tag-lvef">LVEF</span>'
            f'<span class="tag tag-key">Keywords/Pattern</span>'
            f'<span class="tag tag-lge">LGE</span>'
            f'<span class="tag tag-nc">No LGE</span>'
            f'<span class="tag tag-ter">Coronary Territory</span>'
            f'<span class="tag tag-loc">Location</span></div>'
            + "".join(cards)
            + """<script>
function _toggle(btn){
  var pill=btn.parentElement;
  var el=pill.querySelector(".redacted");
  if(!el)return;
  var raw=el.getAttribute("data-raw")||"";
  var cur=(el.textContent||"").trim();
  if(cur==="REDACTED"){el.textContent=raw;btn.textContent="Hide";}
  else{el.textContent="REDACTED";btn.textContent="Show";}
}
function toggleMRN(btn){_toggle(btn);}
function toggleAcc(btn){_toggle(btn);}
function togglePDate(btn){_toggle(btn);}
</script></body></html>""")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Gallery: {out_path}")

def write_index(out_path, items, counts=None):
    # Per-class visual config: (c1, c2, icon, description)
    _CLASS_META = {
        "Ischemic":      ("#ff6b6b", "#ee5a24", "🫀", "Subendocardial LGE in coronary distribution"),
        "Non-ischemic":  ("#a29bfe", "#6c5ce7", "💜", "Mid-wall, epicardial, or disease-specific pattern"),
        "Mixed":         ("#fd79a8", "#e84393", "🔀", "Coexisting ischemic + non-ischemic features"),
        "No-scar":       ("#55efc4", "#00b894", "✅", "LGE imaging performed — no enhancement detected"),
        "Not-specified": ("#ffeaa7", "#fdcb6e", "❓", "LGE present but pattern indeterminate"),
    }
    counts = counts or {}
    cards_html = []
    for fn, lbl in items:
        c1, c2, icon, desc = _CLASS_META.get(lbl, ("#dfe6e9", "#b2bec3", "📄", ""))
        n = counts.get(lbl.lower(), counts.get(lbl, None))
        count_badge = (f'<div class="card-count">{n:,} reports</div>' if n is not None else "")
        cards_html.append(f"""
        <a href="{_escape(fn)}" class="class-card" style="--c1:{c1};--c2:{c2};">
          <div class="card-icon">{icon}</div>
          <div class="card-body">
            <div class="card-label">{_escape(lbl)}{count_badge}</div>
            <div class="card-desc">{_escape(desc)}</div>
          </div>
          <div class="card-arrow">→</div>
        </a>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>CMR Scar Phenotyping — 5-Class NER Gallery</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    background: #f0f2f8;
    min-height: 100vh;
    padding: 32px 20px;
  }}
  .page-wrap {{
    max-width: 720px;
    margin: 0 auto;
  }}
  .hero {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 28px 28px 24px;
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(102,126,234,0.35);
    color: white;
  }}
  .hero-eyebrow {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 6px;
  }}
  .hero h1 {{
    font-size: 22px;
    font-weight: 900;
    line-height: 1.3;
    margin-bottom: 8px;
  }}
  .hero-sub {{
    font-size: 13px;
    opacity: 0.8;
    font-weight: 500;
  }}
  .section-label {{
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8892b0;
    margin-bottom: 12px;
    padding-left: 4px;
  }}
  .grid {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .class-card {{
    display: flex;
    align-items: center;
    gap: 16px;
    background: white;
    border-radius: 16px;
    padding: 18px 20px;
    text-decoration: none;
    color: #2c3e50;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border: 2px solid transparent;
    transition: box-shadow 0.18s, transform 0.18s, border-color 0.18s;
    position: relative;
    overflow: hidden;
  }}
  .class-card::before {{
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, var(--c1), var(--c2));
    border-radius: 16px 0 0 16px;
  }}
  .class-card:hover {{
    box-shadow: 0 6px 22px rgba(0,0,0,0.13);
    transform: translateY(-2px);
    border-color: rgba(102,126,234,0.3);
  }}
  .card-icon {{
    font-size: 26px;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--c1) 0%, var(--c2) 100%);
    border-radius: 12px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    filter: saturate(0.85) brightness(1.1);
  }}
  .card-body {{
    flex: 1;
    min-width: 0;
  }}
  .card-label {{
    font-size: 15px;
    font-weight: 900;
    color: #1a1a2e;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .card-count {{
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg, var(--c1), var(--c2));
    border-radius: 999px;
    padding: 2px 9px;
    white-space: nowrap;
  }}
  .card-desc {{
    font-size: 12px;
    color: #636e84;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .card-arrow {{
    font-size: 18px;
    color: #c0c8d8;
    font-weight: 900;
    flex-shrink: 0;
    transition: color 0.18s, transform 0.18s;
  }}
  .class-card:hover .card-arrow {{
    color: #667eea;
    transform: translateX(3px);
  }}
</style>
</head>
<body>
  <div class="page-wrap">
    <div class="hero">
      <div class="hero-eyebrow">NLP Pipeline</div>
      <h1>CMR Scar Phenotyping<br>5-Class NER Gallery</h1>
      <div class="hero-sub">Select a scar phenotype class to explore annotated reports</div>
    </div>
    <div class="section-label">Phenotype Classes</div>
    <div class="grid">{"".join(cards_html)}</div>
  </div>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Index:   {out_path}")

def select_gallery_subset(df, cls, n, require_lge, random_sample=False, random_seed=42):
    sub = df[df["scar_type"] == cls].copy()
    if require_lge and cls in ("ischemic","non-ischemic","mixed"):
        sub = sub[sub["has_lge"] == True]
    if len(sub) == 0: return sub
    if random_sample:
        return sub.sample(min(n, len(sub)), random_state=random_seed).reset_index(drop=True)
    t = sub["cardiac MRI text"].astype(str).str.lower()
    cues = {
        "ischemic":      ["subendocardial","sub-endocardial","endocardial","infarct","infarction","rca","lad","lcx","embolic","embolism"],
        "non-ischemic":  ["midwall","mid-wall","mid-myocardial","mesocardial","mesocardium","epicardial","epicard",
                          "subepicardial","sub-epicardial",
                          "linear enhancement","linear lge","patchy","myocarditis","sarcoid","hcm","dcm",
                          "rv insertion","insertion point"],
        "mixed":         ["subendocardial","midwall","epicardial","myocarditis"],
        "no-scar":       ["lge","delayed enhancement","no scar"],
        "not-specified": [],
    }
    kws = cues.get(cls, [])
    score = sum(t.str.contains(re.escape(k)).astype(int) for k in kws) if kws else t.str.len().clip(upper=5000)
    sub["_score"] = score
    return sub.sort_values(["_score","row_id"], ascending=[False,True]).head(n).drop(columns=["_score"])



# EXPERT ANNOTATION GALLERY — Functions
# =============================================================================

_EXPERT_EXCLUDED = {"RPT0057", "RPT0112", "RPT0190", "RPT0256", "RPT0258", "RPT0263"}

_EXPERT_CLASS_COLORS = {
    "ischemic":      ("#C0392B", "#FADBD8"),
    "non-ischemic":  ("#1A5276", "#D6EAF8"),
    "mixed":         ("#6C3483", "#E8DAEF"),
    "no-scar":       ("#1E8449", "#D5F5E3"),
    "not-specified": ("#784212", "#FDEBD0"),
}
_EXPERT_NONE_COLOR = ("#888888", "#F5F5F5")
_EXPERT_CLASS_DISPLAY = {
    "ischemic": "Ischemic", "non-ischemic": "Non-ischemic",
    "mixed": "Mixed", "no-scar": "No-scar", "not-specified": "Not-specified",
}

def _expert_load(two_expert_xlsx_path: str) -> pd.DataFrame:
    """
    Load all expert annotation data from Two_Expert_Annotation_Comparision.xlsx.
    This single file contains:
      - Report text (from Reader1_Annotation sheet)
      - Reader 1 labels + comments (Reader1_Annotation sheet)
      - Reader 2 labels + comments (Reader2_Annotation sheet)
      - Accession numbers and procedure dates
    The 6 excluded reports are removed automatically.
    """
    print(f"  Loading expert annotations: {two_expert_xlsx_path}")
    xl = pd.read_excel(two_expert_xlsx_path, sheet_name=None, header=None)

    EXCLUDED = _EXPERT_EXCLUDED
    text_map, r1_labels, r1_comments = {}, {}, {}
    accession_map, date_map = {}, {}

    # ── Reader 1: text, labels, accession, date ───────────────────────────────
    for sname in ["Reader1_Annotation (Onur)", "Reader1_Annotation"]:
        if sname not in xl:
            continue
        raw = xl[sname]
        hdr = raw.iloc[2].tolist()
        dr  = raw.iloc[3:].copy(); dr.columns = hdr; dr = dr.reset_index(drop=True)
        dr  = dr[~dr["Report ID"].astype(str).str.strip().isin(EXCLUDED)]
        for _, row in dr.iterrows():
            rpt = str(row.get("Report ID", "") or "").strip()
            if not rpt:
                continue
            txt = str(row.get("CMR Report Text", "") or "").strip()
            if txt and txt.lower() not in ("nan", "none"):
                text_map[rpt] = txt
            for col in ["Reader 1 Label ▼", "Reader 1 Label", "Reader1 Label"]:
                v = str(row.get(col, "") or "").strip()
                if v and v.lower() not in ("nan", "none"):
                    r1_labels[rpt] = v
                    break
            cmt = str(row.get("Comments (optional)", "") or "").strip()
            r1_comments[rpt] = "" if cmt.lower() in ("nan", "none", "") else cmt
            acc = str(row.get("Accession #", "") or "").strip()
            if acc and acc.lower() not in ("nan", "none"):
                accession_map[rpt] = acc
            dt = row.get("Procedure Date", None)
            if dt is not None:
                date_map[rpt] = dt
        print(f"  Reader 1: {len(r1_labels)} labels, {len(text_map)} texts")
        break

    # ── Reader 2: labels + comments ──────────────────────────────────────────
    r2_labels, r2_comments = {}, {}
    for sname in ["Reader2_Annotation (Amir)", "Reader2_Annotation"]:
        if sname not in xl:
            continue
        raw = xl[sname]
        hdr = raw.iloc[2].tolist()
        dr  = raw.iloc[3:].copy(); dr.columns = hdr; dr = dr.reset_index(drop=True)
        dr  = dr[~dr["Report ID"].astype(str).str.strip().isin(EXCLUDED)]
        for _, row in dr.iterrows():
            rpt = str(row.get("Report ID", "") or "").strip()
            if not rpt:
                continue
            for col in ["Reader 2 Label ▼", "Reader 2 Label", "Reader2 Label"]:
                v = str(row.get(col, "") or "").strip()
                if v and v.lower() not in ("nan", "none"):
                    r2_labels[rpt] = v
                    break
            cmt = str(row.get("Comments (optional)", "") or "").strip()
            r2_comments[rpt] = "" if cmt.lower() in ("nan", "none", "") else cmt
        print(f"  Reader 2: {len(r2_labels)} labels")
        break

    # ── Build master DataFrame ────────────────────────────────────────────────
    all_rpts = sorted(set(r1_labels) | set(r2_labels))
    df = pd.DataFrame({"report_id": all_rpts})
    df["MRN"]                   = ""   # not in Two_Expert file; leave blank
    df["accession_number_real"] = df["report_id"].map(accession_map).fillna("")
    df["Procedure Date"]        = pd.to_datetime(
        df["report_id"].map(date_map), errors="coerce"
    )
    df["nlp_class"]   = ""   # filled by NLP pipeline if annotation_list available
    df["r1_label"]    = df["report_id"].map(r1_labels).fillna("")
    df["r1_comment"]  = df["report_id"].map(r1_comments).fillna("")
    df["r2_label"]    = df["report_id"].map(r2_labels).fillna("")
    df["r2_comment"]  = df["report_id"].map(r2_comments).fillna("")
    df["llm_8b"]      = ""
    df["ft5_full"]    = ""
    df["ft5_impr"]    = ""
    df["report_text"] = df["report_id"].map(text_map).fillna("")
    print(f"  Master: {len(df)} reports ({(df['report_text'].str.len() > 0).sum()} with text)")
    return df


# ── Expert highlight rules (same NLP rules — transmural excluded) ─────────────
def _expert_hl_rules():
    return [
        (r"\bLVEF[:\s=]*\(?\s*(\d{1,3})\s*%?\)?\b",              "lvef"),
        (r"\bejection\s+fraction[:\s=]*\s*\d{1,3}\s*%?\b",        "lvef"),
        (r"\blate\s+gadolinium\s+enhancement\b",                   "lge"),
        (r"\bdelayed\s+(contrast\s+)?enhancement\b",               "lge"),
        (r"\bLGE\b",                                               "lge"),
        (r"\bsub[-]?endocardial\b",                                "lge"),
        (r"\bendocardial\s+(?:scar|lge|delayed|late)\b",           "lge"),
        (r"\bembol\w+\b",                                          "lge"),
        (r"\bmid[-\s]?wall\b|\bmid[-\s]?myocardial\b|\bmidmyocardial\b", "lge"),
        (r"\bmesocardial\b|\bmesocardium\b",                       "lge"),
        (r"\bsub[-]?epicardial\b|\bepicardial\b|\bepicardium\b|\bepicard\b", "lge"),
        (r"\bpatchy\b",                                            "lge"),
        (r"\blinear\s+\w*\s*(?:area\s+of\s+)?(?:late|delayed|gadolinium|lge)?\s*enhancement\b", "lge"),
        (r"\blinear\s+lge\b|\blinear\s+delayed\b|\blinear\s+late\b", "lge"),
        (r"\binsertion\s+point\b|\brv\s+insertion\b",              "lge"),
        (r"\bno\s+(?:late\s+gadolinium|delayed)\s+enhancement\b",  "nocontrast"),
        (r"\bno\s+lge\b",                                          "nocontrast"),
        (r"\bLAD\b|\bleft\s+anterior\s+descending\b",              "territory"),
        (r"\bRCA\b|\bright\s+coronary\b",                          "territory"),
        (r"\bLCX?\b|\bleft\s+circumflex\b",                        "territory"),
        (r"\bischemic\b|\binfarct(?:ion)?\b|\bcoronary\s+artery\s+disease\b", "key"),
        (r"\bnon[-\s]?ischemic\b|\bmyocarditis\b|\bsarcoid(?:osis)?\b", "key"),
        (r"\bamyloid(?:osis)?\b|\bhcm\b|\bdcm\b|\bfabry\b",        "key"),
        (r"\binferolateral\b|\banterolateral\b|\binferior\b|\banterior\b", "loc"),
        (r"\blateral\b|\bseptal\b|\bapical\b|\bbasal\b",           "loc"),
    ]


def _expert_highlight(text: str) -> str:
    rules = _expert_hl_rules()
    occ   = [False] * (len(text) + 1)
    spans = []
    for pat, lbl in rules:
        for m in re.finditer(pat, text, re.IGNORECASE):
            a, b = m.start(), m.end()
            if any(occ[a:b]):
                continue
            for i in range(a, b):
                occ[i] = True
            spans.append((a, b, lbl))
    spans.sort()
    out, last = [], 0
    for a, b, lbl in spans:
        out.append(_escape(text[last:a]))
        out.append(f'<span class="hl hl-{lbl}">{_escape(text[a:b])}</span>')
        last = b
    out.append(_escape(text[last:]))
    return "".join(out)


# ── CSS + JS for expert galleries ─────────────────────────────────────────────
_EXPERT_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Arial, sans-serif;
  background: #f0f2f8; margin: 0 auto; padding: 20px; max-width: 1440px; }
.page-top { background: linear-gradient(135deg, #667eea, #764ba2);
  color:#fff; padding:18px 22px 16px; border-radius:16px;
  box-shadow:0 4px 14px rgba(102,126,234,.35); margin-bottom:14px; }
.page-top h1 { margin:0; font-size:20px; font-weight:900; }
.page-top .sub { font-size:12px; opacity:.8; margin-top:4px; }
.legend { background:#fff; border-radius:14px; padding:10px 14px;
  box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:16px;
  display:flex; gap:8px; flex-wrap:wrap; align-items:center; font-size:12px; font-weight:700; color:#555; }
.tag { padding:4px 10px; border-radius:999px; font-weight:900; font-size:11px; border:2px solid transparent; }
.tag-lvef{background:#ffe8e8;border-color:#ff4d4d}
.tag-key {background:#fff4e5;border-color:#ff8c1a}
.tag-lge {background:#e6f2ff;border-color:#3399ff}
.tag-ter {background:#e9fff3;border-color:#00b894}
.tag-loc {background:#fffbe6;border-color:#f1c40f}
.tag-nc  {background:#f3e5ff;border-color:#9b59b6}
.card { background:#fff; border-radius:16px;
  box-shadow:0 2px 10px rgba(0,0,0,.09); margin-bottom:18px;
  overflow:hidden; border:1.5px solid rgba(0,0,0,.06); }
.card-header { padding:14px 18px 12px; border-bottom:1.5px solid #f0f2f8; background:#fafbff; }
.header-top { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:11px; }
.report-id { font-size:15px; font-weight:900; color:#1a1a2e; }
.meta-chip { display:inline-flex; align-items:center; gap:5px;
  padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700;
  background:#f0f2f8; color:#444; border:1.5px solid #dde1ef; }
.meta-chip b { color:#666; font-weight:800; }
.toggle-btn { margin-left:5px; padding:2px 8px; border-radius:999px;
  border:1.5px solid rgba(102,126,234,.55); background:#fff;
  font-weight:900; cursor:pointer; font-size:10px; color:#667eea; }
.toggle-btn:hover { background:#f0f2ff; }
.redacted { font-style:italic; color:#aaa; min-width:70px; display:inline-block; text-align:center; }
.label-row { display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; }
.label-box { display:flex; flex-direction:column; gap:3px;
  padding:9px 14px; border-radius:12px; border:2px solid transparent; min-width:130px; flex:1; }
.lb-title { font-size:10px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; opacity:.6; }
.lb-value { font-size:14px; font-weight:900; }
.lb-comment { font-size:11px; font-style:italic; opacity:.7; margin-top:2px; }
.badge { display:inline-block; font-size:11px; font-weight:900;
  padding:2px 8px; border-radius:999px; margin-top:3px; }
.b-match   { background:#e8f5e9; color:#1e8449; border:1.5px solid #82c78f; }
.b-mismatch{ background:#fdecea; color:#c0392b; border:1.5px solid #e88484; }
.b-na      { background:#f5f5f5; color:#888;    border:1.5px solid #ccc; }
.report-body { padding:16px 18px; line-height:1.9; font-size:13.5px;
  white-space:pre-wrap; color:#2c3e50; overflow-x:auto; }
.hl{padding:2px 5px;border-radius:7px;border:2px solid transparent;font-weight:900}
.hl-lvef      {background:#ffe8e8;border-color:#ff4d4d}
.hl-key       {background:#fff4e5;border-color:#ff8c1a}
.hl-lge       {background:#e6f2ff;border-color:#3399ff}
.hl-territory {background:#e9fff3;border-color:#00b894}
.hl-loc       {background:#fffbe6;border-color:#f1c40f}
.hl-nocontrast{background:#f3e5ff;border-color:#9b59b6}
"""

_EXPERT_JS = """
<script>
function _tog(btn){
  var chip=btn.closest('.meta-chip'),el=chip&&chip.querySelector('.redacted');
  if(!el)return;
  var raw=el.getAttribute('data-raw')||'';
  if((el.textContent||'').trim()==='REDACTED'){el.textContent=raw;btn.textContent='Hide';}
  else{el.textContent='REDACTED';btn.textContent='Show';}
}
function toggleMRN(b){_tog(b);}
function toggleAcc(b){_tog(b);}
function togglePDate(b){_tog(b);}
</script>
"""

# ── Card builders ──────────────────────────────────────────────────────────────
def _exp_chip(label, value):
    return f'<span class="meta-chip"><b>{_escape(label)}</b> {_escape(value)}</span>'

def _exp_chip_redact(label, value, redact, fn):
    if not redact:
        return _exp_chip(label, value)
    return (f'<span class="meta-chip"><b>{_escape(label)}</b>'
            f'<span class="redacted" data-raw="{_escape(value)}">REDACTED</span>'
            f'<button class="toggle-btn" onclick="{fn}(this)">Show</button></span>')

def _exp_badge(label, ref):
    if not label or label.lower() in ("nan", "none", ""):
        return ''
    if not ref or ref.lower() in ("nan", "none", ""):
        return '<span class="badge b-na">no ref</span>'
    if label == ref:
        return '<span class="badge b-match">&#10003;</span>'
    return '<span class="badge b-mismatch">&#10007;</span>'

def _exp_label_box(title, label, ref_label=None, comment="", show_badge=True):
    fg, bg = _EXPERT_CLASS_COLORS.get(label, _EXPERT_NONE_COLOR)
    display = label if label and label.lower() not in ("nan", "none", "") else "—"
    badge_html = (_exp_badge(label, ref_label) if show_badge and ref_label is not None else "")
    cmt_html = (f'<div class="lb-comment">&#128172; {_escape(comment)}</div>'
                if comment and comment.lower() not in ("nan", "none", "") else "")
    return (f'<div class="label-box" style="background:{bg};border-color:{fg}44;">'
            f'<div class="lb-title">{_escape(title)}</div>'
            f'<div class="lb-value" style="color:{fg};">{_escape(display)}</div>'
            f'{badge_html}{cmt_html}'
            f'</div>')

def _exp_build_header(row, redact_mrn, redact_acc, redact_pd):
    rid   = str(row.get("report_id", ""))
    mrn   = str(row.get("MRN", ""))
    acc   = str(row.get("accession_number_real", ""))
    praw  = row.get("Procedure Date", "")
    pdate = str(praw)[:10] if praw and str(praw) != "NaT" else "—"
    r1    = str(row.get("r1_label", "") or "")
    r2    = str(row.get("r2_label", "") or "")
    r1c   = str(row.get("r1_comment", "") or "")
    r2c   = str(row.get("r2_comment", "") or "")
    nlp   = str(row.get("nlp_class", "") or "")
    llm   = str(row.get("llm_8b", "") or "")
    ft5i  = str(row.get("ft5_impr", "") or "")
    top = (f'<div class="header-top">'
           f'<span class="report-id">{_escape(rid)}</span>'
           + _exp_chip_redact("MRN",       mrn,   redact_mrn,  "toggleMRN")
           + _exp_chip_redact("Accession", acc,   redact_acc,  "toggleAcc")
           + _exp_chip_redact("Date",      pdate, redact_pd,   "togglePDate")
           + '</div>')
    labels = (f'<div class="label-row">'
              + _exp_label_box("Reader 1", r1, ref_label=None, comment=r1c, show_badge=False)
              + _exp_label_box("Reader 2", r2, ref_label=r1, comment=r2c)
              + _exp_label_box("NLP Pipeline", nlp, ref_label=r1)
              + _exp_label_box("LLM-8B", llm, ref_label=r1)
              + _exp_label_box("FlanT5-Impr", ft5i, ref_label=r1)
              + '</div>')
    return f'<div class="card-header">{top}{labels}</div>'


# ── Gallery + index writers ────────────────────────────────────────────────────
def _exp_gallery_html(title, subtitle, cards_html):
    legend = ('<div class="legend"><b>Highlight legend:</b>'
              '<span class="tag tag-lvef">LVEF</span>'
              '<span class="tag tag-key">Pattern/Keyword</span>'
              '<span class="tag tag-lge">LGE</span>'
              '<span class="tag tag-nc">No LGE</span>'
              '<span class="tag tag-ter">Coronary Territory</span>'
              '<span class="tag tag-loc">Location</span>'
              '</div>')
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"/>'
            f'<title>{_escape(title)}</title>'
            f'<style>{_EXPERT_CSS}</style></head><body>'
            f'<div class="page-top"><h1>{_escape(title)}</h1>'
            f'<div class="sub">{_escape(subtitle)}</div></div>'
            + legend + cards_html + _EXPERT_JS + '</body></html>')


def _exp_write_gallery(df_sub, title, subtitle, out_path, redact_mrn, redact_acc, redact_pd):
    cards = []
    for _, row in df_sub.iterrows():
        text = str(row.get("report_text", "") or "")
        body = _expert_highlight(text)
        hdr  = _exp_build_header(row, redact_mrn, redact_acc, redact_pd)
        cards.append(f'<div class="card">{hdr}<div class="report-body">{body}</div></div>')
    html = _exp_gallery_html(title, subtitle, "".join(cards))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  {os.path.basename(out_path):<55} {len(df_sub):>4} reports")


_EXPERT_INDEX_META = {
    "Ischemic":         ("#ff6b6b", "#ee5a24", "&#x1FAC0;", "Subendocardial LGE in coronary distribution"),
    "Non-ischemic":     ("#a29bfe", "#6c5ce7", "&#x1F49C;", "Mid-wall, epicardial, or disease-specific pattern"),
    "Mixed":            ("#fd79a8", "#e84393", "&#x1F500;", "Coexisting ischemic + non-ischemic features"),
    "No-scar":          ("#55efc4", "#00b894", "&#x2705;",  "LGE imaging performed — no enhancement detected"),
    "Not-specified":    ("#ffeaa7", "#fdcb6e", "&#x2753;",  "LGE present but pattern indeterminate"),
    "Agreements":       ("#00b4d8", "#0077b6", "&#x1F91D;", "Cases where Reader 1 = Reader 2"),
    "Mismatches":       ("#e63946", "#9d0208", "&#x26A0;",  "Cases where Reader 1 \u2260 Reader 2"),
    "NLP Disagreement": ("#f77f00", "#d62828", "&#x1F4CA;", "Expert agree, but NLP gives a different label"),
    "NLP on Mismatches":("#6a0572", "#c77dff", "&#x1F9E9;", "What NLP classified for each expert disagreement"),
}

def _exp_index_card(fn, label, n, desc=""):
    meta = _EXPERT_INDEX_META.get(label, ("#dfe6e9", "#b2bec3", "&#x1F4C4;", ""))
    c1, c2, icon, default_desc = meta
    desc = desc or default_desc
    return (f'<a href="{_escape(fn)}" class="class-card" style="--c1:{c1};--c2:{c2};">'
            f'<div class="card-icon">{icon}</div>'
            f'<div class="card-body">'
            f'<div class="card-label">{_escape(label)}</div>'
            f'<div class="card-desc">{desc} &nbsp;&middot;&nbsp; n={n}</div>'
            f'</div><div class="card-arrow">&#8594;</div></a>')

def _exp_write_index(out_path, sections, nlp_stats=None, nlp_overall=None,
                     n_agree=None, n_total=None, n_mismatch=None):
    section_html = []
    for sec_title, items in sections:
        cards = "".join(_exp_index_card(fn, lbl, n, desc) for fn, lbl, n, desc in items)
        section_html.append(
            f'<div class="section-label">{_escape(sec_title)}</div>'
            f'<div class="grid">{cards}</div>'
            f'<div style="height:22px"></div>'
        )

    # Build accuracy panel HTML
    acc_html = ""
    if nlp_stats and nlp_overall is not None:
        CLASS_COLORS_HEX = {
            "ischemic":      ("#C0392B", "#FADBD8"),
            "non-ischemic":  ("#1A5276", "#D6EAF8"),
            "mixed":         ("#6C3483", "#E8DAEF"),
            "no-scar":       ("#1E8449", "#D5F5E3"),
            "not-specified": ("#784212", "#FDEBD0"),
        }
        rows = ""
        for cls in ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]:
            if cls not in nlp_stats: continue
            correct, total, pct = nlp_stats[cls]
            if total == 0: continue
            fg, bg = CLASS_COLORS_HEX.get(cls, ("#333", "#eee"))
            bar_w = max(4, int(pct))
            rows += (
                f'<div class="acc-row">'
                f'<div class="acc-label" style="color:{fg}">{cls}</div>'
                f'<div class="acc-bar-wrap">'
                f'<div class="acc-bar" style="width:{bar_w}%;background:{fg}"></div>'
                f'</div>'
                f'<div class="acc-pct">{pct:.0f}%</div>'
                f'<div class="acc-n" style="color:{fg}">{correct}/{total}</div>'
                f'</div>'
            )
        inter_reader = ""
        if n_agree is not None and n_total is not None:
            ir_pct = n_agree / max(1, n_total) * 100
            inter_reader = (
                f'<div class="acc-divider"></div>'
                f'<div class="acc-row">'
                f'<div class="acc-label" style="color:#0077b6">Inter-reader (R1=R2)</div>'
                f'<div class="acc-bar-wrap">'
                f'<div class="acc-bar" style="width:{ir_pct:.0f}%;background:#0077b6"></div>'
                f'</div>'
                f'<div class="acc-pct">{ir_pct:.1f}%</div>'
                f'<div class="acc-n" style="color:#0077b6">{n_agree}/{n_total}</div>'
                f'</div>'
            )
        acc_html = f"""
<div class="section-label" style="margin-top:4px">NLP Accuracy vs Expert Annotations</div>
<div class="acc-panel">
  <div class="acc-overall">Overall NLP accuracy: <strong>{nlp_overall:.1f}%</strong>
    &nbsp;({int(sum(s[0] for s in nlp_stats.values()))}/{n_total} reports correct)</div>
  {rows}
  {inter_reader}
</div>
<div style="height:22px"></div>
"""

    css_extra = """
.acc-panel{background:#fff;border-radius:16px;padding:18px 20px;
  box-shadow:0 2px 10px rgba(0,0,0,.07);margin-bottom:4px;display:flex;
  flex-direction:column;gap:10px}
.acc-overall{font-size:13px;font-weight:700;color:#2c3e50;padding-bottom:4px;
  border-bottom:1.5px solid #f0f2f8;margin-bottom:2px}
.acc-row{display:flex;align-items:center;gap:10px}
.acc-label{font-size:12px;font-weight:800;width:120px;flex-shrink:0}
.acc-bar-wrap{flex:1;background:#f0f2f8;border-radius:999px;height:10px;overflow:hidden}
.acc-bar{height:100%;border-radius:999px;transition:width .3s}
.acc-pct{font-size:12px;font-weight:900;color:#2c3e50;width:38px;text-align:right;flex-shrink:0}
.acc-n{font-size:11px;font-weight:700;width:52px;text-align:right;flex-shrink:0}
.acc-divider{border-top:1px dashed #dde1ef;margin:2px 0}
"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>CMR Expert Annotation Gallery</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  background:#f0f2f8;min-height:100vh;padding:32px 20px}}
.page-wrap{{max-width:740px;margin:0 auto}}
.hero{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  border-radius:20px;padding:28px 28px 24px;margin-bottom:28px;
  box-shadow:0 8px 24px rgba(102,126,234,.35);color:white}}
.hero-eye{{font-size:11px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;opacity:.75;margin-bottom:6px}}
.hero h1{{font-size:22px;font-weight:900;line-height:1.3;margin-bottom:8px}}
.hero-sub{{font-size:13px;opacity:.8;font-weight:500}}
.section-label{{font-size:11px;font-weight:800;letter-spacing:2px;
  text-transform:uppercase;color:#8892b0;margin-bottom:12px;padding-left:4px}}
.grid{{display:flex;flex-direction:column;gap:12px}}
.class-card{{display:flex;align-items:center;gap:16px;background:white;
  border-radius:16px;padding:18px 20px;text-decoration:none;color:#2c3e50;
  box-shadow:0 2px 10px rgba(0,0,0,.07);border:2px solid transparent;
  transition:box-shadow .18s,transform .18s,border-color .18s;
  position:relative;overflow:hidden}}
.class-card::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;
  background:linear-gradient(180deg,var(--c1),var(--c2));border-radius:16px 0 0 16px}}
.class-card:hover{{box-shadow:0 6px 22px rgba(0,0,0,.13);
  transform:translateY(-2px);border-color:rgba(102,126,234,.3)}}
.card-icon{{font-size:26px;width:44px;height:44px;display:flex;align-items:center;
  justify-content:center;background:linear-gradient(135deg,var(--c1) 0%,var(--c2) 100%);
  border-radius:12px;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,.12);
  filter:saturate(.85) brightness(1.1)}}
.card-body{{flex:1;min-width:0}}
.card-label{{font-size:15px;font-weight:900;color:#1a1a2e;margin-bottom:3px}}
.card-desc{{font-size:12px;color:#636e84;font-weight:500;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card-arrow{{font-size:18px;color:#c0c8d8;font-weight:900;flex-shrink:0;
  transition:color .18s,transform .18s}}
.class-card:hover .card-arrow{{color:#667eea;transform:translateX(3px)}}
{css_extra}
</style></head><body>
<div class="page-wrap">
  <div class="hero">
    <div class="hero-eye">Expert Annotation</div>
    <h1>CMR Scar Phenotyping<br>Expert Annotation Gallery</h1>
    <div class="hero-sub">
      {n_total or 264} reports &middot; 5-class &middot; Reader 1 &amp; Reader 2<br>
      6 excluded (non-contrast / vascular agent)<br>
      Each card: Reader 1 &nbsp;|&nbsp; Reader 2 &nbsp;|&nbsp; NLP &nbsp;|&nbsp;
      LLM-8B &nbsp;|&nbsp; FlanT5 &nbsp;&middot;&nbsp; &#10003;/&#10007; vs Reader 1
    </div>
  </div>
{acc_html}
{"".join(section_html)}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  index_expert_gallery.html")





# ─────────────────────────────────────────────────────────────────────────────
# Expert annotation Excel export
# ─────────────────────────────────────────────────────────────────────────────
def export_expert_excel(df_expert: pd.DataFrame, out_path: str) -> None:
    """
    Build expert_annotation_results.xlsx with 4 sheets:

    Sheet 1  Expert_Annotations   — one row per report:
               report_id | accession | procedure_date | r1_label | r1_comment
               r2_label | r2_comment | r1_r2_agree | nlp_class | nlp_vs_r1
               lvef | has_lge | lge_pattern | lge_location

    Sheet 2  Agreement_Summary    — inter-reader + NLP accuracy stats

    Sheet 3  Confusion_Matrix     - NLP vs Expert (Reader 1) confusion matrix

    Sheet 4  Class_Distribution   — per-class counts for R1, R2, NLP
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    VALID = {"ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"}
    CLASSES = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]

    df = df_expert.copy()

    # ── Derived columns ───────────────────────────────────────────────────────
    df["r1_r2_agree"] = df.apply(
        lambda r: ("AGREE"    if r["r1_label"] in VALID and r["r2_label"] in VALID
                                  and r["r1_label"] == r["r2_label"]
                   else "DISAGREE" if r["r1_label"] in VALID and r["r2_label"] in VALID
                   else "N/A"),
        axis=1
    )
    df["nlp_vs_r1"] = df.apply(
        lambda r: ("MATCH"    if r["r1_label"] in VALID and r["nlp_class"] in VALID
                                  and r["r1_label"] == r["nlp_class"]
                   else "MISMATCH" if r["r1_label"] in VALID and r["nlp_class"] in VALID
                   else "N/A"),
        axis=1
    )

    # ── Sheet 1: Expert_Annotations ───────────────────────────────────────────
    cols1 = ["report_id", "accession_number_real", "Procedure Date",
             "r1_label", "r1_comment", "r2_label", "r2_comment",
             "r1_r2_agree", "nlp_class", "nlp_vs_r1",
             "lvef", "has_lge", "lge_pattern", "lge_location"]
    # Only keep columns that exist
    cols1 = [c for c in cols1 if c in df.columns]
    df_out = df[cols1].copy()
    df_out.rename(columns={
        "accession_number_real": "accession",
        "Procedure Date":        "procedure_date",
    }, inplace=True)

    # ── Sheet 2: Agreement_Summary ────────────────────────────────────────────
    df_both   = df[df["r1_label"].isin(VALID) & df["r2_label"].isin(VALID)]
    df_r1valid = df[df["r1_label"].isin(VALID)]
    df_nlp_r1  = df[df["r1_label"].isin(VALID) & df["nlp_class"].isin(VALID)]

    ir_agree   = (df_both["r1_label"] == df_both["r2_label"]).sum()
    ir_total   = len(df_both)
    nlp_agree  = (df_nlp_r1["r1_label"] == df_nlp_r1["nlp_class"]).sum()
    nlp_total  = len(df_nlp_r1)

    rows_summ = [
        ["Metric", "Count", "Total", "Percentage", "Notes"],
        ["Total reports (annotated)", len(df_r1valid), "", "", "Reports with Reader 1 label"],
        ["Inter-reader agreements (R1=R2)", ir_agree, ir_total,
         f"{ir_agree/ir_total*100:.1f}%" if ir_total else "—",
         "RPT0100 is the only disagreement"],
        ["Inter-reader disagreements (R1≠R2)", ir_total - ir_agree, ir_total,
         f"{(ir_total-ir_agree)/ir_total*100:.1f}%" if ir_total else "—", ""],
        ["NLP matches Expert (R1)", nlp_agree, nlp_total,
         f"{nlp_agree/nlp_total*100:.1f}%" if nlp_total else "—", "NLP vs Reader 1"],
        ["NLP mismatches Expert (R1)", nlp_total - nlp_agree, nlp_total,
         f"{(nlp_total-nlp_agree)/nlp_total*100:.1f}%" if nlp_total else "—", ""],
        [],
        ["Per-class NLP accuracy (vs Reader 1)", "", "", "", ""],
    ]
    for cls in CLASSES:
        sub = df_r1valid[df_r1valid["r1_label"] == cls]
        if len(sub) == 0:
            continue
        nlp_sub = sub[sub["nlp_class"].isin(VALID)]
        n_match = (nlp_sub["nlp_class"] == cls).sum()
        rows_summ.append([
            f"  {cls}", n_match, len(sub),
            f"{n_match/len(sub)*100:.1f}%",
            f"errors → {dict(sub[sub['nlp_class']!=cls]['nlp_class'].value_counts().head(3))}"
        ])
    df_summ = pd.DataFrame(rows_summ[1:], columns=rows_summ[0])

    # ── Sheet 3: Confusion Matrix ─────────────────────────────────────────────
    conf_data = []
    for r in CLASSES:
        sub = df_r1valid[df_r1valid["r1_label"] == r]
        row = {"Expert vs NLP →": r}
        for c in CLASSES:
            row[c] = int((sub["nlp_class"] == c).sum())
        row["Total"] = len(sub)
        conf_data.append(row)
    df_conf = pd.DataFrame(conf_data)

    # Sheet 4: Class Distribution ───────────────────────────────────────────
    dist_data = []
    for cls in CLASSES:
        dist_data.append({
            "Class":        cls,
            "R1 (n)":       int((df["r1_label"] == cls).sum()),
            "R2 (n)":       int((df["r2_label"] == cls).sum()),
            "NLP (n)":      int((df["nlp_class"] == cls).sum()),
            "R1 (%)":       f"{(df['r1_label']==cls).sum()/len(df)*100:.1f}%",
            "R2 (%)":       f"{(df['r2_label']==cls).sum()/len(df)*100:.1f}%",
            "NLP (%)":      f"{(df['nlp_class']==cls).sum()/len(df)*100:.1f}%",
        })
    df_dist = pd.DataFrame(dist_data)

    # ── Write to Excel with openpyxl styling ─────────────────────────────────
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_out  .to_excel(writer, sheet_name="Expert_Annotations",  index=False)
        df_summ .to_excel(writer, sheet_name="Agreement_Summary",   index=False)
        df_conf .to_excel(writer, sheet_name="Confusion_Matrix",    index=False)
        df_dist .to_excel(writer, sheet_name="Class_Distribution",  index=False)

    # ── Apply styling ─────────────────────────────────────────────────────────
    wb = load_workbook(out_path)

    CLASS_FILLS = {
        "ischemic":      "FADBD8",  # light red
        "non-ischemic":  "D6EAF8",  # light blue
        "mixed":         "E8DAEF",  # light purple
        "no-scar":       "D5F5E3",  # light green
        "not-specified": "FDEBD0",  # light orange
        "AGREE":         "D5F5E3",
        "DISAGREE":      "FADBD8",
        "MATCH":         "D5F5E3",
        "MISMATCH":      "FADBD8",
        "N/A":           "F5F5F5",
    }
    CLASS_FONTS = {
        "ischemic":      "922B21",
        "non-ischemic":  "1A5276",
        "mixed":         "6C3483",
        "no-scar":       "1E8449",
        "not-specified": "784212",
        "AGREE":         "1E8449",
        "DISAGREE":      "922B21",
        "MATCH":         "1E8449",
        "MISMATCH":      "922B21",
        "N/A":           "888888",
    }

    def _style_ws(ws, header_color="2C3E50"):
        """Auto-width columns, freeze top row, style header."""
        thin = Side(style="thin", color="CCCCCC")
        bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
        # Header row
        for cell in ws[1]:
            cell.font      = Font(name="Arial", bold=True, size=9, color="FFFFFF")
            cell.fill      = PatternFill("solid", fgColor=header_color)
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border    = bdr
        ws.row_dimensions[1].height = 28
        ws.freeze_panes = "A2"
        # Data rows
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font      = Font(name="Arial", size=9)
                cell.alignment = Alignment(vertical="center", wrap_text=False)
                cell.border    = bdr
        # Auto-width
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col
            )
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(
                max(max_len + 2, 8), 40
            )

    # Style Expert_Annotations sheet
    ws1 = wb["Expert_Annotations"]
    _style_ws(ws1, "1A3A5C")
    # Color-code label and agreement columns
    label_cols = {}
    for i, cell in enumerate(ws1[1], 1):
        if cell.value in ("r1_label", "r2_label", "nlp_class",
                          "r1_r2_agree", "nlp_vs_r1"):
            label_cols[i] = cell.value
    for row in ws1.iter_rows(min_row=2):
        for cell in row:
            if cell.column in label_cols:
                val = str(cell.value or "")
                if val in CLASS_FILLS:
                    cell.fill = PatternFill("solid", fgColor=CLASS_FILLS[val])
                    cell.font = Font(name="Arial", size=9, bold=True,
                                     color=CLASS_FONTS.get(val, "000000"))

    # Style Agreement_Summary
    ws2 = wb["Agreement_Summary"]
    _style_ws(ws2, "4A1D6E")

    # Style Confusion Matrix — highlight diagonal
    ws3 = wb["Confusion_Matrix"]
    _style_ws(ws3, "1A5276")
    # Color diagonal cells (correct predictions)
    for i, cls in enumerate(CLASSES, 2):
        diag_col = i + 1  # col 1=label, cols 2-6=classes
        cell = ws3.cell(row=i, column=diag_col)
        if cell.value and int(cell.value) > 0:
            cell.fill = PatternFill("solid", fgColor="D5F5E3")
            cell.font = Font(name="Arial", size=9, bold=True, color="1E8449")

    # Style Class Distribution
    ws4 = wb["Class_Distribution"]
    _style_ws(ws4, "1E8449")
    for row in ws4.iter_rows(min_row=2):
        cls_cell = row[0]
        val = str(cls_cell.value or "")
        if val in CLASS_FILLS:
            for cell in row:
                cell.fill = PatternFill("solid", fgColor=CLASS_FILLS[val])

    wb.save(out_path)
    print(f"  Saved: {out_path}")
    print(f"    Sheet 1 Expert_Annotations: {len(df_out)} rows")
    print(f"    Sheet 2 Agreement_Summary:  {len(df_summ)} rows")
    print(f"    Sheet 3 Confusion_Matrix:   {len(df_conf)} rows")
    print(f"    Sheet 4 Class_Distribution: {len(df_dist)} rows")


# ── Top-level runner called from main() ───────────────────────────────────────
def _run_expert_galleries(args):
    kw = dict(redact_mrn=args.redact_mrn,
              redact_acc=args.redact_accession,
              redact_pd=args.redact_procedure_date)
    out_dir = BASE_DIR

    df = _expert_load(args.two_expert_xlsx)
    VALID = set(CLASSES)

    # ── Run NLP classifier on every expert report ─────────────────────────────
    # This MUST happen before accuracy computation, gallery generation, and Excel export
    print("  Running NLP classifier on expert reports...")
    df["nlp_class"] = df["report_text"].apply(
        lambda t: classify_5class(t) if t and str(t).strip() else ""
    )
    # Also extract LGE metadata now so it is available for both galleries and Excel
    print("  Extracting LGE metadata...")
    def _exp_enrich(text):
        if not text or not str(text).strip():
            return pd.Series({"lvef": None, "has_lge": False,
                              "lge_pattern": "", "lge_location": ""})
        p   = extract_lge_info(text)
        return pd.Series({
            "lvef":         extract_lvef(text),
            "has_lge":      p["has_lge"],
            "lge_pattern":  p["pattern"],
            "lge_location": p["location"],
        })
    enriched = df["report_text"].apply(_exp_enrich)
    df = pd.concat([df.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)

    df_valid    = df[df["r1_label"].isin(VALID)].copy()
    df_agree    = df_valid[df_valid["r2_label"].isin(VALID) &
                           (df_valid["r1_label"] == df_valid["r2_label"])].copy()
    df_mismatch = df_valid[df_valid["r2_label"].isin(VALID) &
                           (df_valid["r1_label"] != df_valid["r2_label"])].copy()
    print(f"  Inter-reader: {len(df_agree)} agreements, {len(df_mismatch)} mismatches "
          f"({len(df_agree)/max(1,len(df_agree)+len(df_mismatch))*100:.1f}% agreement)")

    # ── NLP accuracy vs expert (Reader 1 as ground truth) ─────────────────────
    nlp_stats = {}
    df_nlp_eval = df_valid.copy()
    total_correct = 0
    print("\n  ── NLP accuracy vs Expert (Reader 1) ──")
    for cls in CLASSES:
        sub = df_nlp_eval[df_nlp_eval["r1_label"] == cls]
        if len(sub) == 0:
            nlp_stats[cls] = (0, 0, 0.0)
            continue
        correct = (sub["nlp_class"] == cls).sum()
        total_correct += correct
        pct = correct / len(sub) * 100
        nlp_stats[cls] = (correct, len(sub), pct)
        print(f"    {cls:15s}: {correct:3d}/{len(sub):3d} = {pct:.0f}%")
    overall_pct = total_correct / len(df_nlp_eval) * 100 if len(df_nlp_eval) > 0 else 0
    print(f"    {'OVERALL':15s}: {total_correct:3d}/{len(df_nlp_eval):3d} = {overall_pct:.1f}%")

    idx_by_class, idx_agreement, idx_mismatch, idx_nlp = [], [], [], []

    print("\n  ── By Reader 1 class ──")
    for cls in CLASSES:
        sub = df_valid[df_valid["r1_label"] == cls].copy()
        if len(sub) == 0:
            continue
        fn = f"by_class_{cls.replace('-','_')}.html"
        _exp_write_gallery(sub,
            title=f"By Class — {_EXPERT_CLASS_DISPLAY[cls]}  (Reader 1, n={len(sub)})",
            subtitle=f"Reports labelled '{cls}' by Reader 1",
            out_path=os.path.join(out_dir, fn), **kw)
        idx_by_class.append((fn, _EXPERT_CLASS_DISPLAY[cls], len(sub), ""))

    print("\n  ── Expert agreements ──")
    fn_agree = "expert_agreements.html"
    _exp_write_gallery(df_agree,
        title=f"Expert Agreements — Reader 1 = Reader 2  (n={len(df_agree)})",
        subtitle="All cases where Reader 1 and Reader 2 assigned the same label",
        out_path=os.path.join(out_dir, fn_agree), **kw)
    idx_agreement.append((fn_agree, "Agreements", len(df_agree), "All cases where Reader 1 = Reader 2"))

    print("\n  ── Expert mismatches ──")
    fn_mm = "expert_mismatches.html"
    _exp_write_gallery(df_mismatch,
        title=f"Expert Mismatches — Reader 1 \u2260 Reader 2  (n={len(df_mismatch)})",
        subtitle=f"All {len(df_mismatch)} cases where Reader 1 and Reader 2 disagree",
        out_path=os.path.join(out_dir, fn_mm), **kw)
    idx_mismatch.append((fn_mm, "Mismatches", len(df_mismatch),
                         f"All {len(df_mismatch)} cases where Reader 1 \u2260 Reader 2"))

    print("\n  ── NLP on mismatches ──")
    df_nlp_mm = df_mismatch.copy().sort_values(["nlp_class", "report_id"], na_position="last")
    fn_nlp_mm = "nlp_on_disagreements.html"
    _exp_write_gallery(df_nlp_mm,
        title=f"NLP on Expert Mismatches  (n={len(df_nlp_mm)})",
        subtitle="Expert disagreements sorted by NLP label — shows which reader NLP agrees with",
        out_path=os.path.join(out_dir, fn_nlp_mm), **kw)
    idx_mismatch.append((fn_nlp_mm, "NLP on Mismatches", len(df_nlp_mm),
                         f"NLP result for each of the {len(df_nlp_mm)} expert disagreements"))

    print("\n  ── Agreements where NLP disagrees ──")
    df_agree_nlp = df_agree[
        df_agree["nlp_class"].isin(VALID) & (df_agree["nlp_class"] != df_agree["r1_label"])
    ].copy().sort_values(["r1_label", "nlp_class"])
    fn_agree_nlp = "agreements_nlp_mismatch.html"
    _exp_write_gallery(df_agree_nlp,
        title=f"Expert Agreements where NLP Disagrees  (n={len(df_agree_nlp)})",
        subtitle=f"{len(df_agree_nlp)} cases where R1=R2 but NLP assigns a different label",
        out_path=os.path.join(out_dir, fn_agree_nlp), **kw)
    idx_agreement.append((fn_agree_nlp, "NLP Disagreement", len(df_agree_nlp),
                           f"R1=R2 but NLP differs — {len(df_agree_nlp)} cases"))

    print("\n  ── By NLP class ──")
    for cls in CLASSES:
        sub = df[df["nlp_class"] == cls].copy()
        if len(sub) == 0:
            continue
        fn = f"nlp_{cls.replace('-','_')}.html"
        _exp_write_gallery(sub,
            title=f"NLP Results — {_EXPERT_CLASS_DISPLAY[cls]}  (n={len(sub)})",
            subtitle=f"Reports classified as '{cls}' by the NLP pipeline",
            out_path=os.path.join(out_dir, fn), **kw)
        idx_nlp.append((fn, _EXPERT_CLASS_DISPLAY[cls], len(sub), f"NLP class: {cls}"))

    print("\n  ── Expert index ──")
    _exp_write_index(os.path.join(out_dir, "index_expert_gallery.html"), [
        ("By Expert Label  (Reader 1)",              idx_by_class),
        ("Expert Agreements  (Reader 1 = Reader 2)", idx_agreement),
        ("Expert Disagreements  (Reader 1 \u2260 Reader 2)", idx_mismatch),
        ("By NLP Pipeline Class",                    idx_nlp),
    ],
    nlp_stats=nlp_stats,
    nlp_overall=overall_pct,
    n_agree=len(df_agree),
    n_total=len(df_nlp_eval),
    n_mismatch=len(df_mismatch))
    n_files = len(idx_by_class) + len(idx_agreement) + len(idx_mismatch) + len(idx_nlp) + 1
    print(f"\n  Expert galleries: {n_files} files written to {out_dir}")
    print(f"  NLP overall accuracy: {total_correct}/{len(df_nlp_eval)} = {overall_pct:.1f}%")

    # ── Export expert annotation Excel ────────────────────────────────────────
    # df_valid already has nlp_class + lge metadata from enrichment above
    print("\n  ── Expert Excel export ──")
    export_expert_excel(df_valid, EXPERT_OUT_XLSX)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
# NLP pipeline only (original behaviour):
#   python cardiac_mri_nlp_5class.py \
#     --redact_mrn --redact_accession --redact_procedure_date \
#     --gallery_n 60 --require_lge_for_gallery
#
# NLP pipeline + random gallery sampling:
#   python cardiac_mri_nlp_5class.py \
#     --redact_mrn --redact_accession --redact_procedure_date \
#     --gallery_n 60 --require_lge_for_gallery --random_gallery --random_seed 42
#
# NLP pipeline + expert annotation galleries:
#   python cardiac_mri_nlp_5class.py \
#     --redact_mrn --redact_accession --redact_procedure_date \
#     --gallery_n 60 --require_lge_for_gallery \
#     --expert_galleries \
#     --annotation_list data/Expert_Annotation_list.xlsx \
#     --expert_xlsx     data/CMR_Expert_Annotation_and_all_models.xlsx
#
# NLP pipeline + expert galleries (only Two_Expert file needed):
#   python cardiac_mri_nlp_5class.py \
#     --redact_mrn --redact_accession --redact_procedure_date \
#     --gallery_n 60 --require_lge_for_gallery \
#     --expert_galleries \
#     --two_expert_xlsx data/Two_Expert_Annotation_Comparision.xlsx
#
# All options combined (random + expert):
#   python cardiac_mri_nlp_5class.py \
#     --redact_all --gallery_n 60 --require_lge_for_gallery \
#     --random_gallery --random_seed 42 \
#     --expert_galleries \
#     --two_expert_xlsx data/Two_Expert_Annotation_Comparision.xlsx

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    spark = initialize_spark()
    print("=" * 70)
    print("CARDIAC MRI NLP — 5-CLASS PIPELINE")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Output:  {OUT_XLSX}")
    print("=" * 70)

    try:
        df_spark, df_raw = load_excel_to_spark(spark, EXCEL_PATH)
        df               = process_row_level(df_spark)
        export_excel(df, OUT_XLSX, df_raw=df_raw)

        print("\nGenerating NLP HTML galleries...")
        gallery_mode = "random" if args.random_gallery else "keyword-scored"
        print(f"  Gallery selection mode: {gallery_mode}")
        for cls, path in [
            ("ischemic",      GALLERY_ISC_HTML),
            ("non-ischemic",  GALLERY_NONISC_HTML),
            ("mixed",         GALLERY_MIXED_HTML),
            ("no-scar",       GALLERY_NOSCAR_HTML),
            ("not-specified", GALLERY_NOTSPEC_HTML),
        ]:
            sub = select_gallery_subset(
                df, cls, args.gallery_n, args.require_lge_for_gallery,
                random_sample=args.random_gallery, random_seed=args.random_seed
            )
            write_gallery(sub, f"NER Gallery — {cls.upper()} (n={len(sub)})",
                          path, args.redact_mrn, args.redact_accession, args.redact_procedure_date)

        # Compute per-class report counts for the index page badges
        cls_counts_map = df["scar_type"].value_counts().to_dict()

        write_index(INDEX_HTML, [
            (os.path.basename(GALLERY_ISC_HTML),    "Ischemic"),
            (os.path.basename(GALLERY_NONISC_HTML), "Non-ischemic"),
            (os.path.basename(GALLERY_MIXED_HTML),  "Mixed"),
            (os.path.basename(GALLERY_NOSCAR_HTML), "No-scar"),
            (os.path.basename(GALLERY_NOTSPEC_HTML),"Not-specified"),
        ], counts=cls_counts_map)

        print(f"\n✅  NLP pipeline done")
        print(f"   📊 {OUT_XLSX}")
        print(f"   🌐 {INDEX_HTML}")

        # ── Expert annotation galleries (optional) ────────────────────────────
        if args.expert_galleries:
            print("\n" + "=" * 70)
            print("EXPERT ANNOTATION GALLERIES")
            print("=" * 70)
            _run_expert_galleries(args)

    finally:
        try: spark.stop()
        except: pass


if __name__ == "__main__":
    main()

"""The paper's rule-based NLP (paper_scripts/cardiac_mri_nlp_5class.py), imported without its PySpark batch dependency."""

import os
import sys
from typing import Optional
from unittest import mock

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_scripts")


def _import_nlp():
    # The module imports pyspark for its full-cohort pipeline; the classifier itself is plain Python.
    names = ["pyspark", "pyspark.sql", "pyspark.sql.functions", "pyspark.sql.types"]
    stubs = {}
    try:
        import pyspark  # noqa: F401
    except ImportError:
        stubs = {n: mock.MagicMock() for n in names}
    sys.path.insert(0, SRC_DIR)
    try:
        sys.modules.update(stubs)
        import cardiac_mri_nlp_5class as nlp
    finally:
        sys.path.remove(SRC_DIR)
        for name in stubs:
            del sys.modules[name]
    return nlp


_nlp = _import_nlp()
CLASSES = tuple(_nlp.CLASSES)


def classify_scar(text) -> Optional[str]:
    """One of CLASSES, or None when the rules exclude the report (cardiac device, no contrast or no LGE section)."""
    text = str(text or "")
    if not text.strip() or _nlp._is_device(text) or _nlp._is_no_contrast(text):
        return None
    return _nlp.classify_5class(text)


def extract_lvef(text) -> Optional[float]:
    """Left ventricular ejection fraction in percent, or None if the report gives none in 10-100%."""
    return _nlp.extract_lvef(str(text or ""))


def explain(text) -> dict:
    """The class plus why: the exclusion reason, or the pattern the rules matched."""
    text = str(text or "")
    if not text.strip():
        return {"label": None, "reason": "empty report"}
    if _nlp._is_device(text):
        return {"label": None, "reason": "cardiac device: excluded by the rules"}
    if _nlp._is_no_contrast(text):
        return {"label": None, "reason": "no contrast or no LGE section: excluded by the rules"}
    p = _nlp._extract_patterns(text)
    return {"label": _nlp.classify_5class(text), "reason": p["pattern"],
            "ischemic_evidence": p["has_ischemic"], "nonischemic_evidence": p["has_nonischemic"], "scar_mentioned": p["has_scar"]}

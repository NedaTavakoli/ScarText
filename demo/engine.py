"""Rule-based NLP wrapper and HTML rendering for the demo.

Classification comes unchanged from paper_scripts/cardiac_mri_nlp_5class.py. This file
only explains that result: it adds the exclusion reason, the trigger highlights,
and the result cards.
"""

import html
import os
import re
import sys
from unittest import mock

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_scripts")


def _import_nlp():
    # The module imports pyspark at load time for its batch pipeline. The
    # classifier itself is plain Python, so stub pyspark when it is not installed.
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


nlp = _import_nlp()

CLASSES = nlp.CLASSES
CLASS_COLORS = nlp._EXPERT_CLASS_COLORS
CLASS_DISPLAY = {**nlp._EXPERT_CLASS_DISPLAY, "no-scar": "No scar", "not-specified": "Unspecified"}   # names as in the paper; the label stays not-specified
NEUTRAL_COLORS = nlp._EXPERT_NONE_COLOR

CLASS_MEANING = {
    "ischemic": "Scar in a coronary pattern, such as subendocardial or infarct.",
    "non-ischemic": "Scar in a non-coronary pattern, such as mid-wall or subepicardial.",
    "mixed": "Both ischemic and non-ischemic scar patterns are described.",
    "no-scar": "The report describes no myocardial scar or LGE.",
    "not-specified": "Scar is described, but its type is not stated.",
}

# Matching order sets precedence: earlier rules claim the text first.
_HL_RULES = [
    ("pattern", r"\bsub[-\s]?endocardial\b|\btransmural\b"
                r"|\bmid[-\s]?wall\b|\bmid[-\s]?myocardial\b|\bmesocardi(?:al|um)\b"
                r"|\bsub[-\s]?epicardial\b"
                r"|\bepicardi(?:al|um)\b(?!\s+(?:fat|adipose|coronary|vessel|arter|disease|lead|electrode|pacing))"
                r"|\bpatchy\b|\blinear\b|\b(?:rv\s+)?insertion\s+points?\b|\brv\s+insertion\b"
                r"|\bnon[-\s]?vascular\b|\blate\s+gadolinium\s+enhancement\b"
                r"|\bdelayed\s+(?:\w+\s+){0,2}enhancement\b|\bhyperenhancement\b|\bLGE\b"),
    ("disease", r"\binfarct(?:ion|s)?\b|\bscar(?:ring|s)?\b|\bfibrosis\b|\bSTEMI\b|\bNSTEMI\b"
                r"|\bnon[-\s]?ischemic\b|\bischemic\s+cardiomyopathy\b|\bischemic\b"
                r"|\bcoronary\s+artery\s+disease\b|\bCAD\b|\bcoronary\s+distribution\b"
                r"|\bembol\w+\b|\bmyocarditis\b|\bsarcoid(?:osis)?\b|\bamyloid(?:osis)?\b"
                r"|\bfabry\b|\bhypertrophic\s+cardiomyopathy\b|\bdilated\s+cardiomyopathy\b"
                r"|\bHCM\b|\bDCM\b|\binfiltrative\b|\binflammatory\b"),
    ("anatomy", r"\b(?:LAD|RCA|LCx)\s+territory\b|\bcoronary\s+territory\b"
                r"|\binferolateral\b|\banterolateral\b|\binferoseptal\b|\banteroseptal\b"
                r"|\binferoseptum\b|\banteroseptum\b|\binferior\b|\banterior\b|\blateral\b"
                r"|\bseptal\b|\bseptum\b|\bapical\b|\bapex\b|\bbasal\b|\bcircumferential\b"),
]

_HL_LABELS = {
    "anatomy": ("Anatomy", "#FFF176", "#3B3000"),
    "pattern": ("LGE pattern", "#90CAF9", "#0B2A45"),
    "disease": ("Disease", "#FFB74D", "#402300"),
    "negative": ("No-LGE statement", "#C8E6C9", "#123A18"),
}


def exclusion_reason(text):
    """Why the rules return None for this text. None means the text is classified."""
    if not str(text or "").strip():
        return "empty"
    if nlp._is_device(text):
        return "device"
    if nlp._is_no_contrast(text):
        return "no_lge"
    return None


_EXCLUSION_MESSAGES = {
    "device": "The report mentions a cardiac device such as a pacemaker. The rule-based pipeline "
              "excludes these reports, so no class is assigned.",
    "no_lge": "No LGE section was found, or the report says no contrast was given. The rule-based "
              "pipeline needs a term such as \"late gadolinium enhancement\" or \"delayed enhancement\", "
              "so no class is assigned.",
}


def run_nlp(text):
    """Return the rule-based result as a dict."""
    reason = exclusion_reason(text)
    if reason:
        return {"label": None, "reason": reason, "pattern": "", "flags": {}}
    label = nlp.classify_5class(text)
    p = nlp._extract_patterns(text)
    return {
        "label": label,
        "reason": None,
        "pattern": p["pattern"],
        "flags": {k: p[k] for k in ("has_ischemic", "has_nonischemic", "has_scar")},
    }


def findings_mask(text):
    """Mark each character True if the pattern rules read it.

    The rules drop History and Technique sections and return only the kept text.
    Section headers start at line starts, so whole lines are matched back in order.
    """
    kept = nlp._strip_non_findings_sections(text)
    if kept == text:
        return [True] * len(text)
    mask = [False] * len(text)
    pos, start = 0, 0
    for line in text.splitlines(keepends=True):
        content = line.strip()
        while pos < len(kept) and kept[pos].isspace():
            pos += 1
        if content and kept.startswith(content, pos):
            mask[start:start + len(line)] = [True] * len(line)
            pos += len(content)
        start += len(line)
    # If the lines cannot all be matched, show everything rather than dim wrongly.
    return mask if not kept[pos:].strip() else [True] * len(text)


def _negative_spans(text):
    lowered = text.lower()
    spans = []
    for phrase in nlp._NO_LGE_EXACT:
        spans += [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), lowered)]
    for rx in nlp._NO_LGE_REGEX:
        spans += [m.span() for m in rx.finditer(lowered)]
    # Longest first, so the whole statement wins over a shorter phrase inside it.
    return sorted(spans, key=lambda s: s[0] - s[1])


def highlight_spans(text):
    """Return sorted (start, end, label) spans for the text the rules read."""
    mask = findings_mask(text)
    taken = [False] * len(text)
    spans = []

    def claim(a, b, label):
        if any(taken[a:b]) or not all(mask[a:b]):
            return
        taken[a:b] = [True] * (b - a)
        spans.append((a, b, label))

    for a, b in _negative_spans(text):
        claim(a, b, "negative")
    for label, pattern in _HL_RULES:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            claim(m.start(), m.end(), label)
    return sorted(spans)


def _mark(fragment, label):
    _, bg, fg = _HL_LABELS[label]
    return f'<mark style="background:{bg};color:{fg};padding:0 2px;border-radius:3px">{html.escape(fragment)}</mark>'


def render_report(text):
    """Report text with highlights. Sections the pattern rules skip are dimmed."""
    if not str(text or "").strip():
        return ""
    mask = findings_mask(text)
    spans = highlight_spans(text)
    starts = {a: (b, label) for a, b, label in spans}
    out, i = [], 0
    while i < len(text):
        if i in starts:
            b, label = starts[i]
            out.append(_mark(text[i:b], label))
            i = b
            continue
        j = i + 1
        while j < len(text) and j not in starts and mask[j] == mask[i]:
            j += 1
        fragment = html.escape(text[i:j])
        out.append(fragment if mask[i] else f'<span style="color:#8A8A8A">{fragment}</span>')
        i = j
    legend = "".join(
        f'<span style="background:{bg};color:{fg};padding:1px 8px;border-radius:3px;margin-right:6px;'
        f'display:inline-block;margin-bottom:4px">{name}</span>'
        for name, bg, fg in _HL_LABELS.values()
    ) + '<span style="color:#8A8A8A;display:inline-block">Grey text: section the pattern rules skip</span>'
    return (
        f'<div style="font-size:13px;margin-bottom:8px">{legend}</div>'
        '<div style="background:#FFFFFF;color:#1F1F1F;border:1px solid #D0D0D0;border-radius:8px;padding:14px;'
        'white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.6">'
        f'{"".join(out)}</div>'
    )


def _card(source, title, body, colors, footnote=""):
    fg, bg = colors
    note = f'<div style="color:{fg} !important;font-size:13px;margin-top:6px;opacity:.85">{html.escape(footnote)}</div>' if footnote else ""
    return (
        f'<div style="background:{bg};color:{fg};border-left:6px solid {fg};border-radius:8px;padding:14px 16px;height:100%;box-sizing:border-box">'
        f'<div style="color:{fg} !important;font-size:12px;letter-spacing:.04em;text-transform:uppercase;font-weight:600">{html.escape(source)}</div>'
        f'<div style="color:{fg} !important;font-size:26px;font-weight:700;margin:2px 0 4px">{html.escape(title)}</div>'
        f'<div style="color:{fg} !important;font-size:14px">{html.escape(body)}</div>{note}</div>'
    )


def nlp_card(result):
    source = "Rule-based NLP"
    label = result["label"]
    if result["reason"] == "empty":
        return _card(source, "No report yet", "Paste a report or pick a case.", NEUTRAL_COLORS)
    if label is None:
        return _card(source, "Not classified", _EXCLUSION_MESSAGES[result["reason"]], NEUTRAL_COLORS)
    return _card(source, CLASS_DISPLAY[label], CLASS_MEANING[label], CLASS_COLORS[label],
                 f"Rule match: {result['pattern']}" if result["pattern"] else "")


# Online (Hugging Face Spaces) the LLM runs on a shared GPU rather than a local Ollama server.
_ONLINE = bool(os.environ.get("SPACE_ID")) or os.environ.get("CMR_DEMO_PUBLIC") == "1"
WAITING_TEXT = ("Running Qwen3-32B on a shared GPU. The first report after a quiet period can take up to 2 minutes."
                if _ONLINE else "Sending the report to the local Ollama server.")


def llm_card(llm, nlp_label=None, source="Zero-shot LLM", agree_note="In the study, 97.8% of such reports were correct."):
    """Card for an llm_client.LLMResult. The 97.8% is the paper's zero-shot Qwen3-32B agreement result."""
    if llm is None:
        return _card(source, "Waiting", WAITING_TEXT, NEUTRAL_COLORS)
    if llm.label is None:
        return _card(source, "NLP only", llm.message, NEUTRAL_COLORS)
    # The paper's agreement analysis: where the rules and Qwen3-32B agreed, 97.8% of expert reports were correct.
    if nlp_label is None:
        agreement = "The rules assigned no class to this report: send it for expert review."
    elif nlp_label == llm.label:
        agreement = f"Agrees with the rules. {agree_note}".strip()
    else:
        agreement = "Differs from the rules: send this report for expert review."
    return _card(f"{source}: {llm.model}", CLASS_DISPLAY[llm.label], CLASS_MEANING[llm.label],
                 CLASS_COLORS[llm.label], agreement)

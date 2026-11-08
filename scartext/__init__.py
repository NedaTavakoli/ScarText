"""ScarText: classify myocardial scar patterns from free-text cardiac MRI reports.

    from scartext import classify_scar, extract_lvef, classify_llm, compare, analyze_csv

- classify_scar(text): the paper's rule-based NLP (no GPU, deterministic).
- extract_lvef(text): LVEF in percent, if the report states it.
- classify_llm(text, model="qwen3:32b", examples=None): a local open-weight LLM through Ollama, with the
  paper's zero-shot prompt and settings, or few-shot when labeled examples are given.
- compare(rule_label, llm_label): the paper's agreement check (agree: accept; disagree: expert review).
- analyze_csv(path, text_column, ...): classify a column of reports and add the results as columns.
- show_report(text): the report in color in a notebook, with the terms the rules use highlighted.

The rule and prompt code is the paper's code in src/, unchanged. Research use only. Not for clinical use.
"""

from .rules import CLASSES, classify_scar, extract_lvef, explain
from .llm import classify_llm, ollama_available
from .workflow import compare, analyze_csv
from .display import show_report, report_html

__version__ = "0.1.0"
__all__ = ["CLASSES", "classify_scar", "extract_lvef", "explain", "classify_llm", "ollama_available",
           "compare", "analyze_csv", "show_report", "report_html"]

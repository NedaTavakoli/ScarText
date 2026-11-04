"""Colored view of a report, as in the demo: the rule class card plus the report with the terms the rules use highlighted
(anatomy yellow, LGE pattern blue, disease orange, no-LGE statements green; grey: sections the pattern rules skip)."""

import importlib.util
import os

_engine = None


def _load_engine():
    # The demo's engine holds the highlighting rules; load it from demo/ so both use the same code.
    global _engine
    if _engine is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "engine.py")
        spec = importlib.util.spec_from_file_location("cmrreportanalyzer_demo_engine", path)
        _engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_engine)
    return _engine


def report_html(text):
    """HTML: the rule-based class card above the highlighted report."""
    e = _load_engine()
    card = e.nlp_card(e.run_nlp(text)).replace("height:100%", "height:auto")   # the demo card fills a grid cell
    return f'<div style="max-width:460px;margin-bottom:10px">{card}</div>{e.render_report(text)}'


def show_report(text):
    """Display the colored report in a notebook (Jupyter or Colab); returns the HTML string outside a notebook."""
    html = report_html(text)
    try:
        from IPython.display import HTML
        return HTML(html)
    except ImportError:
        return html

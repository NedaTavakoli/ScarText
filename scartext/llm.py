"""Zero-shot and few-shot classification with a local LLM through Ollama, as in the paper.

Prompts come from the paper's scripts in paper_scripts/ (zero-shot: cardiac_mri_llm_5class.py; few-shot:
cardiac_mri_llm_fewshot_all.py). Settings: temperature 0, seed 42, 8,192-token context, at most 512
output tokens, reasoning off. An answer that is not one of the five classes is returned as None (missing).
Report text is sent only to the Ollama server at `host` (default: this computer).
"""

import ast
import contextlib
import io
import json
import os
import re
import sys
import urllib.request

from .rules import CLASSES, SRC_DIR

DEFAULT_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
MAX_CHARS = 5000
OPTIONS = {"temperature": 0.0, "seed": 42, "num_predict": 512, "num_ctx": 8192}


def _zero_shot_prompt():
    tree = ast.parse(open(os.path.join(SRC_DIR, "cardiac_mri_llm_5class.py"), encoding="utf-8").read())
    found = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ("SYSTEM_PROMPT", "USER_TEMPLATE")}
    return found["SYSTEM_PROMPT"], found["USER_TEMPLATE"]


_fs = None


def _fewshot_module():
    global _fs
    if _fs is None:
        sys.path.insert(0, SRC_DIR)
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # the script prints setup notes
                import cardiac_mri_llm_fewshot_all as fs
        finally:
            sys.path.remove(SRC_DIR)
        _fs = fs
    return _fs


def parse_answer(raw):
    """The paper's parser: the JSON 'classification' field, with a regex fallback; None if not a valid class."""
    raw = re.sub(r"^```(?:json)?\s*", "", str(raw).strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        cls = str(json.loads(raw).get("classification", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r'"classification"\s*:\s*"(ischemic|non-ischemic|mixed|no-scar|not-specified)"', raw, re.I)
        cls = m.group(1).lower() if m else ""
    return cls if cls in CLASSES else None


def _post(host, path, payload, timeout):
    req = urllib.request.Request(host.rstrip("/") + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ollama_available(host=DEFAULT_HOST):
    """The model names the Ollama server offers, or [] if no server answers."""
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except (OSError, ValueError):
        return []


def _settings(model):
    """gpt-oss reasons before it answers, so it runs at its lowest reasoning level with a 4,096-token output cap,
    as in the paper; the other models are asked to answer directly."""
    if model.startswith("gpt-oss"):
        return "low", dict(OPTIONS, num_predict=4096)
    return False, OPTIONS


def classify_llm(text, model="qwen3:32b", examples=None, host=DEFAULT_HOST, timeout=300):
    """Classify one report with a local LLM.

    examples: None for zero-shot, or a pandas DataFrame of labeled reports with columns
    `report_text` and `gold_label` for few-shot; 5 examples are retrieved per report by TF-IDF
    similarity, one per class first, as in the paper.
    Returns one of CLASSES, or None when the answer could not be read as a class.
    """
    if examples is None:
        system, user_template = _zero_shot_prompt()
        user = user_template.format(report_text=str(text)[:MAX_CHARS])
    else:
        fs = _fewshot_module()
        pool = examples.reset_index(drop=True).copy()
        if "report_id" not in pool:
            pool["report_id"] = [f"EX{i}" for i in range(len(pool))]
        key = id(examples)
        if getattr(classify_llm, "_key", None) != key:
            with contextlib.redirect_stdout(io.StringIO()):
                classify_llm._retriever = fs.FewShotRetriever(pool, knn_pool=15)
            classify_llm._key = key
        system, user = fs.SYSTEM_PROMPT, fs.build_few_shot_prompt(text, classify_llm._retriever.retrieve(text, 5))
    think, options = _settings(model)
    payload = {"model": model, "stream": False, "think": think, "options": options,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    return parse_answer(_post(host, "/api/chat", payload, timeout)["message"]["content"])

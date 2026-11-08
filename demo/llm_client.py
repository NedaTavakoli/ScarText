"""Zero-shot LLM slot, as in the paper: a locally served open-weight model called through Ollama.

The prompt is the paper's zero-shot prompt (SYSTEM_PROMPT and USER_TEMPLATE in
src/cardiac_mri_llm_5class.py), read from that file so the two cannot drift. Settings
match the paper: temperature 0, seed 42, 8,192-token context, at most 512 output tokens,
and reasoning ("thinking") off. The model's own answer is used, with no keyword rule.

The default URL is a local Ollama server, or an SSH tunnel to one on the GPU server.
Set CMR_LLM_URL to change it, or to an empty value to turn the LLM off, and
CMR_LLM_MODEL to choose the model (default qwen3:32b, the paper's best zero-shot model).
Any failure returns an LLMResult with no label, and the app then shows the NLP result alone.

Few-shot (opt-in, CMR_DEMO_FEWSHOT=1) uses the paper's few-shot code (src/cardiac_mri_llm_fewshot_all.py):
for each report, k=5 examples are retrieved from the 198 expert-labeled training reports
(Results/finetune_llm/data/A_train_expert198.jsonl) and added to the prompt. These are real study reports:
they are sent only to the Ollama server and are never shown on screen.
"""

import ast
import contextlib
import io
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:32b"
CLASSES = ("ischemic", "non-ischemic", "mixed", "no-scar", "not-specified")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
HEALTH_TIMEOUT_S = 3
CLASSIFY_TIMEOUT_S = 300
MAX_CHARS = 5000   # the paper's report truncation
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_SOURCE = os.path.join(PROJECT_DIR, "src", "cardiac_mri_llm_5class.py")
FEWSHOT_POOL = os.path.join(PROJECT_DIR, "Results", "finetune_llm", "data", "A_train_expert198.jsonl")
FEWSHOT_K = 5


def load_prompt(path=PROMPT_SOURCE):
    """SYSTEM_PROMPT and USER_TEMPLATE from the paper's script, without importing it."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("SYSTEM_PROMPT", "USER_TEMPLATE"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["SYSTEM_PROMPT"], found["USER_TEMPLATE"]


def parse_answer(raw):
    """The paper's parser: JSON 'classification', with the same regex fallback. None if no valid class."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        cls = str(json.loads(raw).get("classification", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r'"classification"\s*:\s*"(ischemic|non-ischemic|mixed|no-scar|not-specified)"', raw, re.I)
        cls = m.group(1).lower() if m else ""
    return cls if cls in CLASSES else None


@dataclass
class LLMResult:
    label: Optional[str]
    message: str = ""
    model: str = ""


class NoBackend:
    configured = False

    def __init__(self, message="The LLM is turned off. Showing the rule-based result only."):
        self.message = message

    def classify(self, text):
        return LLMResult(None, self.message)


class OllamaBackend:
    """Zero-shot classification through Ollama's /api/chat, normally on the GPU server through an SSH tunnel."""

    configured = True

    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model
        self.system_prompt, self.user_template = load_prompt()

    def _request(self, path, payload, timeout):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    def _messages(self, text):
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_template.format(report_text=str(text)[:MAX_CHARS])}]

    def classify(self, text):
        # Report text is never included in these messages or logged.
        try:
            self._request("/api/tags", None, HEALTH_TIMEOUT_S)
        except (OSError, ValueError):
            return LLMResult(None, "No Ollama server is connected. Showing the rule-based result only.")
        payload = {"model": self.model, "stream": False, "think": False, "messages": self._messages(text),
                   "options": {"temperature": 0.0, "seed": 42, "num_predict": 512, "num_ctx": 8192}}
        try:
            raw = self._request("/api/chat", payload, CLASSIFY_TIMEOUT_S)["message"]["content"]
        except (OSError, ValueError, KeyError, TypeError):
            return LLMResult(None, f"{self.model} did not answer. Is it pulled on the Ollama server? Showing the rule-based result only.")
        label = parse_answer(raw)
        if label is None:
            return LLMResult(None, "The model's answer was not one of the five classes. In the paper this counts as missing.")
        return LLMResult(label, model=self.model)


class FewShotOllamaBackend(OllamaBackend):
    """The paper's few-shot prompting: k=5 TF-IDF-retrieved, class-balanced examples from the expert training reports."""

    def __init__(self, url, model, pool_path=FEWSHOT_POOL):
        self.url, self.model = url.rstrip("/"), model
        import sys
        import pandas as pd
        sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
        with contextlib.redirect_stdout(io.StringIO()):   # the module prints setup notes
            import cardiac_mri_llm_fewshot_all as fs
            rows = [json.loads(line) for line in open(pool_path, encoding="utf-8")]
            pool = pd.DataFrame({"report_id": [r["report_id"] for r in rows], "report_text": [r["text"] for r in rows],
                                 "gold_label": [r["label"] for r in rows]})
            self.retriever = fs.FewShotRetriever(pool, knn_pool=15)
        self.fs = fs

    def _messages(self, text):
        examples = self.retriever.retrieve(text, FEWSHOT_K)
        return [{"role": "system", "content": self.fs.SYSTEM_PROMPT},
                {"role": "user", "content": self.fs.build_few_shot_prompt(text, examples)}]


def _check_url(env):
    """The URL to send report text to, or a NoBackend explaining why not."""
    url = env.get("CMR_LLM_URL", DEFAULT_URL).strip()
    if not url:
        return NoBackend()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return NoBackend("CMR_LLM_URL is not a valid http(s) URL. Showing the rule-based result only.")
    # Report text goes to this URL, so only a local server or tunnel is allowed by default.
    if parsed.hostname not in LOCAL_HOSTS and env.get("CMR_LLM_ALLOW_REMOTE") != "1":
        return NoBackend("CMR_LLM_URL is not a local address, so no text was sent. Use an SSH tunnel.")
    return url


def get_fewshot_backend(env=os.environ):
    if env.get("CMR_DEMO_FEWSHOT") != "1":
        return NoBackend("Few-shot is off. Set CMR_DEMO_FEWSHOT=1 to turn it on.")
    url = _check_url(env)
    if isinstance(url, NoBackend):
        return url
    try:
        return FewShotOllamaBackend(url, env.get("CMR_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
                                    env.get("CMR_FEWSHOT_POOL", FEWSHOT_POOL))
    except OSError:
        return NoBackend("The few-shot example pool was not found. Few-shot is off.")


def get_backend(env=os.environ):
    url = _check_url(env)
    if isinstance(url, NoBackend):
        return url
    return OllamaBackend(url, env.get("CMR_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL)

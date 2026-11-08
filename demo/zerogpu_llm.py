"""Zero-shot Qwen3-32B for the online demo on Hugging Face ZeroGPU.

Same prompt and parser as the paper (paper_scripts/cardiac_mri_llm_5class.py, read by llm_client.load_prompt), greedy decoding
(temperature 0), reasoning off, at most 512 new tokens, report truncated to 5,000 characters. The paper ran
Qwen3-32B through Ollama (4-bit GGUF); here it runs through Transformers with the official FP8 weights, so answers
can differ slightly from the paper's runs. Used only on the Space (online mode); the local demo uses Ollama.
"""

import os

import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_client

MODEL_ID = os.environ.get("CMR_SPACE_MODEL", "Qwen/Qwen3-32B-FP8")
LABEL = "Qwen3-32B (ZeroGPU)"

_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto").to("cuda")   # ZeroGPU attaches a GPU per call
_system, _user_template = llm_client.load_prompt()


_warm = False


def _gpu_seconds(prompt):
    """The first call after a restart also moves the weights onto the GPU; later calls take a few seconds. Requests
    are kept short because ZeroGPU refuses a request above a visitor's remaining daily quota (a few minutes on a
    free account), and it counts more than the seconds requested."""
    return 20 if _warm else 45


@spaces.GPU(duration=_gpu_seconds)
def _generate(prompt):
    ids = _tok(prompt, return_tensors="pt").to("cuda")
    out = _model.generate(**ids, max_new_tokens=512, do_sample=False)
    return _tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


class ZeroGPUBackend:
    configured = True

    def classify(self, text):
        messages = [{"role": "system", "content": _system},
                    {"role": "user", "content": _user_template.format(report_text=str(text)[:llm_client.MAX_CHARS])}]
        prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        global _warm
        try:
            raw = _generate(prompt)
            _warm = True
        except Exception as e:   # quota reached, GPU busy, or a timeout: show the rules alone
            _warm = False         # a new GPU may have been assigned, so the next call gets the longer request
            print("ZeroGPU call failed:", type(e).__name__, str(e)[:500])
            if "quota" in str(e).lower():
                return llm_client.LLMResult(None, "Your free GPU time on Hugging Face is used up for today, so only the "
                                                  "rules ran. It resets daily; signing in to Hugging Face gives more. "
                                                  "The notebook runs the LLM on your own computer.")
            detail = str(e).strip()[:200] or type(e).__name__
            return llm_client.LLMResult(None, f"The online LLM is not available right now: {detail} Try again in a minute.")
        label = llm_client.parse_answer(raw)
        if label is None:
            return llm_client.LLMResult(None, "The model's answer was not one of the five classes. In the paper this counts as missing.")
        return llm_client.LLMResult(label, model=LABEL)

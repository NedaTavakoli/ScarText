"""Zero-shot Qwen3-32B for the online demo on Hugging Face ZeroGPU.

Same prompt and parser as the paper (src/cardiac_mri_llm_5class.py, read by llm_client.load_prompt), greedy decoding
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


@spaces.GPU(duration=60)
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
        try:
            raw = _generate(prompt)
        except Exception as e:   # quota reached, GPU busy, or a timeout: show the rules alone
            return llm_client.LLMResult(None, f"The online LLM is not available right now ({type(e).__name__}). "
                                              "Signed-in Hugging Face users get more free GPU time.")
        label = llm_client.parse_answer(raw)
        if label is None:
            return llm_client.LLMResult(None, "The model's answer was not one of the five classes. In the paper this counts as missing.")
        return llm_client.LLMResult(label, model=LABEL)

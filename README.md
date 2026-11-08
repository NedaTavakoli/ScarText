# ScarText

**Classify myocardial scar patterns from free-text cardiac MRI reports** — ischemic, non-ischemic, mixed, no-scar or not-specified — with a rule-based NLP pipeline and open-weight LLMs that run on your own computer.

[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/tavakolineda/ScarText) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NedaTavakoli/ScarText/blob/main/notebooks/quickstart.ipynb)

Code for the paper **"ScarText: Can Large Language Models Match Expert Rules in Classifying Myocardial Scar Patterns from Cardiac MRI Reports?"** (Tavakoli N, Rahsepar AA, Ozturk O, Bagci U, Lee DC, Kim D; under review).

**Research use only. Not for clinical use.**

## Demo

[![Demo: rule-based NLP, zero-shot and few-shot Qwen3-32B on synthetic reports](demo/demo_preview.gif)](demo/CMR_scar_demo_synthetic.mp4)

Each report is shown with three results side by side: the rule-based NLP, zero-shot Qwen3-32B and few-shot Qwen3-32B. Where an LLM agrees with the rules the report is accepted; where it disagrees, it is flagged for expert review. All reports shown are synthetic. [Watch the full video (36 s)](demo/CMR_scar_demo_synthetic.mp4).

**Try it online:** the [live demo](https://huggingface.co/spaces/tavakolineda/ScarText) classifies any report you paste or upload (.txt or .docx) with the rule-based NLP and highlights the terms it uses. It runs the rules only; the LLMs run on your own computer (below). Do not upload identifiable patient information.

## Quick start

**In the browser:** open the [quickstart notebook in Colab](https://colab.research.google.com/github/NedaTavakoli/ScarText/blob/main/notebooks/quickstart.ipynb).

**On your computer:**

```bash
git clone https://github.com/NedaTavakoli/ScarText.git
cd ScarText
pip install -r notebooks/requirements-notebook.txt
```

```python
from scartext import classify_scar, extract_lvef, classify_llm, compare, analyze_csv

text = open("examples/synthetic_reports/mixed.txt").read()
classify_scar(text)                       # 'mixed'   (rule-based NLP, no GPU)
extract_lvef(text)                        # 32.0
llm = classify_llm(text, model="llama3.1:8b")   # needs Ollama, see below
compare(classify_scar(text), llm)         # 'agree' or 'review'
analyze_csv("reports.csv", "report_text", out_path="classified.csv")   # a whole file
```

## Using the LLMs: no data leave your computer

The LLMs are **public open-weight models used without training**, so there is no private checkpoint: you download the same models the paper used and run them locally with [Ollama](https://ollama.com).

| Model | Download | Paper accuracy (zero-shot, 264 reports) | Hardware |
|---|---|---|---|
| Qwen3-32B | `ollama pull qwen3:32b` (~20 GB) | 89.8% | GPU with ≥24 GB, or a Mac with ≥32 GB |
| Llama-3.1-70B | `ollama pull llama3.1:70b` (~43 GB) | 88.3% | ≥48 GB |
| Llama-3.1-8B | `ollama pull llama3.1:8b` (~5 GB) | 79.2% | a laptop, or free Colab |

`classify_llm` talks to Ollama at `http://127.0.0.1:11434` on your own machine (change it with `host=` or `OLLAMA_HOST_URL`). The model weights are not redistributed here and keep their own licenses (Llama 3.1 Community License; Apache 2.0 for Qwen3).

**Few-shot:** pass your own labeled reports, `classify_llm(text, model=..., examples=df)` with columns `report_text` and `gold_label`; five similar examples, one per class first, are put in each prompt, as in the paper. The paper's expert-labeled reports are not distributed.

## Main results

On 264 expert-labeled reports (two blinded readers, κ = 0.995):

| Method | Accuracy |
|---|---|
| Rule-based NLP | 90.2% (95% CI 86.0–93.2); all 37,869 reports in under 4 minutes on a CPU |
| Qwen3-32B, zero-shot | 89.8% (85.5–92.9) |
| Llama-3.1-70B, zero-shot | 88.3% |
| Llama-3.1-8B, zero-shot | 79.2% |
| MedGemma-4B-IT, zero-shot | 57.2% |
| Flan-T5-base, zero-shot | ≤20.8% |

On the 66 held-out reports, few-shot prompting (5 examples) reached 95.5% for Qwen3-32B and 90.9% for Llama-3.1-70B (rule-based NLP: 92.4%). Where the rules and zero-shot Qwen3-32B agreed (85.6% of reports), 97.8% were correct.

The rules were derived by a cardiologist and two cardiothoracic radiologists, and their accuracy was measured against the radiologists' blinded labels.

## Reproducing the paper

Every model in the paper has its script in `src/`. The tool (`classify_llm`) offers the Ollama models, which matched the rules; MedGemma-4B-IT and Flan-T5-base ran through Hugging Face Transformers in the paper and are kept as research scripts.

| Model | Runner in the paper | Script | Zero-shot accuracy (264 reports) |
|---|---|---|---|
| Rule-based NLP | Python / PySpark (CPU) | `src/cardiac_mri_nlp_5class.py` | 90.2% |
| Qwen3-32B | Ollama | `src/cardiac_mri_llm_5class.py`, `src/rerun_llm_expert264.py` | 89.8% |
| Llama-3.1-70B | Ollama | same | 88.3% |
| Llama-3.1-8B | Ollama | same | 79.2% |
| MedGemma-4B-IT | Hugging Face Transformers, 4-bit NF4 (needs accepting Google's terms for `google/medgemma-4b-it`) | `src/cardiac_mri_medgemma_fewshot.py` | 57.2% |
| Flan-T5-base | Hugging Face Transformers (512-token input) | `src/cardiac_mri_flanT5_5class.py` | ≤20.8% |
| Few-shot (k=5), all Ollama models and MedGemma | as above | `src/cardiac_mri_llm_fewshot_all.py`, `src/fewshot_curve.py`, `src/fewshot_order_test.py` | see the paper |

The scripts read the study's reports, which are not distributed; point `CMR_DATA_DIR` at your own data.

## Repository layout

| Path | What it is |
|---|---|
| `scartext/` | The tool: `classify_scar`, `extract_lvef`, `explain`, `show_report` (report in color), `classify_llm`, `compare`, `analyze_csv` |
| `notebooks/quickstart.ipynb` | Step-by-step notebook (rules, your own report, zero-shot, agreement, few-shot, CSV) |
| `examples/synthetic_reports/` | Six synthetic reports, one per class plus a negated-wording no-scar case |
| `demo/` | Local web demo (Gradio): NLP, zero-shot and optional few-shot side by side |
| `src/` | The paper's research scripts, unchanged in logic: rule-based NLP, zero-shot (Llama, Qwen, MedGemma, Flan-T5), few-shot, and the few-shot curves and order test |

## Settings that matter

The paper found that deployment settings changed LLM accuracy as much as model size. The code sets them explicitly: Ollama context window 8,192 tokens (the default is shorter than many reports plus the prompt); temperature 0 and seed 42; Qwen3 reasoning off; failed or unreadable answers recorded as **missing**, never as a class; and no keyword rule changing the model's answer.

## Data

The study used 37,869 LGE cardiac MRI reports (2005–2025) from Northwestern Medicine. The reports, expert labels and model outputs contain protected health information and are **not distributed**. Everything in this repository runs on synthetic reports; the research scripts in `src/` read study data from the folder set by `CMR_DATA_DIR`.

## Citation

See `CITATION.cff`. [Authors: add the DOI when the paper is published.]

## License

MIT, see `LICENSE`. The LLM weights are not part of this repository.

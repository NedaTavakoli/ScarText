# ScarText

**Classify myocardial scar patterns from free-text cardiac MRI reports** — ischemic, non-ischemic, mixed, no-scar or not-specified — with a rule-based NLP pipeline and open-weight LLMs that run on your own computer.

[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/tavakolineda/ScarText) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NedaTavakoli/ScarText/blob/main/notebooks/quickstart.ipynb)

Code for the paper **"ScarText: Can Large Language Models Match Expert Rules in Classifying Myocardial Scar Patterns from Cardiac MRI Reports?"** (Tavakoli N, Rahsepar AA, Ozturk O, Bagci U, Lee DC, Kim D; under review).

**Research use only. Not for clinical use.**

## What's in this repository

| Folder | What it is | Who it is for |
|---|---|---|
| [`scartext/`](scartext) | **The tool.** A small Python package: `classify_scar` (rules), `classify_llm` (local LLM), `compare`, `analyze_csv`, `extract_lvef`, `explain`, `show_report` | Anyone who wants to classify their own reports |
| [`notebooks/`](notebooks) | `quickstart.ipynb`: a step-by-step tour that runs in Google Colab | First-time users |
| [`demo/`](demo) | A local web app showing the rules and an LLM side by side | Anyone who wants to try it with a browser |
| [`examples/synthetic_reports/`](examples/synthetic_reports) | Six made-up reports, one per class plus a negated no-scar wording | Testing |
| [`paper_scripts/`](paper_scripts) | The paper's original research scripts (rules, every LLM, few-shot, pool-size curves, order test), unchanged in logic | Reviewers and anyone reproducing the paper |

In short: **use `scartext/`; read `paper_scripts/` to see exactly how the paper's numbers were produced.** No patient data are included anywhere.

## Demo

[![Demo: rule-based NLP, zero-shot and few-shot Qwen3-32B on synthetic reports](demo/demo_preview.gif)](demo/CMR_scar_demo_synthetic.mp4)

Each report is shown with three results side by side: the rule-based NLP, zero-shot Qwen3-32B and few-shot Qwen3-32B. Where an LLM agrees with the rules the report is accepted; where it disagrees, it is flagged for expert review. All reports shown are synthetic. [Watch the full video (36 s)](demo/CMR_scar_demo_synthetic.mp4).

**Try it online:** the [live demo](https://huggingface.co/spaces/tavakolineda/ScarText) classifies any report you paste or upload (.txt or .docx) with the rule-based NLP and highlights the terms it uses. It runs the rules and zero-shot Qwen3-32B on Hugging Face's shared GPUs; the other LLMs and few-shot prompting run on your own computer (below). Do not upload identifiable patient information.

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
| gpt-oss-20b (OpenAI, open-weight; runs locally, unlike cloud GPT-4) | `ollama pull gpt-oss:20b` (~13 GB; needs a recent Ollama) | 90.9% | GPU with ≥16 GB, or a Mac with ≥24 GB; about 2 s per report |
| Qwen3-32B (Alibaba) | `ollama pull qwen3:32b` (~20 GB) | 89.8% | GPU with ≥24 GB, or a Mac with ≥32 GB |
| Llama-3.1-70B (Meta) | `ollama pull llama3.1:70b` (~43 GB) | 88.3% | ≥48 GB |
| Llama-3.3-70B (Meta) | `ollama pull llama3.3:70b` (~43 GB) | 87.9% | ≥48 GB |
| MedGemma-27B (Google) | `ollama pull medgemma:27b` (~17 GB; needs a recent Ollama) | 84.5% | GPU with ≥24 GB |
| Llama-3.1-8B (Meta) | `ollama pull llama3.1:8b` (~5 GB) | 79.2% | a laptop, or free Colab |

gpt-oss-20b reasons before it answers; `classify_llm` runs it at its lowest reasoning level with a 4,096-token output cap, as in the paper.

`classify_llm` talks to Ollama at `http://127.0.0.1:11434` on your own machine (change it with `host=` or `OLLAMA_HOST_URL`). The model weights are not redistributed here and keep their own licenses (Llama 3.1 and 3.3 Community Licenses; Apache 2.0 for Qwen3 and gpt-oss; Health AI Developer Foundations terms for MedGemma).

**Few-shot:** pass your own labeled reports, `classify_llm(text, model=..., examples=df)` with columns `report_text` and `gold_label`; five similar examples, one per class first, are put in each prompt, as in the paper. The paper's expert-labeled reports are not distributed.

## Main results

On 264 expert-labeled reports (two blinded readers, κ = 0.995):

| Method | Accuracy |
|---|---|
| Rule-based NLP | 90.2% (95% CI 86.0–93.2); all 37,869 reports in under 4 minutes on a CPU |
| gpt-oss-20b, zero-shot | 90.9% |
| Qwen3-32B, zero-shot | 89.8% (85.5–92.9) |
| Llama-3.1-70B, zero-shot | 88.3% |
| Llama-3.3-70B, zero-shot | 87.9% |
| MedGemma-27B, zero-shot | 84.5% |
| Llama-3.1-8B, zero-shot | 79.2% |
| MedGemma-4B-IT, zero-shot | 57.2% |
| Flan-T5-base, zero-shot | ≤20.8% |

gpt-oss-20b, Qwen3-32B, Llama-3.1-70B and Llama-3.3-70B did not differ from the rules; Llama-3.3-70B, gpt-oss-20b and MedGemma-27B were added after the main analysis. On the 66 held-out reports, few-shot prompting (5 examples) reached 95.5% for Qwen3-32B, 92.4% for Llama-3.3-70B, 90.9% for Llama-3.1-70B and gpt-oss-20b, and 87.9% for MedGemma-27B (rule-based NLP: 92.4%). Where the rules and zero-shot Qwen3-32B agreed (85.6% of reports), 97.8% were correct.

The rules were derived by a cardiologist and two cardiothoracic radiologists, and their accuracy was measured against the radiologists' blinded labels.

## Reproducing the paper

Every model in the paper has its script in `paper_scripts/`. The tool (`classify_llm`) offers the Ollama models, which matched the rules; MedGemma-4B-IT and Flan-T5-base ran through Hugging Face Transformers in the paper and are kept as research scripts.

| Model | Runner in the paper | Script | Zero-shot accuracy (264 reports) |
|---|---|---|---|
| Rule-based NLP | Python / PySpark (CPU) | `paper_scripts/cardiac_mri_nlp_5class.py` | 90.2% |
| Qwen3-32B | Ollama | `paper_scripts/cardiac_mri_llm_5class.py`, `paper_scripts/rerun_llm_expert264.py` | 89.8% |
| Llama-3.1-70B | Ollama | same | 88.3% |
| Llama-3.1-8B | Ollama | same | 79.2% |
| Llama-3.3-70B | Ollama | same | 87.9% |
| gpt-oss-20b | Ollama (recent version) | same, with `--think low --num_predict 4096` | 90.9% |
| MedGemma-27B | Ollama (recent version) | same | 84.5% |
| MedGemma-4B-IT | Hugging Face Transformers, 4-bit NF4 (needs accepting Google's terms for `google/medgemma-4b-it`) | `paper_scripts/cardiac_mri_medgemma_fewshot.py` | 57.2% |
| Flan-T5-base | Hugging Face Transformers (512-token input) | `paper_scripts/cardiac_mri_flanT5_5class.py` | ≤20.8% |
| Few-shot (k=5), all Ollama models and MedGemma | as above | `paper_scripts/cardiac_mri_llm_fewshot_all.py`, `paper_scripts/fewshot_curve.py`, `paper_scripts/fewshot_order_test.py` | see the paper |

The scripts read the study's reports, which are not distributed; point `CMR_DATA_DIR` at your own data.

## Settings that matter

The paper found that deployment settings changed LLM accuracy as much as model size. The code sets them explicitly: Ollama context window 8,192 tokens (the default is shorter than many reports plus the prompt); temperature 0 and seed 42; Qwen3 reasoning off; failed or unreadable answers recorded as **missing**, never as a class; and no keyword rule changing the model's answer.

## Data

The study used 37,869 LGE cardiac MRI reports (2005–2025) from Northwestern Medicine. The reports, expert labels and model outputs contain protected health information and are **not distributed**. Everything in this repository runs on synthetic reports; the research scripts in `paper_scripts/` read study data from the folder set by `CMR_DATA_DIR`.

## Citation

See `CITATION.cff`. [Authors: add the DOI when the paper is published.]

## License

MIT, see `LICENSE`. The LLM weights are not part of this repository.

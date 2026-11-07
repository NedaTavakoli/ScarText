# CMR scar classification demo

**Research use only. Not for clinical use.**

This local Gradio app classifies a cardiac MRI report as ischemic, non-ischemic, mixed, no-scar, or not-specified, the way the paper does:

- **Rule-based NLP:** the study's rules (`src/cardiac_mri_nlp_5class.py`, unchanged), with the terms they look for highlighted.
- **Zero-shot LLM:** an open-weight model served locally by Ollama, with the paper's zero-shot prompt and settings. No model is trained.
- **Agreement:** whether the two agree. In the paper, where the rules and zero-shot Qwen3-32B agreed (85.6% of the 264 expert reports), 97.8% were correct, so the app flags disagreements for expert review.

## Synthetic reports only

The built-in reports are synthetic, written for this demo: one per class, plus a no-scar report with negated scar words ("no evidence of scar"), the wording that a keyword rule in the paper got wrong. With few-shot off (the default) the app reads no study data, so it can be shown or recorded. Pasting or uploading a real report is possible, but do that only on a computer approved for patient data.

Real study reports are not used, even de-identified: removing every name, date and identifier from free-text reports needs a validated de-identification step and a reader's check, and synthetic reports show the same behavior without that risk.

## Run

```bash
./demo/run.sh
```

The first run installs its packages into `~/.cache/cmr-scar-demo-venv`, outside Google Drive. The app opens at http://127.0.0.1:7860. It listens only on this computer and creates no share link. Do not host or publish it without the project lead's explicit approval.

To run the tests:

```bash
~/.cache/cmr-scar-demo-venv/bin/python demo/test_demo.py
```

## Connect the LLMs

Without an LLM the app shows the rule-based result alone. The app uses Ollama on this computer (`ollama pull qwen3:32b`, or `llama3.1:8b` on a laptop); a GPU server can be used instead through an SSH tunnel. Report text is sent only to a local address unless you set `CMR_LLM_ALLOW_REMOTE=1`.

1. On the GPU server, make sure Ollama is running and the model is pulled (`ollama list` should show `qwen3:32b`; the paper also used `llama3.1:70b` and `llama3.1:8b`). The paper's runs used port 11434, and some used a second instance on 11435.
2. On this computer, open the tunnel, replacing the user and host with your own (use 11435 on the right if that is the instance you want):

   ```bash
   ssh -N -L 11434:127.0.0.1:11434 <user>@<gpu-server>
   ```

3. Start the demo. Zero-shot is on whenever the tunnel is up:

   ```bash
   ./demo/run.sh
   ```

Settings: `CMR_LLM_URL` (default `http://127.0.0.1:11434`; empty turns the LLMs off), `CMR_LLM_MODEL` (default `qwen3:32b`, the paper's best model).

Both LLMs use the paper's code and settings: temperature 0, seed 42, an 8,192-token context window, at most 512 output tokens, reasoning off, and the paper's parser. An answer that is not one of the five classes is shown as missing, as in the paper, and no keyword rule changes the model's answer.

- **Zero-shot:** the prompt is read from `src/cardiac_mri_llm_5class.py`.
- **Few-shot (opt-in):** start the demo with `CMR_DEMO_FEWSHOT=1 ./demo/run.sh`. A third card shows the few-shot class. It uses `src/cardiac_mri_llm_fewshot_all.py`: for each report, 5 examples are taken from the 194 expert-labeled training reports by TF-IDF similarity, one per class first, and placed in the prompt. **These examples are real study reports.** In the study they were read from `Results/finetune_llm/data/A_train_expert.jsonl`, which is not distributed; to use your own labeled reports, set `CMR_FEWSHOT_POOL` to a JSON-lines file with `report_id`, `text` and `label` fields. They are sent only to the Ollama server and never shown on screen. Keep few-shot mode on approved computers, and leave it off for recordings shared outside the team.

## Using it

- **Highlights:** anatomy is yellow, LGE pattern is blue and disease is orange, as in Figure 2. No-LGE statements are green. Grey text is a History or Technique section that the pattern rules skip. A highlighted term does not always decide the class.
- **Not classified:** the rules assign no class to reports that mention a cardiac device or have no LGE section.
- **Accuracy:** the rules were derived by a cardiologist and two cardiothoracic radiologists and scored 90.2% on 264 expert-labeled reports. The agreement figure comes from an exploratory analysis in the paper.

**No copies:** an uploaded file's temporary copy is deleted as soon as its text is read.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fewshot_curve.py
================
How many expert-labeled examples does an LLM need? Two curves on the 66 held-out reports, using the prompt,
TF-IDF example retrieval, example format and num_ctx of cardiac_mri_llm_fewshot_all.py (imported, unchanged):
  1. number of in-context examples: k = 0, 1, 2, 3, 5, 10, retrieved from all 198 training reports
  2. size of the labeled pool the examples are retrieved from: 10, 25, 50, 100 reports (k = 5; three stratified
     draws each, the subsets of finetune_llm/prep_subsets.py); 198 = curve 1, k = 5
Then k = 0 on all 264 expert reports (k0_expert264.csv; pool = 0 in summary.csv), a prompt-sensitivity check: this
prompt is shorter than the zero-shot prompt of cardiac_mri_llm_5class.py.
No keyword guard: the model's own answer is scored (raw_label). Failed or unparseable replies are recorded as missing.
Qwen is called with thinking off, as in the zero-shot run.
Writes Results/fewshot_curve198/<model>/k{k}_pool{n}[_r{r}].csv and summary.csv; finished settings are skipped on restart.

Usage
  OLLAMA_HOST=127.0.0.1:11435 python src/fewshot_curve.py --model llama3.1:8b [--n_test 3]
"""
import os, sys, json, time, argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cardiac_mri_llm_fewshot_all as fs
import ollama

SUB = os.path.join(PROJ, "Results", "finetune_llm", "data", "subsets198")
N_TRAIN, N_TEST = 198, 66   # the stratified 75/25 split of the 264 expert reports (Figure 1)
KS, POOLS, DRAWS = [0, 1, 2, 3, 5, 10], [10, 25, 50, 100], [1, 2, 3]


class Chat:
    """stands in for the ollama module inside fs.call_ollama; adds think=False for reasoning models"""
    def __init__(self, think):
        self.client, self.think = ollama.Client(host=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")), think
    def chat(self, **kw):
        if self.think is not None: kw["think"] = self.think
        if os.environ.get("NUM_PREDICT"): kw.setdefault("options", {})["num_predict"] = int(os.environ["NUM_PREDICT"])   # reasoning models (Nov 2026)
        return self.client.chat(**kw)


def run_setting(model, k, pool_df, test_df, path, args):
    retr = fs.FewShotRetriever(pool_df, knn_pool=15)
    rows, t0 = [], time.time()
    for _, r in test_df.iterrows():
        ex = retr.retrieve(r.report_text, k) if k > 0 else pd.DataFrame()
        t = time.time()
        out = fs.call_ollama(model, fs.SYSTEM_PROMPT, fs.build_few_shot_prompt(r.report_text, ex),
                             0.0, retries=6, num_ctx=args.num_ctx)
        rows.append({"report_id": r.report_id, "gold": r.gold_label, "pred": out["label"], "status": out["status"],
                     "example_ids": ";".join(ex.report_id) if k > 0 else "",
                     "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
                     "done_reason": out["done_reason"], "elapsed_s": round(time.time() - t, 2)})
    d = pd.DataFrame(rows); d.to_csv(path, index=False)
    ok = d.pred.notna()
    return {"correct": int((d.pred == d.gold).sum()), "n": len(d), "missing": int((~ok).sum()),
            "accuracy": round((d.pred == d.gold).mean(), 4),   # missing counted as wrong
            "kappa": round(cohen_kappa_score(d.gold[ok], d.pred[ok]), 4),
            "max_prompt_tokens": int(d.prompt_tokens.max()) if d.prompt_tokens.notna().any() else None,
            "minutes": round((time.time() - t0) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tag: llama3.1:8b, llama3.1:70b, qwen3:32b")
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--n_test", type=int, default=None, help="smoke test")
    ap.add_argument("--out", default=os.path.join(PROJ, "Results", "fewshot_curve198"))
    ap.add_argument("--only", default=None, help="run only these settings, e.g. k5_pool198")
    args = ap.parse_args()
    fs.ollama = Chat(os.environ.get("THINK") or (False if args.model.startswith("qwen") else None))   # THINK=low for gpt-oss (Nov 2026)
    df = fs.load_expert_data(os.path.join(HERE, "Two_Expert_Annotation_Comparision.xlsx"), None)
    train_df, test_df = fs.make_split(df, os.path.join(HERE, "train_test_split_expert.xlsx"))
    assert len(train_df) == N_TRAIN and len(test_df) == N_TEST, (len(train_df), len(test_df))
    if args.n_test: test_df = test_df.head(args.n_test)
    od = os.path.join(args.out, args.model.replace(":", "_") + ("_smoke" if args.n_test else ""))
    os.makedirs(od, exist_ok=True)
    settings = [(k, N_TRAIN, 0, train_df) for k in KS]
    for n in POOLS:
        for r in DRAWS:
            ids = {json.loads(l)["report_id"] for l in open(os.path.join(SUB, f"n{n}_r{r}.jsonl"))}
            assert ids <= set(train_df.report_id), f"subset n{n}_r{r} not within the {N_TRAIN} training reports"
            settings.append((5, n, r, train_df[train_df.report_id.isin(ids)].reset_index(drop=True)))
    summ_path = os.path.join(od, "summary.csv")
    summ = pd.read_csv(summ_path).to_dict("records") if os.path.exists(summ_path) else []
    for k, n, r, pool in settings:
        name = f"k{k}_pool{n}" + (f"_r{r}" if r else "")
        path = os.path.join(od, name + ".csv")
        if os.path.exists(path) or (args.only and name not in args.only.split(",")): continue
        s = {"model": args.model, "k": k, "pool": n, "draw": r, **run_setting(args.model, k, pool, test_df, path, args)}
        summ.append(s); pd.DataFrame(summ).to_csv(summ_path, index=False)
        print(json.dumps(s), flush=True)
    # prompt sensitivity: this script's shorter system prompt, no examples, on all 264 expert reports (annotated text),
    # to compare with the zero-shot runs (cardiac_mri_llm_5class.SYSTEM_PROMPT) on the same reports
    path = os.path.join(od, "k0_expert264.csv")
    if not os.path.exists(path) and not args.only:
        e = pd.DataFrame([json.loads(l) for l in open(os.path.join(PROJ, "Results", "finetune_llm", "data", "B_test_expert264.jsonl"))])
        e = e.rename(columns={"text": "report_text", "label": "gold_label"})
        if args.n_test: e = e.head(args.n_test)
        s = {"model": args.model, "k": 0, "pool": 0, "draw": 0, **run_setting(args.model, 0, train_df, e, path, args)}
        summ.append(s); pd.DataFrame(summ).to_csv(summ_path, index=False)
        print(json.dumps(s), flush=True)
    print("FEWSHOT_CURVE_DONE", args.model, flush=True)


if __name__ == "__main__":
    main()

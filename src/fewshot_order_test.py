#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fewshot_order_test.py
=====================
Does Llama-3.1-8B copy the label of the last few-shot example? Repeats the k = 5 pool settings of fewshot_curve.py
with the same examples in reversed order, on the held-out 64 reports, and compares with the original runs.
Control: k5_pool25_r1 in the original order, to check that a repeat run reproduces the original answers.
Writes Results/fewshot_order_test/<model>/ and report-free comparison.csv; finished settings are skipped on restart.

Usage
  OLLAMA_HOST=127.0.0.1:11435 python src/fewshot_order_test.py --model llama3.1:8b [--n_test 3]
"""
import os, sys, json, time, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fewshot_curve as fc

fs = fc.fs
ORIG = os.path.join(fc.PROJ, "Results", "fewshot_curve")


def run_setting(model, pool_df, test_df, orig, path, reverse, args):
    retr = fs.FewShotRetriever(pool_df, knn_pool=15)
    rows = []
    for _, r in test_df.iterrows():
        ex = retr.retrieve(r.report_text, 5)
        assert ";".join(ex.report_id) == orig.loc[r.report_id, "example_ids"], "retrieved examples differ from the original run"
        if reverse: ex = ex.iloc[::-1].reset_index(drop=True)
        out = fs.call_ollama(model, fs.SYSTEM_PROMPT, fs.build_few_shot_prompt(r.report_text, ex), 0.0, retries=6, num_ctx=args.num_ctx)
        rows.append({"report_id": r.report_id, "gold": r.gold_label, "pred": out["label"], "status": out["status"],
                     "example_ids": ";".join(ex.report_id), "last_label": ex.gold_label.iloc[-1], "first_label": ex.gold_label.iloc[0]})
    d = pd.DataFrame(rows); d.to_csv(path, index=False)
    return d


def compare(name, d, orig, labels):
    o = orig.loc[d.report_id]
    o_last = o.example_ids.map(lambda s: labels[s.split(";")[-1]]).values
    return {"setting": name, "n": len(d), "correct_orig": int((o.pred.values == d.gold.values).sum()), "correct_new": int((d.pred == d.gold).sum()),
            "same_answer": int((o.pred.values == d.pred.values).sum()),
            "orig_answer_is_orig_last": int((o.pred.values == o_last).sum()), "new_answer_is_new_last": int((d.pred == d.last_label).sum()),
            "new_answer_is_orig_last": int((d.pred.values == o_last).sum()),
            "gold_is_orig_last": int((d.gold.values == o_last).sum()), "gold_is_new_last": int((d.gold == d.last_label).sum()),
            "missing_new": int(d.pred.isna().sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--n_test", type=int, default=None, help="smoke test")
    ap.add_argument("--full_pool", action="store_true", help="also test k = 5 from all 194 training reports, first")
    args = ap.parse_args()
    fs.ollama = fc.Chat(False if args.model.startswith("qwen") else None)  # thinking off for Qwen, as in fewshot_curve.py
    df = fs.load_expert_data(os.path.join(HERE, "Two_Expert_Annotation_Comparision.xlsx"), None)
    train_df, test_df = fs.make_split(df, os.path.join(HERE, "train_test_split_expert.xlsx"))
    assert len(train_df) == 194 and len(test_df) == 64, (len(train_df), len(test_df))
    if args.n_test: test_df = test_df.head(args.n_test)
    tag = args.model.replace(":", "_")
    od = os.path.join(fc.PROJ, "Results", "fewshot_order_test", tag + ("_smoke" if args.n_test else ""))
    os.makedirs(od, exist_ok=True)
    labels = dict(zip(train_df.report_id, train_df.gold_label))
    settings = [(n, r, True) for n in fc.POOLS for r in fc.DRAWS] + [(25, 1, False)]
    if args.full_pool: settings = [(194, 0, True), (194, 0, False)] + settings
    results = []
    for n, r, reverse in settings:
        base = f"k5_pool{n}_r{r}" if n != 194 else "k5_pool194"
        name = base + ("_reversed" if reverse else "_control")
        path = os.path.join(od, name + ".csv")
        orig = pd.read_csv(os.path.join(ORIG, tag, base + ".csv")).set_index("report_id")
        if os.path.exists(path):
            d = pd.read_csv(path)
        else:
            t0 = time.time()
            if n == 194:
                pool = train_df
            else:
                ids = {json.loads(l)["report_id"] for l in open(os.path.join(fc.SUB, f"n{n}_r{r}.jsonl"))}
                pool = train_df[train_df.report_id.isin(ids)].reset_index(drop=True)
            d = run_setting(args.model, pool, test_df, orig, path, reverse, args)
            print(name, "minutes", round((time.time() - t0) / 60, 1), flush=True)
        results.append(compare(name, d, orig, labels))
        pd.DataFrame(results).to_csv(os.path.join(od, "comparison.csv"), index=False)
    print("ORDER_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()

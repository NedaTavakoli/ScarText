#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_llm_expert264.py
======================
Re-run Llama-3.1 (8B / 70B) zero-shot on the 264 expert-annotated reports with an
explicit Ollama context window (num_ctx), keeping everything else identical to
cardiac_mri_llm_5class.py (same SYSTEM_PROMPT, USER_TEMPLATE, temperature, seed,
num_predict, max_chars, retry logic and post-processing guard).

Adds per-report diagnostics so failure causes can be separated:
  prompt_tokens   -- Ollama prompt_eval_count (>2048 => would be truncated by the
                     old 2,048-token default context)
  output_tokens   -- Ollama eval_count (== num_predict => output was cut off)
  done_reason     -- 'stop' or 'length'
  parse_status    -- json_ok | regex_fallback | invalid_class | parse_failed | error
  guard_triggered -- post-processing no-scar -> not-specified reassignment

Usage:
  python src/rerun_llm_expert264.py --model llama3.1:8b  --num_ctx 8192
  python src/rerun_llm_expert264.py --model llama3.1:70b --num_ctx 8192
"""
import os, re, sys, json, time, argparse
import pandas as pd
from tqdm import tqdm
import ollama

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cardiac_mri_llm_5class import SYSTEM_PROMPT, USER_TEMPLATE, CLASSES, postprocess

EXPERT = os.path.join(PROJ, "Expert annotation", "CMR_Expert_Annotation_and_all_models.xlsx")
SOURCE = os.path.join(HERE, "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx")


ANNOT = os.path.join(PROJ, "Expert annotation", "Two_Expert_Annotation_Comparision.xlsx")


def _annotated_texts():
    """report_id -> the CMR Report Text shown to the readers (Reader 1 sheet; Reader 2 is identical)."""
    x = pd.read_excel(ANNOT, sheet_name="Reader1_Annotation (Onur)", header=None)
    h = x.index[x.iloc[:, 0].astype(str).str.strip() == "Report ID"][0]
    x = x.iloc[h + 1:, [0, 3]]
    x.columns = ["report_id", "text"]
    x = x[x.report_id.astype(str).str.startswith("RPT")]
    return dict(zip(x.report_id, x.text))


def load_expert264(text_source="annotated"):
    """text_source='annotated': the report text the readers annotated. Many accessions have several
    source rows with different text, so the source row whose text matches the annotated text is used
    (original formatting); the annotated text itself if no row matches.
    text_source='first': first source row per accession (the earlier runs, Results/rerun_numctx)."""
    b = pd.read_excel(EXPERT, sheet_name="_NLP_Labels_BLIND", header=1)
    nm = lambda s: s.astype(str).str.strip().str.lower()
    b["r1"], b["r2"] = nm(b["Reader 1 Label ▼"]), nm(b["Reader 2 Label ▼"])
    b = b[b.r1.isin(CLASSES) & b.r2.isin(CLASSES)].copy()
    b["accession"] = b["accession_number_real"].astype(str).str.strip()
    b["orig_8b"] = nm(b["LLM_8B (Llama-3.1-8B)"])
    b["orig_70b"] = nm(b["LLM_70B (Llama-3.1-70B)"])
    assert len(b) == 264, len(b)

    src = pd.read_excel(SOURCE, header=8, usecols=["accession number", "cardiac MRI text"])
    src["row_id"] = src.index
    src["accession"] = src["accession number"].astype(str).str.strip()
    src = src.dropna(subset=["cardiac MRI text"])
    if text_source == "first":
        # first source row per accession, as in compute_all_model_metrics.py
        src = src.drop_duplicates("accession", keep="first")
        df = b.merge(src[["accession", "row_id", "cardiac MRI text"]], on="accession", how="left")
        df["text_match"] = "first_row"
    else:
        norm = lambda s: " ".join(str(s).split())
        annot = _annotated_texts()
        src["norm"] = src["cardiac MRI text"].map(norm)
        texts, rows, how = [], [], []
        for rid, acc in zip(b.report_id, b.accession):
            a = annot[rid]
            hit = src[(src.accession == acc) & (src.norm == norm(a))]
            if len(hit):
                texts.append(hit["cardiac MRI text"].iloc[0]); rows.append(hit.row_id.iloc[0]); how.append("source_row")
            else:
                texts.append(a); rows.append(None); how.append("annotated_text")
        df = b.copy()
        df["cardiac MRI text"], df["row_id"], df["text_match"] = texts, rows, how
    missing = df["cardiac MRI text"].isna().sum()
    assert missing == 0, f"{missing} expert reports have no source text"
    return df[["report_id", "accession", "row_id", "text_match", "r1", "r2", "orig_8b", "orig_70b", "cardiac MRI text"]]


def _chat(model, messages, options, think):
    """ollama.chat; think=False switches off reasoning output for models that have it
    (e.g. Qwen). Llama models are called exactly as before (no think argument)."""
    if think is None:
        return ollama.chat(model=model, messages=messages, options=options)
    return ollama.chat(model=model, messages=messages, options=options, think=think)


def call_ollama(report_text, model, num_ctx, max_chars=5000, retries=6, think=None):
    """cardiac_mri_llm_5class.call_ollama + num_ctx + diagnostics (same decision logic):
    failed calls, unparseable replies and out-of-taxonomy classes are recorded as missing
    (scar_type_llm=None), never as not-specified."""
    text = str(report_text)[:max_chars]
    prompt = USER_TEMPLATE.format(report_text=text)
    diag = {"prompt_tokens": None, "output_tokens": None, "done_reason": None,
            "thinking_chars": 0, "llm_raw": ""}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    options = {"temperature": 0.0, "seed": 42, "num_predict": 512, "num_ctx": num_ctx}
    missing = lambda status, ev: {"scar_type_llm": None, "llm_confidence": 0.0, "llm_evidence": ev,
                                  "parse_status": status, "llm_status": status, **diag}
    for attempt in range(retries + 1):
        try:
            resp = _chat(model, messages, options, think)
            break
        except TypeError:
            raise   # ollama-python too old for think=: fail loudly instead of retrying
        except Exception as e:
            if attempt < retries:
                time.sleep(min(2 ** (attempt + 1), 60))
            else:
                return missing("call_failed", f"ERROR: {str(e)[:100]}")
    msg = resp["message"]
    diag.update(prompt_tokens=resp.get("prompt_eval_count"), output_tokens=resp.get("eval_count"),
                done_reason=resp.get("done_reason"),
                thinking_chars=len(getattr(msg, "thinking", None)
                                   or (msg.get("thinking") if isinstance(msg, dict) else None) or ""))
    raw = (msg["content"] or "").strip()
    diag["llm_raw"] = raw
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()   # inline reasoning, if any
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'"classification"\s*:\s*"(ischemic|non-ischemic|mixed|no-scar|not-specified)"',
                      raw, re.IGNORECASE)
        if m:
            return {"scar_type_llm": m.group(1).lower(), "llm_confidence": 0.5,
                    "llm_evidence": "JSON parse failed — regex fallback",
                    "parse_status": "regex_fallback", "llm_status": "regex_fallback", **diag}
        return missing("parse_failed", raw[:300])
    cls = str(parsed.get("classification", "")).strip().lower()
    if cls not in CLASSES:
        return missing("invalid_class", f"invalid class: {cls[:50]}")
    try:
        conf = float(parsed.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    return {"scar_type_llm": cls, "llm_confidence": conf,
            "llm_evidence": str(parsed.get("evidence", ""))[:300],
            "parse_status": "json_ok", "llm_status": "json_ok", **diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b",
                    help="Ollama tag, e.g. llama3.1:8b, llama3.1:70b, qwen3.8:27b")
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--think", choices=["off", "on", "default"], default=None,
                    help="Reasoning mode for models that have one (default: off for non-Llama "
                         "models, not sent for Llama)")
    ap.add_argument("--n", type=int, default=None, help="limit to first N reports (smoke test)")
    ap.add_argument("--text_source", choices=["annotated", "first"], default="annotated",
                    help="report text: the one the readers annotated (default) or the first source row per "
                         "accession (used by the earlier runs in Results/rerun_numctx)")
    ap.add_argument("--out_dir", default=os.path.join(PROJ, "Results", "rerun_annot"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    is_llama = args.model.startswith("llama")
    mode = args.think or ("default" if is_llama else "off")
    think = {"off": False, "on": True, "default": None}[mode]

    df = load_expert264(args.text_source)
    if args.n:
        df = df.head(args.n)
    # llama keeps the original file names (llm_8b_..., llm_70b_...)
    tag = args.model.split(":")[1] if is_llama else re.sub(r"[^A-Za-z0-9.]+", "_", args.model)
    out = os.path.join(args.out_dir, f"llm_{tag}_expert264_ctx{args.num_ctx}{'_n%d' % args.n if args.n else ''}.xlsx")
    print(f"{args.model}  num_ctx={args.num_ctx}  think={mode}  n={len(df)}  ->  {out}")

    rows, t0 = [], time.time()
    for _, r in tqdm(df.iterrows(), total=len(df)):
        text = str(r["cardiac MRI text"])
        res = call_ollama(text, args.model, args.num_ctx, think=think)
        res = postprocess(res, text)
        res["excluded_by_postprocess"] = str(res.get("llm_status")).startswith("excluded")
        rows.append(res)
    secs = time.time() - t0

    res = pd.DataFrame(rows)
    keep = df.drop(columns=["cardiac MRI text"]).reset_index(drop=True)
    outdf = pd.concat([keep, res], axis=1)
    outdf["correct"] = outdf["scar_type_llm"] == outdf["r1"]

    from sklearn.metrics import cohen_kappa_score
    valid = outdf[outdf.scar_type_llm.isin(CLASSES)]
    summ = pd.DataFrame([{
        "model": args.model, "num_ctx": args.num_ctx, "think": mode, "n": len(outdf), "n_valid": len(valid),
        "n_missing": int(outdf.scar_type_llm.isna().sum()),
        "n_with_thinking_output": int((outdf.thinking_chars > 0).sum()),
        "n_guard_triggered": int(outdf.guard_triggered.sum()),
        "accuracy": round(valid.correct.mean(), 4),
        "kappa": round(cohen_kappa_score(valid.r1, valid.scar_type_llm), 4),
        "pred_not_specified": int((outdf.scar_type_llm == "not-specified").sum()),
        "prompt_tokens_gt_2048": int((outdf.prompt_tokens.fillna(0) > 2048).sum()),
        "output_hit_512": int((outdf.output_tokens.fillna(0) >= 512).sum()),
        "wall_seconds": round(secs, 1), "sec_per_report": round(secs / len(outdf), 2),
    }])
    status = outdf.parse_status.value_counts().rename_axis("parse_status").reset_index(name="n")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        outdf.to_excel(w, sheet_name="Row_Level", index=False)
        summ.to_excel(w, sheet_name="Summary", index=False)
        status.to_excel(w, sheet_name="Parse_Status", index=False)
    print(summ.T.to_string()); print(status.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

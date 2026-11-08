#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cardiac_mri_medgemma_fewshot.py
================================
Standalone MedGemma few-shot inference for 5-class myocardial scar
phenotyping from cardiac MRI free-text reports.

• Completely independent of NLP labels — no rowlevel NLP file needed.
• Loads report text directly from the source Excel.
• Few-shot examples are drawn from consensus expert annotations
  (Two_Expert_Annotation_Comparision.xlsx, agreement cases only).
• Output mirrors the Row_Level sheet structure of the NLP pipeline:
    row_id | MRN | procedure date | accession number | scar_type_medgemma
    | medgemma_confidence | medgemma_evidence | medgemma_guard_note

MedGemma can run in two modes
------------------------------
  LOCAL (HuggingFace Transformers)  — faster, full control, GPU required
  OLLAMA                            — simpler setup, slightly slower

Speed comparison (approximate, A40 GPU, 4-bit quant)
------------------------------------------------------
  HuggingFace local  ~1–2 s/report  (direct CUDA, no API overhead)
  Ollama             ~3–6 s/report  (HTTP round-trip + process overhead)
  → LOCAL is typically 2–4× faster than Ollama for MedGemma-4B.
  → For 41k reports: local ≈ 12–24 h, Ollama ≈ 35–70 h.
  → Use LOCAL when running the full cohort on the GPU server.

Requirements
------------
  pip install transformers accelerate bitsandbytes torch
  pip install pandas openpyxl tqdm

  # HuggingFace access (MedGemma is gated)
  #   1. Accept terms at https://huggingface.co/google/medgemma-4b-it
  #   2. huggingface-cli login
  # OR pass --local_model_dir for fully offline / air-gapped GPU server

  # Ollama mode only:
  #   curl -fsSL https://ollama.com/install.sh | sh
  #   ollama serve &
  #   ollama pull medgemma  (check https://ollama.com/library for tag)

Usage
-----
  # LOCAL mode (default) — smoke test, 50 reports
  python cardiac_mri_medgemma_fewshot.py \\
      --source_excel "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx" \\
      --expert_excel "Two_Expert_Annotation_Comparision.xlsx" \\
      --n_test 50 \\
      --backend local

  # LOCAL mode — full run, offline weights
  python cardiac_mri_medgemma_fewshot.py \\
      --source_excel "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx" \\
      --expert_excel "Two_Expert_Annotation_Comparision.xlsx" \\
      --local_model_dir /path/to/medgemma-4b-it \\
      --output row_level_medgemma.xlsx

  # OLLAMA mode
  python cardiac_mri_medgemma_fewshot.py \\
      --source_excel "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx" \\
      --expert_excel "Two_Expert_Annotation_Comparision.xlsx" \\
      --backend ollama \\
      --ollama_model medgemma \\
      --output row_level_medgemma.xlsx

  # Skip already-processed rows (resume interrupted run)
  python cardiac_mri_medgemma_fewshot.py \\
      --source_excel "LLM-based_Processing_of_LGE_MR_Images_-_Cardiac_MRI_Report.xlsx" \\
      --expert_excel "Two_Expert_Annotation_Comparision.xlsx" \\
      --resume --output row_level_medgemma.xlsx
"""

import os, re, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES         = ["ischemic", "non-ischemic", "mixed", "no-scar", "not-specified"]
MEDGEMMA_HF_ID  = "google/medgemma-4b-it"
OLLAMA_MODEL    = "medgemma"
N_FEW_SHOT      = 2          # examples per class injected into each prompt
MAX_CHARS       = 5000       # report text truncation
MAX_NEW_TOKENS  = 512
CHECKPOINT_FREQ = 500        # save partial results every N reports

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a board-certified cardiovascular radiologist with 15 years of CMR \
reporting experience.

Your task: classify a cardiac MRI report into EXACTLY ONE of 5 scar phenotype \
classes based on LGE (late gadolinium enhancement) findings.

=== 5-CLASS TAXONOMY ===
1. ischemic       — LGE in subendocardial OR transmural distribution within a
                    coronary artery territory (LAD/RCA/LCx).
                    Key phrases: subendocardial, transmural, CAD, infarction,
                    STEMI, NSTEMI, coronary territory, myocardial infarction
2. non-ischemic   — LGE in mid-wall, epicardial, patchy, or RV-insertion
                    pattern NOT following a coronary territory.
                    Key phrases: mid-wall, epicardial, myocarditis, sarcoidosis,
                    HCM, DCM, amyloid, patchy enhancement, RV insertion
3. mixed          — BOTH ischemic AND non-ischemic LGE patterns present in the
                    SAME study simultaneously.
                    Requires positive evidence of BOTH patterns.
4. no-scar        — LGE imaging performed and gadolinium given, but NO
                    enhancement identified.
                    Key phrases: no LGE, no delayed enhancement, no scar
5. not-specified  — LGE present but pattern genuinely unclassifiable.
                    Use ONLY as a last resort.

=== DECISION HIERARCHY (apply in strict order) ===
Step 1: Explicit "no LGE / no delayed enhancement" in FINDINGS → no-scar
Step 2: Positive evidence of BOTH ischemic AND non-ischemic patterns → mixed
Step 3: Subendocardial/transmural LGE in coronary territory → ischemic
Step 4: Mid-wall/epicardial/non-coronary LGE → non-ischemic
Step 5: LGE confirmed but pattern unclassifiable → not-specified

=== CRITICAL FAILURE MODE WARNINGS ===
⚠ FINDINGS overrides IMPRESSION — always read FINDINGS first.
⚠ After finding ischemic evidence, keep reading for any non-ischemic pattern.
⚠ Use not-specified only when the pattern is genuinely unclassifiable.

=== OUTPUT FORMAT ===
Return ONLY valid JSON, no preamble, no explanation outside the JSON block:
{
  "step1_no_lge_check": "exact quote or none",
  "step2_ischemic_evidence": "exact quote or none",
  "step2_nonischemic_evidence": "exact quote or none",
  "step3_mixed_check": "yes/no + one-line reasoning",
  "classification": "ischemic|non-ischemic|mixed|no-scar|not-specified",
  "confidence": 0.0-1.0,
  "evidence": "brief quote from report that most directly supports classification"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot example builder
# ─────────────────────────────────────────────────────────────────────────────
def build_few_shot_block(expert_df: pd.DataFrame,
                         n_per_class: int = N_FEW_SHOT,
                         max_chars: int = 1500,
                         seed: int = 42) -> str:
    """
    Selects n_per_class consensus-annotated examples per class from the expert
    annotation file and formats them as a few-shot block prepended to the prompt.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for cls in CLASSES:
        subset = expert_df[expert_df["consensus_label"] == cls]
        if len(subset) == 0:
            continue
        chosen = subset.sample(
            n=min(n_per_class, len(subset)),
            random_state=int(rng.integers(0, 9999))
        )
        for _, row in chosen.iterrows():
            text_snip = str(row["CMR Report Text"])[:max_chars]
            example_json = json.dumps({
                "step1_no_lge_check": "see report",
                "step2_ischemic_evidence": "see report" if cls in ("ischemic", "mixed") else "none",
                "step2_nonischemic_evidence": "see report" if cls in ("non-ischemic", "mixed") else "none",
                "step3_mixed_check": "yes" if cls == "mixed" else "no",
                "classification": cls,
                "confidence": 0.95,
                "evidence": "expert-annotated consensus label"
            }, indent=2)
            blocks.append(
                f"=== FEW-SHOT EXAMPLE (label: {cls}) ===\n"
                f"REPORT:\n{text_snip}\n\n"
                f"ANSWER:\n{example_json}\n"
            )
    return "\n".join(blocks)


def build_user_message(report_text: str, few_shot_block: str,
                       max_chars: int = MAX_CHARS) -> str:
    text = str(report_text)[:max_chars]
    return (
        f"{few_shot_block}\n"
        f"=== NOW CLASSIFY THIS REPORT ===\n"
        f"REPORT:\n{text}\n\n"
        f"Return ONLY valid JSON."
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON / response parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_response(raw: str) -> dict:
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean)
    try:
        parsed = json.loads(clean)
        cls = str(parsed.get("classification", "")).strip().lower()
        if cls not in CLASSES:
            cls = "not-specified"
        return {
            "scar_type_medgemma":     cls,
            "medgemma_confidence":    float(parsed.get("confidence", 0.5)),
            "medgemma_evidence":      str(parsed.get("evidence", ""))[:300],
            "medgemma_guard_note":    "",
        }
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(
        r'"classification"\s*:\s*"(ischemic|non-ischemic|mixed|no-scar|not-specified)"',
        raw, re.IGNORECASE
    )
    cls = m.group(1).lower() if m else "not-specified"
    return {
        "scar_type_medgemma":  cls,
        "medgemma_confidence": 0.5,
        "medgemma_evidence":   "JSON parse failed — regex fallback",
        "medgemma_guard_note": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing guard
# ─────────────────────────────────────────────────────────────────────────────
_POS_LGE_SIGNALS = [
    "subendocardial", "transmural", "mid-wall", "midwall", "epicardial",
    "patchy lge", "patchy enhancement", "scar", "fibrosis",
    "lge present", "enhancement present", "delayed enhancement present",
]

def postprocess_guard(result: dict, text: str) -> dict:
    if result["scar_type_medgemma"] == "no-scar":
        tl = text.lower()
        findings_block = re.search(r"findings?.*?(?=impression|conclusion|$)", tl, re.DOTALL)
        if findings_block and any(s in findings_block.group() for s in _POS_LGE_SIGNALS):
            result["scar_type_medgemma"] = "not-specified"
            result["medgemma_guard_note"] = "GUARD: positive LGE in FINDINGS overrides no-scar"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Backend: HuggingFace local
# ─────────────────────────────────────────────────────────────────────────────
class LocalMedGemma:
    """
    Runs MedGemma via HuggingFace Transformers with 4-bit quantization.
    Fastest option for GPU servers — no HTTP overhead.
    """
    def __init__(self, model_id: str = MEDGEMMA_HF_ID,
                 local_model_dir: str = None,
                 use_4bit: bool = True):
        self.model_id = local_model_dir or model_id
        self.use_4bit = use_4bit
        self._load()

    def _load(self):
        offline = bool(self.model_id != MEDGEMMA_HF_ID)
        mode    = "LOCAL path" if offline else "HuggingFace"
        print(f"\nLoading MedGemma from {mode}: {self.model_id}")

        try:
            import torch
            from transformers import pipeline as hf_pipeline, BitsAndBytesConfig
        except ImportError:
            raise ImportError("pip install transformers accelerate bitsandbytes torch")

        device     = "cuda" if torch.cuda.is_available() else "cpu"
        quant_cfg  = None

        if self.use_4bit and device == "cuda":
            try:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("  4-bit quantization enabled (~4 GB VRAM)")
            except Exception:
                print("  bitsandbytes unavailable — using fp16")

        self.pipe = hf_pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "quantization_config": quant_cfg,
                "torch_dtype": "auto" if quant_cfg else "float16",
            },
            device_map="auto",
            local_files_only=offline,
        )
        print(f"  MedGemma ready on: {device}")

    def generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        out = self.pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1].get("content", "")
        return str(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Ollama
# ─────────────────────────────────────────────────────────────────────────────
class OllamaMedGemma:
    """
    Runs MedGemma via Ollama (local HTTP server).
    Simpler setup but 2–4× slower than the HuggingFace backend due to
    HTTP round-trip and process overhead. Suitable for quick tests.
    """
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self._check()

    def _check(self):
        try:
            import ollama as _ol
            self._ol = _ol
            _ol.list()
            print(f"\nOllama connected — model: {self.model}")
        except ImportError:
            raise ImportError("pip install ollama")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama: {e}\n"
                "Run: ollama serve\n"
                f"Pull model: ollama pull {self.model}"
            )

    def generate(self, system: str, user: str) -> str:
        resp = self._ol.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": 0.0, "seed": 42, "num_predict": MAX_NEW_TOKENS},
        )
        return resp["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_source_excel(path: str) -> pd.DataFrame:
    """
    Load the raw 41,501-row source Excel.
    Header is on row 9 (0-indexed: header=8).
    """
    print(f"\nLoading source Excel: {path}")
    df = pd.read_excel(path, header=8)
    df.columns = df.columns.str.strip()
    df["row_id"] = df.index

    required = ["MRN", "cardiac MRI text", "accession number", "procedure date"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Source Excel missing columns: {missing}\nFound: {df.columns.tolist()}")

    df = df[["row_id", "MRN", "procedure date", "accession number", "cardiac MRI text"]].copy()
    df = df[df["cardiac MRI text"].notna()].reset_index(drop=True)
    print(f"  Loaded {len(df):,} reports with non-empty text")
    return df


def load_expert_annotations(path: str) -> pd.DataFrame:
    """
    Load both reader sheets, keep consensus cases (R1 == R2), return with
    columns: accession number | CMR Report Text | consensus_label
    """
    print(f"\nLoading expert annotations: {path}")

    r1 = pd.read_excel(path, sheet_name="Reader1_Annotation (Onur)", header=2)
    r2 = pd.read_excel(path, sheet_name="Reader2_Annotation (Amir)", header=2)

    r1 = r1[["Accession #", "CMR Report Text", "Reader 1 Label \u25bc"]].copy()
    r2 = r2[["Accession #", "Reader 2 Label \u25bc"]].copy()

    r1.columns = ["accession number", "CMR Report Text", "label_r1"]
    r2.columns = ["accession number", "label_r2"]

    merged = r1.merge(r2, on="accession number", how="inner")
    merged["label_r1"] = merged["label_r1"].str.strip().str.lower()
    merged["label_r2"] = merged["label_r2"].str.strip().str.lower()

    consensus = merged[
        merged["label_r1"].isin(CLASSES) &
        merged["label_r2"].isin(CLASSES) &
        (merged["label_r1"] == merged["label_r2"])
    ].copy()
    consensus["consensus_label"] = consensus["label_r1"]
    consensus = consensus[["accession number", "CMR Report Text", "consensus_label"]]

    print(f"  Total annotated : {len(merged):,}")
    print(f"  Consensus cases : {len(consensus):,}  (R1 == R2)")
    print(f"  Class distribution:\n{consensus['consensus_label'].value_counts().to_string()}")
    return consensus.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(df: pd.DataFrame, backend, few_shot_block: str,
                  output_path: str, checkpoint_freq: int = CHECKPOINT_FREQ) -> pd.DataFrame:

    results = []
    t_start = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="MedGemma inference"):
        text = str(row.get("cardiac MRI text", ""))
        user_msg = build_user_message(text, few_shot_block)

        try:
            raw    = backend.generate(SYSTEM_PROMPT, user_msg)
            result = parse_response(raw)
            result = postprocess_guard(result, text)
        except Exception as e:
            result = {
                "scar_type_medgemma":  "not-specified",
                "medgemma_confidence": 0.0,
                "medgemma_evidence":   f"ERROR: {str(e)[:150]}",
                "medgemma_guard_note": "",
            }

        results.append({
            "row_id":           row["row_id"],
            "MRN":              row["MRN"],
            "procedure date":   row["procedure date"],
            "accession number": row["accession number"],
            **result,
        })

        # Checkpoint save
        if (i + 1) % checkpoint_freq == 0:
            ckpt = output_path.replace(".xlsx", f"_ckpt_{i+1}.xlsx")
            pd.DataFrame(results).to_excel(ckpt, index=False, sheet_name="Row_Level")
            elapsed = time.time() - t_start
            rate    = elapsed / (i + 1)
            remaining = rate * (len(df) - i - 1)
            print(f"\n  Checkpoint saved: {ckpt}")
            print(f"  {i+1}/{len(df)} reports | {rate:.1f} s/report | "
                  f"ETA: {remaining/3600:.1f} h")

    result_df = pd.DataFrame(results)
    elapsed   = time.time() - t_start
    print(f"\nInference complete: {len(result_df):,} reports in "
          f"{elapsed/60:.1f} min ({elapsed/len(result_df):.1f} s/report)")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
def save_results(result_df: pd.DataFrame, output_path: str):
    summary = result_df["scar_type_medgemma"].value_counts().reset_index()
    summary.columns = ["scar_type", "count"]
    summary["percent"] = (summary["count"] / len(result_df) * 100).round(2)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Row_Level", index=False)
        summary.to_excel(writer, sheet_name="ScarType_Summary", index=False)

    print(f"\nSaved: {output_path}")
    print(f"  Sheet 'Row_Level'       — {len(result_df):,} per-report predictions")
    print(f"  Sheet 'ScarType_Summary' — class distribution")
    print(f"\nClass distribution:")
    print(summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="MedGemma few-shot inference — standalone CMR scar phenotyping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source_excel",    required=True,
                   help="Raw 41,501-row source Excel with cardiac MRI text (header=8)")
    p.add_argument("--expert_excel",    required=True,
                   help="Two_Expert_Annotation_Comparision.xlsx for few-shot examples")
    p.add_argument("--output",          default="row_level_medgemma.xlsx",
                   help="Output Excel file. Default: row_level_medgemma.xlsx")
    p.add_argument("--backend",         choices=["local", "ollama"], default="local",
                   help="Inference backend. local=HuggingFace (faster), ollama=Ollama. Default: local")
    p.add_argument("--local_model_dir", default=None, metavar="PATH",
                   help="Path to local MedGemma weights (offline/air-gapped mode)")
    p.add_argument("--ollama_model",    default=OLLAMA_MODEL,
                   help=f"Ollama model tag. Default: {OLLAMA_MODEL}")
    p.add_argument("--no_4bit",         action="store_true",
                   help="Disable 4-bit quantization (uses fp16, more VRAM)")
    p.add_argument("--n_test",          type=int, default=None,
                   help="Limit to N reports (smoke test)")
    p.add_argument("--n_few_shot",      type=int, default=N_FEW_SHOT,
                   help=f"Few-shot examples per class. Default: {N_FEW_SHOT}")
    p.add_argument("--max_chars",       type=int, default=MAX_CHARS,
                   help=f"Max chars of report text per prompt. Default: {MAX_CHARS}")
    p.add_argument("--resume",          action="store_true",
                   help="Skip rows already present in --output (resume interrupted run)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load source data ───────────────────────────────────────────────────
    df = load_source_excel(args.source_excel)

    # ── Resume: skip already-processed rows ───────────────────────────────
    if args.resume and os.path.isfile(args.output):
        done = pd.read_excel(args.output, sheet_name="Row_Level")
        done_ids = set(done["row_id"].tolist())
        df = df[~df["row_id"].isin(done_ids)].reset_index(drop=True)
        print(f"  Resume mode: {len(done_ids):,} already done, {len(df):,} remaining")

    # ── Sample ────────────────────────────────────────────────────────────
    if args.n_test:
        df = df.sample(n=min(args.n_test, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Smoke test: {len(df)} reports")

    # ── Load expert annotations for few-shot ──────────────────────────────
    expert_df    = load_expert_annotations(args.expert_excel)
    few_shot_block = build_few_shot_block(expert_df, n_per_class=args.n_few_shot,
                                          max_chars=1500)
    print(f"\nFew-shot block: {args.n_few_shot} examples × {len(CLASSES)} classes "
          f"= up to {args.n_few_shot * len(CLASSES)} examples per prompt")

    # ── Load model ────────────────────────────────────────────────────────
    if args.backend == "local":
        print("\nBackend: HuggingFace LOCAL  (faster — recommended for full runs)")
        backend = LocalMedGemma(
            local_model_dir=args.local_model_dir,
            use_4bit=not args.no_4bit,
        )
    else:
        print("\nBackend: OLLAMA  (simpler setup, ~2–4× slower than local)")
        backend = OllamaMedGemma(model=args.ollama_model)

    # ── Inference ─────────────────────────────────────────────────────────
    result_df = run_inference(df, backend, few_shot_block,
                              output_path=args.output,
                              checkpoint_freq=CHECKPOINT_FREQ)

    # ── If resuming, merge with previous results ──────────────────────────
    if args.resume and os.path.isfile(args.output):
        prev = pd.read_excel(args.output, sheet_name="Row_Level")
        result_df = pd.concat([prev, result_df], ignore_index=True)
        result_df.sort_values("row_id", inplace=True)

    # ── Save ──────────────────────────────────────────────────────────────
    save_results(result_df, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()

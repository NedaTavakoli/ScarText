"""The paper's practical workflow: rules on every report, an LLM as a check, experts review only disagreements."""

from .rules import classify_scar, extract_lvef


def compare(rule_label, llm_label):
    """'agree' (accept the class), 'review' (send to an expert), or 'rules only' when no LLM answer is available.

    In the paper, where the rules and zero-shot Qwen3-32B agreed (85.6% of 264 expert-labeled reports),
    97.8% were correct, and the disagreements held 21 of the 26 rule errors.
    """
    if llm_label is None:
        return "rules only"
    if rule_label is None or rule_label != llm_label:
        return "review"
    return "agree"


def analyze_csv(path, text_column, out_path=None, model=None, examples=None, host=None):
    """Classify every report in a CSV column and add `scar_rules`, `lvef`, and, with a model, `scar_llm` and `decision`.

    Returns the DataFrame, and also writes it when out_path is given. Keep files with patient data on approved computers.
    """
    import pandas as pd
    df = pd.read_csv(path)
    df["scar_rules"] = df[text_column].map(classify_scar)
    df["lvef"] = df[text_column].map(extract_lvef)
    if model:
        from .llm import DEFAULT_HOST, classify_llm
        df["scar_llm"] = [classify_llm(t, model=model, examples=examples, host=host or DEFAULT_HOST) for t in df[text_column]]
        df["decision"] = [compare(a, b) for a, b in zip(df["scar_rules"], df["scar_llm"])]
    if out_path:
        df.to_csv(out_path, index=False)
    return df

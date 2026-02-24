#!/usr/bin/env python3
"""
LLM baseline for PPA subtype classification from transcript (Phan et al. Interspeech 2024 style).

Given transcript text, call an LLM to classify as control / svPPA / nfvPPA / lvPPA,
or binary svPPA vs (nfv+lv). Serves as baseline vs our S_ph/S_sem pipeline.

Ref: Phan et al. (2024) Interspeech - using LLM for content assessment / classification.
     https://www.isca-archive.org/interspeech_2024/phan24_interspeech.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.phoneme_adequacy import parse_pipeline_output

BINARY_PROMPT = """You are a clinical linguist. Below is a speech transcript from a picture description task (picnic scene). The speaker may have primary progressive aphasia (PPA). 

Classify the transcript into exactly ONE of:
- svPPA (semantic variant: word-finding difficulty, empty speech, semantic errors)
- nflv (non-semantic PPA: nfvPPA or lvPPA: grammar/speech rate/phonological issues)

Reply with only one token: svPPA or nflv"""

MULTICLASS_PROMPT = """You are a clinical linguist. Below is a speech transcript from a picture description task. The speaker may have primary progressive aphasia (PPA) or be a control.

Classify into exactly ONE of: control, svPPA, nfvPPA, lvPPA.

Reply with only one word: control, svPPA, nfvPPA, or lvPPA."""


def get_transcript_from_pipeline(pipeline_path: Path) -> str:
    """Extract participant transcript from pipeline output file."""
    parsed = parse_pipeline_output(pipeline_path)
    return (parsed.get("transcript") or "").strip()


def call_llm(
    text: str,
    prompt_template: str,
    task: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """Call OpenAI API for classification. Requires OPENAI_API_KEY.

    Returns:
      - binary: svPPA | nflv | unknown
      - multiclass: control | svPPA | nfvPPA | lvPPA | unknown
    """
    try:
        import openai
    except ImportError:
        return "unknown"
    client = openai.OpenAI(api_key=api_key or __import__("os").environ.get("OPENAI_API_KEY", ""))
    if not client.api_key:
        return "unknown"
    user = f"Transcript:\n\n{text[:4000]}"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user},
            ],
            max_tokens=20,
        )
        raw = (r.choices[0].message.content or "").strip()
    except Exception:
        return "unknown"

    # Prefer the first alphabetic token (prompt asks for a single token/word).
    raw_up = raw.upper()
    m = re.search(r"[A-Z]+", raw_up)
    token = m.group(0) if m else raw_up.strip()

    if task == "binary":
        if token == "SVPPA" or "SVPPA" in raw_up or "SEMANTIC" in raw_up:
            return "svPPA"
        if token == "NFLV" or "NON-SEMANTIC" in raw_up:
            return "nflv"
        return "unknown"

    # multiclass
    if token == "CONTROL" or "CONTROL" in raw_up:
        return "control"
    if token == "SVPPA" or "SVPPA" in raw_up:
        return "svPPA"
    if token == "NFVPPA" or "NFVPPA" in raw_up:
        return "nfvPPA"
    if token == "LVPPA" or "LVPPA" in raw_up:
        return "lvPPA"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="LLM baseline for PPA classification from transcript")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--pipeline-dir", type=Path, default=Path("data/ppa_pipeline_outputs"))
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary",
                        help="binary: svPPA vs nflv. multiclass: control/svPPA/nfvPPA/lvPPA")
    parser.add_argument("--multiclass", action="store_true", help="Same as --task multiclass")
    parser.add_argument("--output", type=Path, default=Path("outputs/llm_baseline_predictions.csv"))
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None, help="Max samples (for testing)")
    args = parser.parse_args()
    if args.multiclass:
        args.task = "multiclass"

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    prompt = BINARY_PROMPT if args.task == "binary" else MULTICLASS_PROMPT
    api_key = __import__("os").environ.get("OPENAI_API_KEY")

    rows = []
    for _, row in df.iterrows():
        stem = Path(row["audio_path"]).stem
        cand = args.pipeline_dir / f"{stem}_feature_extraction.txt"
        if not cand.exists():
            continue
        transcript = get_transcript_from_pipeline(cand)
        if not transcript or len(transcript) < 20:
            continue
        pred = call_llm(transcript, prompt, task=args.task, api_key=api_key, model=args.model)
        subtype = (row.get("subtype") or "").strip().lower()
        if args.task == "binary":
            true_binary = 1 if subtype == "svppa" else (0 if subtype in ("nfvppa", "lvppa") else -1)
            pred_binary = 1 if pred == "svPPA" else (0 if pred == "nflv" else -1)
        else:
            true_binary = pred_binary = -1
        rows.append({
            "subject_id": row["subject_id"],
            "session": row.get("session", ""),
            "subtype": subtype,
            "label": row.get("label"),
            "prediction": pred,
            "true_binary": true_binary,
            "pred_binary": pred_binary,
        })

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    if args.task == "binary":
        valid = out_df[(out_df["true_binary"] >= 0) & (out_df["pred_binary"] >= 0)]
        if len(valid) > 0:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(valid["true_binary"], valid["pred_binary"])
            f1 = f1_score(valid["true_binary"], valid["pred_binary"], zero_division=0)
            print(f"LLM baseline (sv vs nflv): n={len(valid)} Acc={acc:.3f} F1={f1:.3f}")
        else:
            print("LLM baseline: no valid predictions (set OPENAI_API_KEY for OpenAI API)")
    else:
        # multiclass metrics
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        label_map = {"control": "control", "svppa": "svPPA", "nfvppa": "nfvPPA", "lvppa": "lvPPA"}
        y_true = out_df["subtype"].astype(str).str.strip().str.lower().map(label_map)
        y_pred = out_df["prediction"].astype(str).str.strip()
        labels = ["control", "svPPA", "nfvPPA", "lvPPA"]
        mask = y_true.isin(labels) & y_pred.isin(labels)
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            macro_f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            print(f"LLM baseline (multiclass): n={int(mask.sum())} Acc={acc:.3f} Macro-F1={macro_f1:.3f}")
            print(classification_report(y_true[mask], y_pred[mask], labels=labels, digits=3, zero_division=0))
        else:
            print("LLM baseline (multiclass): no valid predictions (set OPENAI_API_KEY for OpenAI API)")
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()

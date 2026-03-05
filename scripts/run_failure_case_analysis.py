#!/usr/bin/env python3
"""
Failure Case Analysis for the paper.
Selects 1 correctly classified and 1 misclassified example (Control vs svPPA),
and produces a report with transcript, proxy/L2 features, prediction, and scene graph comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.phoneme_adequacy import parse_pipeline_output
from src.features.graph_semantic import get_patient_profile


def build_transcript_dict(parsed: dict) -> dict:
    """Build transcript dict for get_patient_profile."""
    segments = parsed.get("segments", [])
    words_list = []
    full_text_parts = []
    for seg in segments:
        if "Participant" not in str(seg.get("speaker", "")):
            continue
        transcript = (seg.get("transcript") or "").strip()
        if transcript.lower().startswith("s ") and len(transcript) > 2:
            transcript = transcript[2:]
        full_text_parts.append(transcript)
        for w in transcript.split():
            words_list.append({"word": w, "start": 0.0, "end": 0.1})
    text = " ".join(full_text_parts)
    return {"text": text, "words": words_list}


def main():
    ap = argparse.ArgumentParser(description="Failure case analysis")
    ap.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_full.csv"))
    ap.add_argument("--layer2", type=Path, default=Path("outputs/layer2_fixed.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    ap.add_argument("--pipeline-dir", type=Path, default=Path("data/ppa_pipeline_outputs"))
    ap.add_argument("--gold-graph", type=Path, default=Path("data/graph/picnic_gold_star.json"))
    ap.add_argument("--output", type=Path, default=Path("outputs/FAILURE_CASE_ANALYSIS.md"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df_main = pd.read_csv(args.features)
    df_l2 = pd.read_csv(args.layer2)
    manifest = pd.read_csv(args.manifest)

    # Merge
    df = df_main.merge(
        df_l2,
        on=["subject_id", "session"],
        how="inner",
        suffixes=("", "_l2"),
    )
    df = df[[c for c in df.columns if not c.endswith("_l2")]]

    # Control vs svPPA only
    df = df[df["subtype"].isin(["control", "svppa"])].copy()
    df = df[df["s_ph"].notna() & df["s_sem"].notna()].copy()
    if len(df) < 10:
        print("Not enough Control/svPPA samples")
        return

    feat_cols = ["s_ph", "s_sem"]
    X = df[feat_cols].values
    y = (df["subtype"] == "svppa").astype(int).values
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000, random_state=args.seed, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(clf, X_sc, y, cv=cv)

    df = df.reset_index(drop=True)
    df["pred"] = y_pred
    df["correct"] = (df["pred"] == y)

    # Select examples
    correct_examples = df[df["correct"]].groupby("subtype").first()
    wrong_examples = df[~df["correct"]].groupby("subtype").first()

    case_correct = None
    case_wrong = None
    if len(correct_examples) > 0:
        row = correct_examples.iloc[0]
        idx = df[(df["subject_id"] == row["subject_id"]) & (df["session"] == row["session"])].index[0]
        case_correct = df.loc[idx]
    if len(wrong_examples) > 0:
        row = wrong_examples.iloc[0]
        idx = df[(df["subject_id"] == row["subject_id"]) & (df["session"] == row["session"])].index[0]
        case_wrong = df.loc[idx]

    def write_case(s, case, label: str):
        sid = case["subject_id"]
        ses = case["session"]
        subtype = case["subtype"]
        pred_label = "svPPA" if case["pred"] == 1 else "control"
        s.append(f"\n### {label}\n")
        s.append(f"- **Subject:** {sid}, **Session:** {ses}\n")
        s.append(f"- **True label:** {subtype}\n")
        s.append(f"- **Prediction:** {pred_label}\n")

        # Transcript
        manifest_row = manifest[(manifest["subject_id"] == sid) & (manifest["session"] == ses)]
        if len(manifest_row) > 0:
            apath = manifest_row["audio_path"].iloc[0]
            stem = Path(apath).stem
            pipeline_file = args.pipeline_dir / f"{stem}_feature_extraction.txt"
            if pipeline_file.exists():
                parsed = parse_pipeline_output(pipeline_file)
                transcript_dict = build_transcript_dict(parsed)
                text = transcript_dict["text"]
                excerpt = text[:500] + ("..." if len(text) > 500 else "")
                s.append(f"\n**Transcript excerpt:**\n\n> {excerpt}\n")
                # Patient profile
                profile = get_patient_profile(transcript_dict, args.gold_graph, s_ph=float(case["s_ph"]), layer2_scores=None)
                s.append(f"\n**Scene graph comparison:**\n")
                s.append(f"- Entities mentioned: {', '.join(profile['entities_mentioned'][:15])}{'...' if len(profile['entities_mentioned']) > 15 else ''}\n")
                s.append(f"- Entities missed: {', '.join(profile['entities_missed'][:10])}{'...' if len(profile['entities_missed']) > 10 else ''}\n")
                s.append(f"- Content units covered: {profile['content_units_covered']}\n")
                s.append(f"- Content units missed: {profile['content_units_missed']}\n")
                s.append(f"- S_sem: {profile['s_sem']:.3f}, entity recall: {profile['s_sem_nodes']:.3f}, CU coverage: {profile['s_sem_cu']:.3f}\n")

        # Proxy / adequacy scores
        s.append(f"\n**Proxy/Adequacy scores:**\n")
        s.append(f"- S_ph: {case['s_ph']:.3f}, S_sem: {case['s_sem']:.3f}, Δ: {case['delta']:.3f}\n")
        s.append(f"- phoneme_error_rate: {case['phoneme_error_rate']:.3f}, silence_ratio: {case['silence_ratio']:.3f}\n")

        # L2 scores (if available)
        l2_cols = [c for c in df.columns if c.startswith("l2_")]
        if l2_cols:
            s.append(f"\n**Clinical (Layer 2) scores:**\n")
            for c in l2_cols[:8]:
                if c in case and pd.notna(case[c]):
                    s.append(f"- {c}: {case[c]:.3f}\n")

    lines = [
        "# 6. Failure Case Analysis",
        "",
        "We select one correctly classified and one misclassified example from the Control vs. svPPA task (S_ph + S_sem logistic regression, 5-fold CV).",
        "",
    ]

    if case_correct is not None:
        write_case(lines, case_correct, "Case A: Correctly Classified")
    if case_wrong is not None:
        write_case(lines, case_wrong, "Case B: Misclassified")
        lines.append("\n**Interpretation of failure:**\n")
        true_sub = case_wrong["subtype"]
        pred_sub = "svPPA" if case_wrong["pred"] == 1 else "control"
        s_sem = float(case_wrong["s_sem"])
        s_ph = float(case_wrong["s_ph"])
        if true_sub == "control" and pred_sub == "svPPA":
            lines.append(f"The control was misclassified as svPPA. S_sem = {s_sem:.3f}, S_ph = {s_ph:.3f}. Low semantic adequacy (scene graph coverage, entity/CU recall) drives the model toward svPPA. Contributors: brief or disfluent transcript, ASR errors, or sparse content relative to the gold graph.\n")
        else:
            lines.append(f"The svPPA case was misclassified as control. S_sem = {s_sem:.3f}, S_ph = {s_ph:.3f}. Atypically high scores may reflect early-stage disease, preserved naming, or ASR artefacts.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()

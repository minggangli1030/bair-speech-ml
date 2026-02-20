"""CLI entrypoints: preprocess, extract, train, predict."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

# Add project root for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig
from src.features.graph_semantic import compute_s_sem
from src.features.layer1 import LAYER1_NAMES, compute_layer1_features
from src.features.layer2 import LAYER2_NAMES, compute_layer2_features
from src.io.cache import (
    load_cached_phonetic,
    load_cached_transcript,
    save_phonetic,
    save_transcript,
)
from src.io.dataset import load_manifest, iter_subjects
from src.models.asr import get_asr
from src.models.phonetic import get_phonetic_encoder
from src.train.baseline import (
    load_trained_pipeline,
    predict_and_explain,
    save_trained_pipeline,
    stratified_train_val_test_split,
    train_baseline,
    train_baseline_with_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="svPPA speech analysis pipeline")


@app.command()
def preprocess(
    manifest: Path = typer.Option(Path("data/manifest.csv"), "--manifest", "-m", help="Manifest CSV"),
    cache_dir: Path = typer.Option(Path("data/cache"), "--cache-dir", help="Cache directory"),
    audio_dir: Optional[Path] = typer.Option(None, "--audio-dir", help="Base dir for audio paths"),
    asr_backend: str = typer.Option("whisper", "--asr", help="ASR backend"),
    whisper_model: str = typer.Option("base", "--whisper-model", help="Whisper model size"),
    language: Optional[str] = typer.Option(None, "--language", help="ASR language code"),
    phonetic_encoder: Optional[str] = typer.Option(None, "--phonetic-encoder", help="Phonetic encoder path (module:Class)"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing", help="Skip if transcript/phonetic cache exists"),
) -> None:
    """
    Load audio, run ASR and phonetic encoder, save transcripts and phonetics to cache.
    """
    cfg = PipelineConfig()
    cfg.paths.manifest_path = manifest
    cfg.paths.cache_dir = cache_dir
    if audio_dir is not None:
        cfg.paths.audio_dir = audio_dir
    cfg.asr.backend = asr_backend
    cfg.asr.whisper_model = whisper_model
    cfg.asr.language = language
    if phonetic_encoder:
        cfg.phonetic.encoder_path = phonetic_encoder

    df = load_manifest(cfg.paths.manifest_path, cfg.paths.audio_dir)
    asr = get_asr(
        backend=cfg.asr.backend,
        model_name=cfg.asr.whisper_model,
        device=cfg.asr.device,
        **cfg.asr.extra,
    )
    phonetic_enc = get_phonetic_encoder(
        encoder_path=cfg.phonetic.encoder_path,
        use_stub_if_unavailable=cfg.phonetic.use_stub_if_unavailable,
    )

    for subject_id, group, _ in iter_subjects(df):
        for _, row in group.iterrows():
            path = row["audio_path_resolved"]
            path_str = str(path)
            if skip_existing:
                t = load_cached_transcript(cache_dir, str(subject_id), path_str)
                if t is not None:
                    logger.info("Skip (cached) ASR: %s %s", subject_id, path_str)
                    if cfg.phonetic.enabled:
                        p = load_cached_phonetic(cache_dir, str(subject_id), path_str)
                        if p is not None:
                            logger.info("Skip (cached) phonetic: %s %s", subject_id, path_str)
                        else:
                            out = phonetic_enc.encode(audio_path=path, text=t.get("text"))
                            save_phonetic(cache_dir, str(subject_id), path_str, out)
                    continue
            if not path.exists():
                logger.warning("Missing audio: %s", path)
                continue
            logger.info("ASR: %s %s", subject_id, path_str)
            transcript = asr.transcribe(path, language=cfg.asr.language)
            save_transcript(cache_dir, str(subject_id), path_str, transcript)
            if cfg.phonetic.enabled:
                out = phonetic_enc.encode(audio_path=path, text=transcript.get("text"))
                save_phonetic(cache_dir, str(subject_id), path_str, out)
    logger.info("Preprocess done.")


@app.command()
def extract(
    manifest: Path = typer.Option(Path("data/manifest.csv"), "--manifest", "-m"),
    cache_dir: Path = typer.Option(Path("data/cache"), "--cache-dir"),
    output: Path = typer.Option(Path("outputs/features.csv"), "--output", "-o"),
    audio_dir: Optional[Path] = typer.Option(None, "--audio-dir"),
    use_semantic_distance: bool = typer.Option(True, "--semantic/--no-semantic"),
    use_graph_semantic: bool = typer.Option(True, "--graph-semantic/--no-graph-semantic", help="S_sem from spatiosemantic scene graph"),
    gold_graph: Optional[Path] = typer.Option(None, "--gold-graph", help="Path to gold scene graph JSON (default: data/graph/picnic_gold_star.json)"),
    spacy_model: str = typer.Option("en_core_web_sm", "--spacy-model"),
) -> None:
    """
    Load cached transcripts/phonetics, compute Layer 1 and Layer 2 features, aggregate per subject, write CSV.
    """
    cfg = PipelineConfig()
    cfg.paths.manifest_path = manifest
    cfg.paths.cache_dir = cache_dir
    cfg.paths.output_dir = output.parent
    if audio_dir:
        cfg.paths.audio_dir = audio_dir
    cfg.features.use_semantic_distance = use_semantic_distance
    cfg.features.use_graph_semantic = use_graph_semantic
    cfg.features.gold_graph_path = gold_graph
    cfg.features.spacy_model = spacy_model

    df = load_manifest(cfg.paths.manifest_path, cfg.paths.audio_dir)
    rows = []
    for subject_id, group, label in iter_subjects(df):
        l1_agg: list[dict[str, float]] = []
        s_sem_agg: list[float] = []
        for _, row in group.iterrows():
            path = row["audio_path_resolved"]
            path_str = str(path)
            transcript = load_cached_transcript(cache_dir, str(subject_id), path_str)
            if transcript is None:
                logger.warning("No transcript for %s %s", subject_id, path_str)
                continue
            phonetic = load_cached_phonetic(cache_dir, str(subject_id), path_str)
            expected = row.get("expected_entities")
            if isinstance(expected, str):
                expected = [e.strip() for e in expected.split(",")] if expected else None
            else:
                expected = None
            l1 = compute_layer1_features(
                transcript,
                phonetic_data=phonetic or {},
                expected_entities=expected,
                spacy_model=cfg.features.spacy_model,
                use_semantic_distance=cfg.features.use_semantic_distance,
            )
            l1_agg.append(l1)
            if cfg.features.use_graph_semantic:
                s_sem_val, _ = compute_s_sem(transcript, gold_graph_path=cfg.features.gold_graph_path)
                s_sem_agg.append(s_sem_val)
        if not l1_agg:
            logger.warning("No features for subject %s", subject_id)
            continue
        # Aggregate: mean across recordings
        l1_mean = {k: pd.Series([d[k] for d in l1_agg]).mean() for k in LAYER1_NAMES}
        raw_z, l2_scores = compute_layer2_features(
            l1_mean,
            layer2_weights=cfg.features.layer2_weights,
        )
        out_row = {"subject_id": subject_id, "label": label}
        for k in LAYER1_NAMES:
            out_row[k] = l1_mean[k]
        for k in LAYER2_NAMES:
            out_row[k] = l2_scores[k]
        # S_sem: spatiosemantic graph completeness (separate branch, not Layer 1)
        s_sem = float(pd.Series(s_sem_agg).mean()) if s_sem_agg else float("nan")
        out_row["s_sem"] = s_sem
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    logger.info("Wrote %s with %d subjects.", output, len(out_df))


@app.command()
def train(
    features_csv: Path = typer.Option(Path("outputs/features.csv"), "--features", "-f"),
    model_dir: Path = typer.Option(Path("outputs/models"), "--model-dir"),
    target: str = typer.Option("label", "--target"),
    use_layer1: bool = typer.Option(False, "--layer1/--no-layer1"),
    use_layer2: bool = typer.Option(True, "--layer2/--no-layer2"),
    use_s_sem: bool = typer.Option(True, "--s-sem/--no-s-sem", help="Include s_sem (spatiosemantic graph) in features"),
    cv_folds: int = typer.Option(5, "--cv-folds"),
    report_path: Optional[Path] = typer.Option(None, "--report", help="Write report JSON here"),
    split: Optional[str] = typer.Option(
        None,
        "--split",
        help="Train/val/test ratios, e.g. '0.6,0.2,0.2'. If set, fit on train only and report val/test metrics.",
    ),
    splits_dir: Optional[Path] = typer.Option(
        None,
        "--splits-dir",
        help="Directory to save train/val/test CSV splits (subject_id list). Default: model_dir/splits",
    ),
) -> None:
    """
    Train logistic regression on Layer 2 (and optionally Layer 1), save model and report.
    Use --split 0.6,0.2,0.2 for stratified train/val/test when you have limited data.
    """
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")
    df = pd.read_csv(features_csv)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    if split is not None:
        parts = [float(x.strip()) for x in split.split(",")]
        if len(parts) != 3:
            raise ValueError("--split must be three numbers, e.g. 0.6,0.2,0.2")
        train_ratio, val_ratio, test_ratio = parts
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        clf, scaler, feat_names, report, train_df, val_df, test_df = train_baseline_with_splits(
            df,
            target_column=target,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_layer1=use_layer1,
            use_layer2=use_layer2,
            use_s_sem=use_s_sem,
        )
        save_trained_pipeline(Path(model_dir), clf, scaler, feat_names, report)
        out_splits = Path(splits_dir) if splits_dir else Path(model_dir) / "splits"
        out_splits.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_splits / "train.csv", index=False)
        val_df.to_csv(out_splits / "val.csv", index=False)
        test_df.to_csv(out_splits / "test.csv", index=False)
        logger.info("Saved splits to %s (train=%d, val=%d, test=%d)", out_splits, len(train_df), len(val_df), len(test_df))
        print("Train/Val/Test split report:")
        print(f"  Train:  n={report['n_train']}  accuracy={report['train_accuracy']:.4f}  F1={report['train_f1']:.4f}  ROC-AUC={report['train_roc_auc']:.4f}")
        print(f"  Val:    n={report['n_val']}  accuracy={report['val_accuracy']:.4f}  F1={report['val_f1']:.4f}  ROC-AUC={report['val_roc_auc']:.4f}")
        print(f"  Test:   n={report['n_test']}  accuracy={report['test_accuracy']:.4f}  F1={report['test_f1']:.4f}  ROC-AUC={report['test_roc_auc']:.4f}")
        print("  Val confusion matrix:", report["val_confusion_matrix"])
        print("  Test confusion matrix:", report["test_confusion_matrix"])
        print("  Top + coef:", report["top_positive_coefficients"][:5])
        print("  Top - coef:", report["top_negative_coefficients"][:5])
    else:
        clf, scaler, feat_names, report = train_baseline(
            df,
            target_column=target,
            use_layer1=use_layer1,
            use_layer2=use_layer2,
            use_s_sem=use_s_sem,
            cv_folds=cv_folds,
        )
        save_trained_pipeline(Path(model_dir), clf, scaler, feat_names, report)
        print("Training report:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  F1:       {report['f1']:.4f}")
        print(f"  ROC-AUC:  {report['roc_auc']:.4f}")
        print(f"  CV ROC-AUC: {report['cv_roc_auc_mean']:.4f} ± {report['cv_roc_auc_std']:.4f}")
        print("  Confusion matrix:", report["confusion_matrix"])
        print("  Top + coef:", report["top_positive_coefficients"][:5])
        print("  Top - coef:", report["top_negative_coefficients"][:5])

    if report_path:
        import json
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote report: %s", report_path)


@app.command()
def predict(
    features_csv: Path = typer.Option(Path("outputs/features.csv"), "--features", "-f"),
    model_dir: Path = typer.Option(Path("outputs/models"), "--model-dir"),
    output: Path = typer.Option(Path("outputs/predictions.csv"), "--output", "-o"),
    top_k: int = typer.Option(5, "--top-k", help="Top contributing features per subject"),
) -> None:
    """
    Load trained pipeline and feature CSV, output per-subject probabilities and explanation.
    """
    clf, scaler, feat_names, report, medians = load_trained_pipeline(Path(model_dir))
    df = pd.read_csv(features_csv)
    cols = [c for c in feat_names if c in df.columns]
    if len(cols) != len(feat_names):
        missing = set(feat_names) - set(cols)
        raise ValueError(f"Feature CSV missing columns: {missing}")

    results = []
    for _, row in df.iterrows():
        x = row[feat_names].to_numpy(dtype=float)
        prob, top_contrib, _ = predict_and_explain(
            clf, scaler, feat_names, x, top_k=top_k, feature_medians=medians
        )
        results.append({
            "subject_id": row.get("subject_id", ""),
            "prob_svPPA": prob,
            "predicted": 1 if prob >= 0.5 else 0,
            "top_contributing": str(top_contrib),
        })
    out_df = pd.DataFrame(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    logger.info("Wrote %s", output)
    for r in results[:5]:
        print(r)
    if len(results) > 5:
        print("...")


if __name__ == "__main__":
    app()

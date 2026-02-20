"""Logistic regression baseline: training, evaluation, and per-subject prediction with explainability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / val / test by label.
    train_ratio + val_ratio + test_ratio should equal 1.0.
    Returns (train_df, val_df, test_df).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    df = df.dropna(subset=[target_column]).copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])
    y = df[target_column].astype(int)
    # First split: train+val vs test
    test_frac = test_ratio / (train_ratio + val_ratio + test_ratio)
    train_val, test_df = train_test_split(
        df, test_size=test_ratio, stratify=y, random_state=random_state
    )
    # Second split: train vs val
    val_frac = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_frac,
        stratify=train_val[target_column],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def _feature_matrix(
    df: pd.DataFrame,
    layer1_names: list[str],
    layer2_names: list[str],
    use_layer1: bool,
    use_layer2: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build X matrix and feature name list from subject-level dataframe."""
    from ..features.layer1 import LAYER1_NAMES
    from ..features.layer2 import LAYER2_NAMES
    names = []
    if use_layer2:
        names.extend(LAYER2_NAMES)
    if use_layer1:
        names.extend(LAYER1_NAMES)
    cols = [c for c in names if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    # Impute NaN with column median for training
    med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        np.putmask(X[:, j], np.isnan(X[:, j]), med[j])
    return X, cols


def train_baseline(
    features_df: pd.DataFrame,
    target_column: str = "label",
    use_layer1: bool = False,
    use_layer2: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> tuple[LogisticRegression, StandardScaler, list[str], dict[str, Any]]:
    """
    Train logistic regression with StandardScaler and StratifiedKFold CV.
    Returns (model, scaler, feature_names, report_dict).
    """
    from ..features.layer1 import LAYER1_NAMES
    from ..features.layer2 import LAYER2_NAMES

    y = features_df[target_column].astype(int).to_numpy()
    X, feat_names = _feature_matrix(
        features_df, LAYER1_NAMES, LAYER2_NAMES, use_layer1, use_layer2
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(clf, Xs, y, cv=cv, scoring=["roc_auc", "accuracy", "f1"], return_train_score=False)
    y_pred_cv = cross_val_predict(clf, Xs, y, cv=cv)

    clf.fit(Xs, y)

    # Metrics
    y_pred = clf.predict(Xs)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="binary")
    try:
        roc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
    except Exception:
        roc = float("nan")
    cm = confusion_matrix(y, y_pred)

    # Coefficients
    coef = clf.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1]
    top_positive = [(feat_names[i], float(coef[i])) for i in order if coef[i] > 0][:10]
    top_negative = [(feat_names[i], float(coef[i])) for i in order if coef[i] < 0][:10]

    # Store medians for imputation at predict time
    medians = np.nanmedian(features_df[feat_names].to_numpy(dtype=float), axis=0).tolist()

    report = {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(roc),
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "confusion_matrix": cm.tolist(),
        "top_positive_coefficients": top_positive,
        "top_negative_coefficients": top_negative,
        "feature_names": feat_names,
        "coefficients": {feat_names[i]: float(coef[i]) for i in range(len(feat_names))},
        "use_layer1": use_layer1,
        "use_layer2": use_layer2,
        "feature_medians": medians,
    }
    return clf, scaler, feat_names, report


def train_baseline_with_splits(
    features_df: pd.DataFrame,
    target_column: str = "label",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    use_layer1: bool = False,
    use_layer2: bool = True,
    random_state: int = 42,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 1000,
) -> tuple[LogisticRegression, StandardScaler, list[str], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split, fit on train only, evaluate on val and test.
    Returns (clf, scaler, feat_names, report, train_df, val_df, test_df).
    """
    from ..features.layer1 import LAYER1_NAMES
    from ..features.layer2 import LAYER2_NAMES

    train_df, val_df, test_df = stratified_train_val_test_split(
        features_df, target_column, train_ratio, val_ratio, test_ratio, random_state
    )

    y_train = train_df[target_column].astype(int).to_numpy()
    X_train, feat_names = _feature_matrix(
        train_df, LAYER1_NAMES, LAYER2_NAMES, use_layer1, use_layer2
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    # Medians from train set for predict-time imputation (use for val/test too)
    medians = np.nanmedian(train_df[feat_names].to_numpy(dtype=float), axis=0).tolist()

    # Metrics on train, val, test (val/test imputed with train medians)
    def _metrics(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        X_imp = X.copy()
        for j in range(X_imp.shape[1]):
            np.putmask(X_imp[:, j], np.isnan(X_imp[:, j]), medians[j] if j < len(medians) else 0.0)
        Xs = scaler.transform(X_imp)
        y_pred = clf.predict(Xs)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="binary")
        try:
            roc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
        except Exception:
            roc = float("nan")
        return {"accuracy": acc, "f1": f1, "roc_auc": roc, "confusion_matrix": confusion_matrix(y, y_pred).tolist()}

    X_val = val_df[feat_names].to_numpy(dtype=float)
    for j in range(X_val.shape[1]):
        np.putmask(X_val[:, j], np.isnan(X_val[:, j]), medians[j])
    y_val = val_df[target_column].astype(int).to_numpy()
    X_test = test_df[feat_names].to_numpy(dtype=float)
    for j in range(X_test.shape[1]):
        np.putmask(X_test[:, j], np.isnan(X_test[:, j]), medians[j])
    y_test = test_df[target_column].astype(int).to_numpy()

    train_metrics = _metrics(X_train, y_train)
    val_metrics = _metrics(X_val, y_val)
    test_metrics = _metrics(X_test, y_test)

    coef = clf.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1]
    top_positive = [(feat_names[i], float(coef[i])) for i in order if coef[i] > 0][:10]
    top_negative = [(feat_names[i], float(coef[i])) for i in order if coef[i] < 0][:10]

    report = {
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_f1": float(train_metrics["f1"]),
        "train_roc_auc": float(train_metrics["roc_auc"]),
        "train_confusion_matrix": train_metrics["confusion_matrix"],
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_f1": float(val_metrics["f1"]),
        "val_roc_auc": float(val_metrics["roc_auc"]),
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "split": {"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio},
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "top_positive_coefficients": top_positive,
        "top_negative_coefficients": top_negative,
        "feature_names": feat_names,
        "coefficients": {feat_names[i]: float(coef[i]) for i in range(len(feat_names))},
        "use_layer1": use_layer1,
        "use_layer2": use_layer2,
        "feature_medians": medians,
    }
    return clf, scaler, feat_names, report, train_df, val_df, test_df


def evaluate_model(
    clf: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Compute metrics on given X, y (X raw, not scaled)."""
    Xs = scaler.transform(X)
    y_pred = clf.predict(Xs)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="binary")
    try:
        roc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
    except Exception:
        roc = float("nan")
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(roc),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def save_trained_pipeline(
    model_dir: Path,
    clf: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
    report: dict[str, Any],
) -> None:
    """Save model (joblib), scaler, and feature schema (JSON)."""
    import joblib
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "logistic_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    schema = {
        "feature_names": feature_names,
        "report": report,
        "feature_medians": report.get("feature_medians", []),
    }
    with open(model_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def load_trained_pipeline(
    model_dir: Path,
) -> tuple[LogisticRegression, StandardScaler, list[str], dict[str, Any], list[float]]:
    """Load model, scaler, schema, and feature medians from disk."""
    import joblib
    model_dir = Path(model_dir)
    clf = joblib.load(model_dir / "logistic_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    with open(model_dir / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    medians = schema.get("feature_medians", [])
    return clf, scaler, schema["feature_names"], schema.get("report", {}), medians


def predict_and_explain(
    clf: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
    X_row: np.ndarray,
    top_k: int = 5,
    feature_medians: list[float] | None = None,
) -> tuple[float, list[tuple[str, float]], np.ndarray]:
    """
    Per-subject prediction: probability and top contributing features.
    X_row: 1d array in same order as feature_names (raw, will be scaled).
    feature_medians: optional training medians for NaN imputation.
    Returns (prob_positive, [(feat_name, contribution), ...], scaled_vector).
    Contribution = coef_i * scaled_value_i (signed).
    """
    X = np.asarray(X_row, dtype=float).reshape(1, -1)
    if feature_medians is not None and len(feature_medians) == X.shape[1]:
        for j in range(X.shape[1]):
            if np.isnan(X[0, j]):
                X[0, j] = feature_medians[j]
    else:
        X = np.nan_to_num(X, nan=0.0)
    Xs = scaler.transform(X)
    prob = float(clf.predict_proba(Xs)[0, 1])
    coef = clf.coef_.ravel()
    contrib = coef * Xs.ravel()
    order = np.argsort(np.abs(contrib))[::-1]
    top = [(feature_names[i], float(contrib[i])) for i in order[:top_k]]
    return prob, top, Xs.ravel()

"""Training: logistic regression baseline and reporting."""

from .baseline import (
    evaluate_model,
    load_trained_pipeline,
    predict_and_explain,
    stratified_train_val_test_split,
    train_baseline,
    train_baseline_with_splits,
)

__all__ = [
    "evaluate_model",
    "load_trained_pipeline",
    "predict_and_explain",
    "stratified_train_val_test_split",
    "train_baseline",
    "train_baseline_with_splits",
]

"""Classification metrics: macro-F1, balanced accuracy, and per-class recall."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Counts with true classes as rows and predicted classes as columns."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]
) -> dict[str, float]:
    """Macro-F1, balanced accuracy, accuracy, and recall per class."""
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    support = matrix.sum(axis=1)
    if (support == 0).any():
        absent = [name for name, n in zip(class_names, support) if n == 0]
        raise ValueError(f"classes {absent} have no true examples; recall is undefined")
    hits = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0)
    recall = hits / support
    # a class that is never predicted has precision 0, not undefined
    precision = np.divide(hits, predicted, out=np.zeros_like(hits), where=predicted > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(hits),
        where=denominator > 0,
    )
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(hits.sum() / matrix.sum()),
        **{f"recall_{name}": float(r) for name, r in zip(class_names, recall)},
    }

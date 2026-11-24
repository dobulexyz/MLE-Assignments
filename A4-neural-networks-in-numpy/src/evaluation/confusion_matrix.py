"""Confusion matrix and derived per-class metrics."""

import numpy as np

from evaluation.metric import Metric


class ConfusionMatrix(Metric):
    """Accumulates a K×K confusion matrix over a pass.

    Entry [i, j] counts samples whose true class is i and whose
    predicted class is j. `compute` returns a copy so callers can
    mutate or normalise freely without affecting internal state.
    """

    name = "confusion_matrix"

    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._matrix = np.zeros(
            (self._num_classes, self._num_classes), dtype=np.int64
        )

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        ### BEGIN SOLUTION
        preds = outputs.argmax(axis=1)
        np.add.at(self._matrix, (targets, preds), 1)
        ### END SOLUTION

    def compute(self) -> np.ndarray:
        return self._matrix.copy()


def _per_class_precision(cm: np.ndarray) -> np.ndarray:
    ### BEGIN SOLUTION
    diag = np.diag(cm).astype(float)
    col_sums = cm.sum(axis=0).astype(float)
    out = np.zeros_like(diag)
    return np.divide(diag, col_sums, out=out, where=col_sums > 0)
    ### END SOLUTION


def _per_class_recall(cm: np.ndarray) -> np.ndarray:
    ### BEGIN SOLUTION
    diag = np.diag(cm).astype(float)
    row_sums = cm.sum(axis=1).astype(float)
    out = np.zeros_like(diag)
    return np.divide(diag, row_sums, out=out, where=row_sums > 0)
    ### END SOLUTION


def accuracy_from_matrix(cm: np.ndarray) -> float:
    """Overall accuracy: fraction of correctly classified samples."""
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float(np.trace(cm) / total)


def precision(
    cm: np.ndarray, class_label: int | None = None
) -> float | np.ndarray:
    """Per-class precision; 0.0 where column sum is zero."""
    per_class = _per_class_precision(cm)
    if class_label is not None:
        return float(per_class[class_label])
    return per_class


def recall(
    cm: np.ndarray, class_label: int | None = None
) -> float | np.ndarray:
    """Per-class recall; 0.0 where row sum is zero."""
    per_class = _per_class_recall(cm)
    if class_label is not None:
        return float(per_class[class_label])
    return per_class


def f1(cm: np.ndarray, class_label: int | None = None) -> float | np.ndarray:
    """Per-class F1; 0.0 where both precision and recall are zero."""
    ### BEGIN SOLUTION
    p = _per_class_precision(cm)
    r = _per_class_recall(cm)
    denom = p + r
    out = np.zeros_like(p)
    per_class: np.ndarray = np.divide(
        2 * p * r, denom, out=out, where=denom > 0
    )
    if class_label is not None:
        return float(per_class[class_label])
    return per_class
    ### END SOLUTION

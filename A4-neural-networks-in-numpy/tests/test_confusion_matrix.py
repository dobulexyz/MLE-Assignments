import numpy as np
import pytest

from evaluation.confusion_matrix import (
    ConfusionMatrix,
    accuracy_from_matrix,
    f1,
    precision,
    recall,
)

# Alias for test readability: the cm-based function was renamed.
accuracy = accuracy_from_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 3-class: 2 correct per class, no misclassifications
_CM_PERFECT = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)

# 2-class: [[TP0, FN0], [FP0, TP1]] = [[3, 1], [2, 4]]
# precision: [3/5, 4/5]  recall: [3/4, 4/6]  f1: [3/4.5, 4/5.5]
_CM_MIXED = np.array([[3, 1], [2, 4]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_confusion_matrix_smoke():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        np.array([0, 1]),
    )
    assert isinstance(cm.compute(), np.ndarray)


def test_accuracy_smoke():
    assert isinstance(accuracy(_CM_PERFECT), float)


def test_precision_smoke():
    result = precision(_CM_MIXED)
    assert isinstance(result, np.ndarray)


def test_recall_smoke():
    result = recall(_CM_MIXED)
    assert isinstance(result, np.ndarray)


def test_f1_smoke():
    result = f1(_CM_MIXED)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_name():
    assert ConfusionMatrix.name == "confusion_matrix"


def test_confusion_matrix_shape():
    cm = ConfusionMatrix(num_classes=4)
    cm.update(np.eye(4), np.arange(4))
    assert cm.compute().shape == (4, 4)


def test_confusion_matrix_perfect():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(np.eye(3), np.array([0, 1, 2]))
    np.testing.assert_array_equal(cm.compute(), np.eye(3, dtype=np.int64))


def test_confusion_matrix_all_wrong():
    cm = ConfusionMatrix(num_classes=2)
    cm.update(
        np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]),
        np.array([0, 0, 0]),
    )
    np.testing.assert_array_equal(
        cm.compute(), np.array([[0, 3], [0, 0]], dtype=np.int64)
    )


def test_confusion_matrix_known_entries():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(
        np.array([[0.1, 0.9, 0.0], [0.1, 0.8, 0.1], [0.9, 0.0, 0.1]]),
        np.array([0, 1, 2]),
    )
    expected = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(cm.compute(), expected)


def test_confusion_matrix_accumulates_across_batches():
    cm = ConfusionMatrix(num_classes=2)
    cm.update(np.array([[0.9, 0.1]]), np.array([0]))
    cm.update(np.array([[0.1, 0.9]]), np.array([1]))
    cm.update(np.array([[0.1, 0.9]]), np.array([0]))
    np.testing.assert_array_equal(
        cm.compute(), np.array([[1, 1], [0, 1]], dtype=np.int64)
    )


def test_confusion_matrix_reset():
    cm = ConfusionMatrix(num_classes=2)
    cm.update(np.array([[0.9, 0.1]]), np.array([0]))
    cm.reset()
    np.testing.assert_array_equal(
        cm.compute(), np.zeros((2, 2), dtype=np.int64)
    )


def test_confusion_matrix_compute_returns_copy():
    cm = ConfusionMatrix(num_classes=2)
    cm.update(np.eye(2), np.array([0, 1]))
    result = cm.compute()
    result[0, 0] = 999
    np.testing.assert_array_equal(cm.compute(), np.eye(2, dtype=np.int64))


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_perfect():
    assert accuracy(_CM_PERFECT) == pytest.approx(1.0)


def test_accuracy_all_wrong():
    cm = np.array([[0, 3], [0, 0]], dtype=np.int64)
    assert accuracy(cm) == pytest.approx(0.0)


def test_accuracy_known_value():
    # _CM_MIXED: 3+4=7 correct out of 10
    assert accuracy(_CM_MIXED) == pytest.approx(0.7)


def test_accuracy_empty_returns_nan():
    assert np.isnan(accuracy(np.zeros((2, 2), dtype=np.int64)))


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------


def test_precision_perfect():
    np.testing.assert_array_almost_equal(
        precision(_CM_PERFECT), [1.0, 1.0, 1.0]
    )


def test_precision_known_values():
    # col sums: [5, 5]; diag: [3, 4]
    np.testing.assert_array_almost_equal(precision(_CM_MIXED), [3 / 5, 4 / 5])


def test_precision_single_class():
    result = precision(_CM_MIXED, class_label=0)
    assert isinstance(result, float)
    assert result == pytest.approx(3 / 5)


def test_precision_zero_column():
    # class 1 never predicted → precision[1] = 0.0
    cm = np.array([[2, 0], [3, 0]], dtype=np.int64)
    assert precision(cm, class_label=1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def test_recall_perfect():
    np.testing.assert_array_almost_equal(recall(_CM_PERFECT), [1.0, 1.0, 1.0])


def test_recall_known_values():
    # row sums: [4, 6]; diag: [3, 4]
    np.testing.assert_array_almost_equal(recall(_CM_MIXED), [3 / 4, 4 / 6])


def test_recall_single_class():
    result = recall(_CM_MIXED, class_label=1)
    assert isinstance(result, float)
    assert result == pytest.approx(4 / 6)


def test_recall_zero_row():
    # class 1 has no true instances → recall[1] = 0.0
    cm = np.array([[2, 1], [0, 0]], dtype=np.int64)
    assert recall(cm, class_label=1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# f1
# ---------------------------------------------------------------------------


def test_f1_perfect():
    np.testing.assert_array_almost_equal(f1(_CM_PERFECT), [1.0, 1.0, 1.0])


def test_f1_known_values():
    p = np.array([3 / 5, 4 / 5])
    r = np.array([3 / 4, 4 / 6])
    expected = 2 * p * r / (p + r)
    np.testing.assert_array_almost_equal(f1(_CM_MIXED), expected)


def test_f1_single_class():
    result = f1(_CM_MIXED, class_label=0)
    assert isinstance(result, float)
    p, r = 3 / 5, 3 / 4
    assert result == pytest.approx(2 * p * r / (p + r))


def test_f1_both_zero():
    # class with no predictions and no true instances → f1 = 0.0
    cm = np.array([[0, 0], [0, 0]], dtype=np.int64)
    np.testing.assert_array_almost_equal(f1(cm), [0.0, 0.0])

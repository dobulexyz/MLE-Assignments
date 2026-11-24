import numpy as np
import pytest

from evaluation.accuracy import Accuracy, accuracy
from evaluation.metric import Metric, MetricCollection


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_accuracy_smoke():
    result = accuracy(np.array([0, 1]), np.array([0, 1]))
    assert isinstance(result, float)


def test_accuracy_metric_smoke():
    m = Accuracy()
    outputs = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = np.array([0, 1])
    m.update(outputs, targets)
    assert isinstance(m.compute(), float)


def test_metric_collection_smoke():
    col = MetricCollection([Accuracy()])
    outputs = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = np.array([0, 1])
    col.update(outputs, targets)
    result = col.compute()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# accuracy (array-based)
# ---------------------------------------------------------------------------


def test_accuracy_perfect_predictions():
    assert accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])) == 1.0


def test_accuracy_all_wrong():
    assert accuracy(np.array([1, 2, 0]), np.array([0, 1, 2])) == 0.0


def test_accuracy_half_correct():
    pred = np.array([0, 1, 1, 1])
    true = np.array([0, 1, 0, 0])
    assert accuracy(pred, true) == 0.5


def test_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0, 1, 2]))


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy(np.array([], dtype=int), np.array([], dtype=int))


# ---------------------------------------------------------------------------
# Metric (ABC)
# ---------------------------------------------------------------------------


def test_metric_is_abstract():
    with pytest.raises(TypeError):
        Metric()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Accuracy (Metric)
# ---------------------------------------------------------------------------


def test_accuracy_metric_name():
    assert Accuracy.name == "acc"


def test_accuracy_metric_perfect():
    m = Accuracy()
    outputs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = np.array([1, 0, 1])
    m.update(outputs, targets)
    assert m.compute() == 1.0


def test_accuracy_metric_all_wrong():
    m = Accuracy()
    outputs = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([0, 1])
    m.update(outputs, targets)
    assert m.compute() == 0.0


def test_accuracy_metric_half_correct():
    m = Accuracy()
    outputs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    targets = np.array([0, 1, 1, 0])
    m.update(outputs, targets)
    assert m.compute() == 0.5


def test_accuracy_metric_empty_returns_nan():
    m = Accuracy()
    assert np.isnan(m.compute())


def test_accuracy_metric_reset_clears_state():
    m = Accuracy()
    outputs = np.array([[0.9, 0.1], [0.1, 0.9]])
    targets = np.array([0, 1])
    m.update(outputs, targets)
    m.reset()
    assert np.isnan(m.compute())


def test_accuracy_metric_accumulates_across_batches():
    m = Accuracy()
    # batch 1: 2 correct out of 2
    m.update(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0, 1]))
    # batch 2: 0 correct out of 2
    m.update(np.array([[0.1, 0.9], [0.9, 0.1]]), np.array([0, 1]))
    # 2 correct out of 4 total
    assert m.compute() == 0.5


def test_accuracy_metric_weighted_batches():
    m = Accuracy()
    # batch of 1: correct
    m.update(np.array([[0.9, 0.1]]), np.array([0]))
    # batch of 3: all wrong
    m.update(
        np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]),
        np.array([0, 0, 0]),
    )
    # 1 correct out of 4 → 0.25 (not simple average of 1.0 and 0.0)
    assert m.compute() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# MetricCollection
# ---------------------------------------------------------------------------


def test_metric_collection_compute_keys():
    col = MetricCollection([Accuracy()])
    col.update(np.array([[0.9, 0.1]]), np.array([0]))
    assert set(col.compute().keys()) == {"acc"}


def test_metric_collection_compute_value():
    col = MetricCollection([Accuracy()])
    col.update(np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([0, 1]))
    assert col.compute()["acc"] == 1.0


def test_metric_collection_reset():
    col = MetricCollection([Accuracy()])
    col.update(np.array([[0.9, 0.1]]), np.array([0]))
    col.reset()
    assert np.isnan(col.compute()["acc"])


def test_metric_collection_multiple_metrics():
    class ConstantMetric(Metric):
        name = "const"

        def reset(self) -> None:
            pass

        def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
            pass

        def compute(self) -> float:
            return 42.0

    col = MetricCollection([Accuracy(), ConstantMetric()])
    col.update(np.array([[0.9, 0.1]]), np.array([0]))
    result = col.compute()
    assert set(result.keys()) == {"acc", "const"}
    assert result["const"] == 42.0


def test_metric_collection_empty():
    col = MetricCollection([])
    col.update(np.array([[0.9, 0.1]]), np.array([0]))
    assert col.compute() == {}

import math

import numpy as np
import pytest

from nn.loss import CrossEntropy


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_cross_entropy_forward_smoke():
    loss = CrossEntropy()
    value = loss.forward(np.zeros((2, 3)), np.array([0, 1]))
    assert isinstance(value, float)


def test_cross_entropy_backward_smoke():
    loss = CrossEntropy()
    loss.forward(np.zeros((2, 3)), np.array([0, 1]))
    grad = loss.backward()
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 3)


# ---------------------------------------------------------------------------
# CrossEntropy.forward
# ---------------------------------------------------------------------------


def test_cross_entropy_uniform_logits_equals_log_C():
    loss = CrossEntropy()
    value = loss.forward(np.zeros((4, 5)), np.array([0, 1, 2, 3]))
    assert value == pytest.approx(math.log(5.0))


def test_cross_entropy_perfect_prediction_near_zero():
    loss = CrossEntropy()
    logits = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
    value = loss.forward(logits, np.array([0, 2]))
    assert value < 1e-6


def test_cross_entropy_numerical_stability():
    loss = CrossEntropy()
    logits = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    value = loss.forward(logits, np.array([0, 1]))
    assert math.isfinite(value)


# ---------------------------------------------------------------------------
# CrossEntropy.backward
# ---------------------------------------------------------------------------


def test_cross_entropy_backward_sums_to_zero_per_row():
    loss = CrossEntropy()
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, -0.5]])
    loss.forward(logits, np.array([0, 2]))
    grad = loss.backward()
    assert np.allclose(grad.sum(axis=1), 0.0)


def test_cross_entropy_backward_matches_softmax_minus_onehot():
    loss = CrossEntropy()
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    y = np.array([2, 0])
    loss.forward(logits, y)
    grad = loss.backward()

    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    probs = e / e.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(2), y] = 1.0
    expected = (probs - one_hot) / 2

    assert np.allclose(grad, expected)


def test_cross_entropy_backward_without_forward_raises():
    loss = CrossEntropy()
    with pytest.raises(RuntimeError):
        loss.backward()


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------


from nn.loss import MSE  # noqa: E402


def test_mse_forward_known_value():
    mse = MSE()
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [1.0, 1.0]])
    # diffs: [[0,1],[2,3]]; squared: [[0,1],[4,9]]; mean = 14/4 = 3.5
    assert mse.forward(pred, target) == pytest.approx(3.5)


def test_mse_forward_returns_float():
    mse = MSE()
    value = mse.forward(np.zeros((3, 2)), np.ones((3, 2)))
    assert isinstance(value, float)


def test_mse_backward_known_gradient():
    mse = MSE()
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [1.0, 1.0]])
    mse.forward(pred, target)
    grad = mse.backward()
    expected = (2.0 / pred.size) * (pred - target)
    assert np.allclose(grad, expected)


def test_mse_backward_without_forward_raises():
    mse = MSE()
    with pytest.raises(RuntimeError):
        mse.backward()

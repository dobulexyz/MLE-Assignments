import numpy as np
import pytest

from nn import SGD
from nn.linear import Linear


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_linear_init_smoke():
    layer = Linear(4, 3, seed=0)
    assert layer.W.shape == (4, 3)
    assert layer.b.shape == (3,)


def test_linear_forward_smoke():
    layer = Linear(4, 3, seed=0)
    out = layer.forward(np.zeros((2, 4)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_linear_backward_smoke():
    layer = Linear(4, 3, seed=0)
    layer.forward(np.zeros((2, 4)))
    grad = layer.backward(np.zeros((2, 3)))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 4)


# ---------------------------------------------------------------------------
# Linear.__init__
# ---------------------------------------------------------------------------


def test_linear_bias_initialized_to_zeros():
    layer = Linear(8, 5, seed=0)
    assert np.all(layer.b == 0.0)


def test_linear_he_initialization_variance():
    in_features = 200
    layer = Linear(in_features, 200, seed=42)
    expected_variance = 2.0 / in_features
    assert layer.W.var() == pytest.approx(expected_variance, rel=0.15)


def test_linear_seed_reproducible():
    a = Linear(5, 3, seed=7)
    b = Linear(5, 3, seed=7)
    assert np.array_equal(a.W, b.W)


def test_linear_different_seeds_differ():
    a = Linear(5, 3, seed=1)
    b = Linear(5, 3, seed=2)
    assert not np.array_equal(a.W, b.W)


# ---------------------------------------------------------------------------
# Linear.forward
# ---------------------------------------------------------------------------


def test_linear_forward_matches_manual_matmul():
    layer = Linear(3, 2, seed=0)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = x @ layer.W + layer.b
    assert np.allclose(layer.forward(x), expected)


def test_linear_forward_caches_input():
    layer = Linear(3, 2, seed=0)
    x = np.array([[1.0, 2.0, 3.0]])
    layer.forward(x)
    assert layer.x is not None
    assert np.array_equal(layer.x, x)


# ---------------------------------------------------------------------------
# Linear.backward
# ---------------------------------------------------------------------------


def test_linear_backward_dx_shape():
    layer = Linear(3, 2, seed=0)
    layer.forward(np.ones((4, 3)))
    dx = layer.backward(np.ones((4, 2)))
    assert dx.shape == (4, 3)


def test_linear_backward_computes_dW():
    layer = Linear(3, 2, seed=0)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dout = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.forward(x)
    layer.backward(dout)
    assert np.allclose(layer.dW, x.T @ dout)


def test_linear_backward_computes_db():
    layer = Linear(3, 2, seed=0)
    layer.forward(np.ones((4, 3)))
    dout = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    layer.backward(dout)
    assert np.allclose(layer.db, dout.sum(axis=0))


def test_linear_backward_computes_dx():
    layer = Linear(3, 2, seed=0)
    x = np.array([[1.0, 2.0, 3.0]])
    dout = np.array([[1.0, 1.0]])
    layer.forward(x)
    dx = layer.backward(dout)
    assert np.allclose(dx, dout @ layer.W.T)


# ---------------------------------------------------------------------------
# Linear.parameters
# ---------------------------------------------------------------------------


def test_linear_parameters_yields_W_and_b():
    layer = Linear(3, 2, seed=0)
    pairs = list(layer.parameters())
    assert len(pairs) == 2
    params = [p for p, _ in pairs]
    assert any(p is layer.W for p in params)
    assert any(p is layer.b for p in params)


def test_linear_sgd_step_changes_weights():
    layer = Linear(3, 2, seed=0)
    layer.forward(np.ones((1, 3)))
    layer.backward(np.ones((1, 2)))
    W_before = layer.W.copy()
    SGD(layer.parameters(), lr=0.1).step()
    assert not np.array_equal(layer.W, W_before)


def test_linear_sgd_step_applies_sgd_rule():
    layer = Linear(2, 1, seed=0)
    layer.forward(np.array([[1.0, 1.0]]))
    layer.backward(np.array([[1.0]]))
    expected_W = layer.W - 0.5 * layer.dW
    expected_b = layer.b - 0.5 * layer.db
    SGD(layer.parameters(), lr=0.5).step()
    assert np.allclose(layer.W, expected_W)
    assert np.allclose(layer.b, expected_b)

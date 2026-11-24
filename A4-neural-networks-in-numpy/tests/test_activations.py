import numpy as np
import pytest

from nn.activations import GELU, LeakyReLU, ReLU


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_relu_forward_smoke():
    relu = ReLU()
    out = relu.forward(np.array([[1.0, -1.0]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_relu_backward_smoke():
    relu = ReLU()
    relu.forward(np.array([[1.0, -1.0]]))
    grad = relu.backward(np.array([[1.0, 1.0]]))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (1, 2)


# ---------------------------------------------------------------------------
# ReLU.forward
# ---------------------------------------------------------------------------


def test_relu_forward_clips_negatives_to_zero():
    relu = ReLU()
    out = relu.forward(np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]]))
    expected = np.array([[0.0, 0.0, 0.0, 0.5, 2.0]])
    assert np.array_equal(out, expected)


def test_relu_forward_preserves_shape():
    relu = ReLU()
    out = relu.forward(np.zeros((3, 4, 5)))
    assert out.shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# ReLU.backward
# ---------------------------------------------------------------------------


def test_relu_backward_zeros_negative_inputs():
    relu = ReLU()
    relu.forward(np.array([[-1.0, 1.0, -2.0, 2.0]]))
    grad = relu.backward(np.array([[5.0, 5.0, 5.0, 5.0]]))
    assert np.array_equal(grad, np.array([[0.0, 5.0, 0.0, 5.0]]))


def test_relu_backward_zero_at_zero_input():
    relu = ReLU()
    relu.forward(np.array([[0.0, 0.0]]))
    grad = relu.backward(np.array([[7.0, 7.0]]))
    assert np.array_equal(grad, np.array([[0.0, 0.0]]))


def test_relu_backward_passes_positive_gradients():
    relu = ReLU()
    relu.forward(np.array([[3.0, 5.0]]))
    grad = relu.backward(np.array([[2.0, 4.0]]))
    assert np.array_equal(grad, np.array([[2.0, 4.0]]))


def test_relu_backward_without_forward_raises():
    relu = ReLU()
    with pytest.raises(RuntimeError):
        relu.backward(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# LeakyReLU
# ---------------------------------------------------------------------------


def test_leaky_relu_forward_smoke():
    leaky = LeakyReLU()
    out = leaky.forward(np.array([[1.0, -1.0]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_leaky_relu_forward_scales_negatives():
    leaky = LeakyReLU(negative_slope=0.01)
    out = leaky.forward(np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]]))
    expected = np.array([[-0.02, -0.005, 0.0, 0.5, 2.0]])
    assert np.allclose(out, expected)


def test_leaky_relu_backward_uses_slope_for_negatives():
    leaky = LeakyReLU(negative_slope=0.1)
    leaky.forward(np.array([[-1.0, 1.0, -2.0, 2.0]]))
    grad = leaky.backward(np.array([[5.0, 5.0, 5.0, 5.0]]))
    assert np.allclose(grad, np.array([[0.5, 5.0, 0.5, 5.0]]))


def test_leaky_relu_backward_without_forward_raises():
    leaky = LeakyReLU()
    with pytest.raises(RuntimeError):
        leaky.backward(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------


def test_gelu_forward_smoke():
    gelu = GELU()
    out = gelu.forward(np.array([[1.0, -1.0]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_gelu_forward_zero_maps_to_zero():
    gelu = GELU()
    out = gelu.forward(np.array([[0.0]]))
    assert np.allclose(out, np.array([[0.0]]))


def test_gelu_forward_saturates_to_identity_and_zero():
    gelu = GELU()
    out = gelu.forward(np.array([[10.0, -10.0]]))
    assert np.allclose(out, np.array([[10.0, 0.0]]), atol=1e-3)


def test_gelu_backward_matches_finite_differences():
    x = np.array([[-2.0, -0.5, 0.3, 1.5]])
    gelu = GELU()
    gelu.forward(x)
    analytic = gelu.backward(np.ones_like(x))

    eps = 1e-6
    numeric = np.zeros_like(x)
    for i in range(x.shape[1]):
        xp = x.copy()
        xm = x.copy()
        xp[0, i] += eps
        xm[0, i] -= eps
        fp = GELU().forward(xp)[0, i]
        fm = GELU().forward(xm)[0, i]
        numeric[0, i] = (fp - fm) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_gelu_backward_without_forward_raises():
    gelu = GELU()
    with pytest.raises(RuntimeError):
        gelu.backward(np.array([[1.0]]))

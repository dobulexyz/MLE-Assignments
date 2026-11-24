"""Tests for the Dropout layer."""

import numpy as np
import pytest

from nn.dropout import Dropout


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_dropout_forward_smoke_train():
    d = Dropout(p=0.5, seed=0)
    out = d.forward(np.ones((4, 8)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 8)


def test_dropout_forward_smoke_eval():
    d = Dropout(p=0.5, seed=0)
    d.eval()
    out = d.forward(np.ones((4, 8)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 8)


# ---------------------------------------------------------------------------
# Eval mode: identity behaviour
# ---------------------------------------------------------------------------


def test_dropout_eval_forward_is_identity():
    d = Dropout(p=0.5, seed=0)
    d.eval()
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(d.forward(x), x)


def test_dropout_eval_backward_is_identity():
    d = Dropout(p=0.5, seed=0)
    d.eval()
    # backward in eval mode passes gradient through unchanged; no forward
    # needed because eval mode never uses the mask.
    dout = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert np.array_equal(d.backward(dout), dout)


# ---------------------------------------------------------------------------
# Train mode: mask statistics and scaling
# ---------------------------------------------------------------------------


def test_dropout_train_zeros_fraction():
    """Dropped fraction should be close to p for a large input."""
    p = 0.4
    d = Dropout(p=p, seed=42)
    x = np.ones((1000, 10))
    out = d.forward(x)
    zero_fraction = (out == 0.0).mean()
    assert abs(zero_fraction - p) < 0.03


def test_dropout_train_survivors_scaled():
    """Surviving (non-zero) entries should equal 1 / (1 - p)."""
    p = 0.3
    d = Dropout(p=p, seed=7)
    x = np.ones((100, 20))
    out = d.forward(x)
    kept = out[out != 0.0]
    expected_scale = 1.0 / (1.0 - p)
    assert np.allclose(kept, expected_scale)


def test_dropout_train_mask_reused_in_backward():
    """The same binary mask that was applied in forward is used in backward."""
    d = Dropout(p=0.5, seed=0)
    x = np.ones((10, 10))
    out = d.forward(x)
    dout = np.ones_like(x)
    dx = d.backward(dout)
    # Where forward output was zero, gradient must also be zero; where
    # forward output was non-zero, gradient must equal dout * mask value.
    assert np.array_equal(dx == 0.0, out == 0.0)


def test_dropout_train_backward_gradient_values():
    """Backward gradient equals dout * mask (same scaling factor)."""
    p = 0.5
    d = Dropout(p=p, seed=3)
    x = np.ones((4, 4))
    d.forward(x)
    dout = np.full((4, 4), 2.0)
    dx = d.backward(dout)
    # Non-zero entries should have gradient = 2.0 * (1/(1-p))
    kept = dx[dx != 0.0]
    assert np.allclose(kept, 2.0 / (1.0 - p))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_dropout_same_seed_same_mask():
    """Two Dropout instances with the same seed produce identical outputs."""
    x = np.ones((5, 5))
    d1 = Dropout(p=0.5, seed=99)
    d2 = Dropout(p=0.5, seed=99)
    out1 = d1.forward(x)
    out2 = d2.forward(x)
    assert np.array_equal(out1, out2)


def test_dropout_different_seeds_different_masks():
    """Different seeds should (with overwhelming probability) differ."""
    x = np.ones((20, 20))
    d1 = Dropout(p=0.5, seed=0)
    d2 = Dropout(p=0.5, seed=1)
    out1 = d1.forward(x)
    out2 = d2.forward(x)
    assert not np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_dropout_invalid_p_negative():
    with pytest.raises(ValueError):
        Dropout(p=-0.1)


def test_dropout_invalid_p_one():
    with pytest.raises(ValueError):
        Dropout(p=1.0)


def test_dropout_invalid_p_above_one():
    with pytest.raises(ValueError):
        Dropout(p=1.5)


def test_dropout_p_zero_valid():
    """p=0.0 is allowed and produces no drops."""
    d = Dropout(p=0.0, seed=0)
    x = np.ones((10, 10))
    out = d.forward(x)
    assert np.all(out == 1.0)

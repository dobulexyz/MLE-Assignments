import numpy as np
import pytest

from nn import SGD, Adam


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_sgd_step_smoke():
    p = np.zeros(3)
    g = np.ones(3)
    result = SGD([(p, g)], lr=0.1).step()
    assert result is None


def test_adam_step_smoke():
    p = np.zeros(3)
    g = np.ones(3)
    result = Adam([(p, g)]).step()
    assert result is None


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------


def test_sgd_vanilla_update():
    p = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    SGD([(p, g)], lr=0.1).step()
    assert np.allclose(p, [1.0 - 0.1 * 0.5, 2.0 - 0.1 * -1.0])


def test_sgd_updates_all_parameters():
    p1 = np.array([1.0])
    g1 = np.array([1.0])
    p2 = np.array([2.0])
    g2 = np.array([2.0])
    SGD([(p1, g1), (p2, g2)], lr=0.1).step()
    assert p1[0] == pytest.approx(0.9)
    assert p2[0] == pytest.approx(1.8)


def test_sgd_momentum_accumulates_velocity():
    p = np.array([0.0])
    g = np.array([1.0])
    opt = SGD([(p, g)], lr=0.1, momentum=0.9)
    opt.step()  # v = 1.0, p = -0.1
    g[...] = 1.0
    opt.step()  # v = 0.9 * 1 + 1 = 1.9, p = -0.1 - 0.19
    assert p[0] == pytest.approx(-0.29)


def test_sgd_nesterov_update():
    p = np.array([0.0])
    g = np.array([1.0])
    # v = 1.0, effective = grad + momentum * v = 1 + 0.9 = 1.9
    SGD([(p, g)], lr=0.1, momentum=0.9, nesterov=True).step()
    assert p[0] == pytest.approx(-0.19)


def test_sgd_nesterov_requires_momentum():
    with pytest.raises(ValueError):
        SGD([(np.zeros(1), np.zeros(1))], lr=0.1, nesterov=True)


def test_sgd_weight_decay_update():
    p = np.array([2.0])
    g = np.array([0.0])
    # grad' = 0 + 0.5 * 2 = 1.0 ; p = 2 - 0.1 * 1.0 = 1.9
    SGD([(p, g)], lr=0.1, weight_decay=0.5).step()
    assert p[0] == pytest.approx(1.9)


def test_sgd_weight_decay_preserves_grad_buffer():
    p = np.array([2.0])
    g = np.array([0.0])
    SGD([(p, g)], lr=0.1, weight_decay=0.5).step()
    assert g[0] == 0.0


def test_sgd_negative_weight_decay_raises():
    with pytest.raises(ValueError):
        SGD([(np.zeros(1), np.zeros(1))], lr=0.1, weight_decay=-1.0)


def test_sgd_zero_grad_clears_buffer():
    p = np.array([1.0])
    g = np.array([5.0])
    SGD([(p, g)], lr=0.1).zero_grad()
    assert g[0] == 0.0


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


def test_adam_first_step_is_bias_corrected():
    p = np.array([0.0])
    g = np.array([7.0])
    # Bias correction makes the first step ~ -lr * sign(grad), independent
    # of the gradient magnitude.
    Adam([(p, g)], lr=0.1).step()
    assert p[0] == pytest.approx(-0.1, rel=1e-6)


def test_adam_converges_to_minimum():
    # Minimize (x - 3)^2 ; grad = 2 (x - 3).
    p = np.array([0.0])
    g = np.zeros_like(p)
    opt = Adam([(p, g)], lr=0.1)
    for _ in range(1000):
        g[...] = 2.0 * (p - 3.0)
        opt.step()
    assert p[0] == pytest.approx(3.0, abs=1e-2)


def test_adam_weight_decay_shrinks_param():
    p = np.array([5.0])
    g = np.zeros_like(p)
    opt = Adam([(p, g)], lr=0.1, weight_decay=0.1)
    before = p[0]
    for _ in range(50):
        g[...] = 0.0  # no data gradient; only the decay term acts
        opt.step()
    assert abs(p[0]) < before


def test_adam_weight_decay_preserves_grad_buffer():
    p = np.array([2.0])
    g = np.array([0.0])
    Adam([(p, g)], lr=0.1, weight_decay=0.5).step()
    assert g[0] == 0.0


def test_adam_negative_weight_decay_raises():
    with pytest.raises(ValueError):
        Adam([(np.zeros(1), np.zeros(1))], weight_decay=-1.0)


def test_adam_zero_grad_clears_buffer():
    p = np.array([1.0])
    g = np.array([5.0])
    Adam([(p, g)]).zero_grad()
    assert g[0] == 0.0

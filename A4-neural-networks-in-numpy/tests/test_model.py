import numpy as np
import pytest

from nn import SGD
from nn.activations import GELU, LeakyReLU, ReLU
from nn.dropout import Dropout
from nn.linear import Linear
from nn.loss import CrossEntropy
from nn.mlp import MLP
from nn.sequential import Sequential


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 0) -> Sequential:
    return Sequential(
        [
            Linear(4, 5, seed=seed),
            ReLU(),
            Linear(5, 3, seed=seed + 1),
        ]
    )


def test_model_forward_smoke():
    out = _tiny_model().forward(np.zeros((2, 4)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_model_backward_smoke():
    model = _tiny_model()
    model.forward(np.ones((2, 4)))
    grad = model.backward(np.ones((2, 3)))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 4)


# ---------------------------------------------------------------------------
# Sequential.forward
# ---------------------------------------------------------------------------


def test_model_forward_matches_manual_pipeline():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    model = _tiny_model(seed=11)

    h1 = model.layers[0].forward(x.copy())
    h2 = model.layers[1].forward(h1)
    h3 = model.layers[2].forward(h2)

    fresh = _tiny_model(seed=11)
    assert np.allclose(fresh.forward(x), h3)


def test_model_callable_equals_forward():
    model = _tiny_model(seed=3)
    x = np.ones((2, 4))
    assert np.array_equal(model(x), model.forward(x))


# ---------------------------------------------------------------------------
# Sequential.backward
# ---------------------------------------------------------------------------


def test_model_backward_runs_layers_in_reverse():
    model = _tiny_model(seed=2)
    x = np.ones((2, 4))
    model.forward(x)
    dout = np.ones((2, 3))
    dx = model.backward(dout)

    # All trainable layers should have non-zero parameter gradients
    # after backward.
    assert np.any(model.layers[0].dW != 0)
    assert np.any(model.layers[2].dW != 0)
    assert dx.shape == x.shape


# ---------------------------------------------------------------------------
# Sequential.parameters
# ---------------------------------------------------------------------------


def test_model_parameters_chains_layer_parameters():
    model = _tiny_model()
    pairs = list(model.parameters())
    # Two Linear layers, each contributes (W, dW) and (b, db).
    assert len(pairs) == 4


def test_model_parameters_skips_relu():
    model = Sequential([ReLU(), ReLU()])
    assert list(model.parameters()) == []


# ---------------------------------------------------------------------------
# SGD step via Sequential.parameters
# ---------------------------------------------------------------------------


def test_model_sgd_step_changes_every_linear():
    model = _tiny_model(seed=4)
    x = np.ones((2, 4))
    y = np.array([0, 1])

    loss = CrossEntropy()
    logits = model.forward(x)
    loss.forward(logits, y)
    model.backward(loss.backward())

    W0_before = model.layers[0].W.copy()
    W2_before = model.layers[2].W.copy()
    SGD(model.parameters(), lr=0.1).step()

    assert not np.array_equal(model.layers[0].W, W0_before)
    assert not np.array_equal(model.layers[2].W, W2_before)


# ---------------------------------------------------------------------------
# End-to-end overfit
# ---------------------------------------------------------------------------


def test_model_can_overfit_tiny_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4)).astype(np.float64)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

    model = Sequential(
        [
            Linear(4, 16, seed=0),
            ReLU(),
            Linear(16, 3, seed=1),
        ]
    )
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.1)

    for _ in range(400):
        logits = model.forward(X)
        loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        optimizer.step()

    preds = model.forward(X).argmax(axis=1)
    assert np.array_equal(preds, y)


# ---------------------------------------------------------------------------
# Sequential empty-layers guard
# ---------------------------------------------------------------------------


def test_sequential_empty_raises():
    with pytest.raises(ValueError):
        Sequential([])


# ===========================================================================
# MLP tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_mlp_forward_smoke():
    mlp = MLP([4, 8, 3], activation="relu", seed=0)
    out = mlp.forward(np.zeros((2, 4)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_mlp_backward_smoke():
    mlp = MLP([4, 8, 3], activation="relu", seed=0)
    mlp.forward(np.ones((2, 4)))
    grad = mlp.backward(np.ones((2, 3)))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 4)


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------


def test_mlp_layer_count_one_hidden():
    # [784, 128, 10]: Linear, ReLU, Linear → 3 layers
    mlp = MLP([784, 128, 10], activation="relu", seed=0)
    assert len(mlp.net.layers) == 3


def test_mlp_layer_count_two_hidden():
    # [4, 8, 16, 3]: Linear, ReLU, Linear, ReLU, Linear → 5 layers
    mlp = MLP([4, 8, 16, 3], activation="relu", seed=0)
    assert len(mlp.net.layers) == 5


def test_mlp_linear_widths():
    mlp = MLP([4, 8, 3], activation="relu", seed=0)
    linears = [lay for lay in mlp.net.layers if isinstance(lay, Linear)]
    assert len(linears) == 2
    assert linears[0].W.shape == (4, 8)
    assert linears[1].W.shape == (8, 3)


def test_mlp_activation_between_linears_not_after_last():
    mlp = MLP([4, 8, 16, 3], activation="relu", seed=0)
    layers = mlp.net.layers
    # Layers: Lin, Act, Lin, Act, Lin
    assert isinstance(layers[0], Linear)
    assert isinstance(layers[1], ReLU)
    assert isinstance(layers[2], Linear)
    assert isinstance(layers[3], ReLU)
    assert isinstance(layers[4], Linear)


def test_mlp_leaky_relu_activation():
    mlp = MLP([4, 8, 3], activation="leaky_relu", seed=0)
    activations = [
        lay for lay in mlp.net.layers if not isinstance(lay, Linear)
    ]
    assert len(activations) == 1
    assert isinstance(activations[0], LeakyReLU)


def test_mlp_gelu_activation():
    mlp = MLP([4, 8, 3], activation="gelu", seed=0)
    activations = [
        lay for lay in mlp.net.layers if not isinstance(lay, Linear)
    ]
    assert len(activations) == 1
    assert isinstance(activations[0], GELU)


def test_mlp_unknown_activation_raises():
    with pytest.raises(ValueError):
        MLP([4, 8, 3], activation="sigmoid", seed=0)


# ---------------------------------------------------------------------------
# Dropout injection
# ---------------------------------------------------------------------------


def test_mlp_no_dropout_by_default():
    mlp = MLP([4, 8, 3], activation="relu", seed=0)
    dropout_layers = [
        lay for lay in mlp.net.layers if isinstance(lay, Dropout)
    ]
    assert len(dropout_layers) == 0


def test_mlp_dropout_injected_after_hidden_activations():
    # [4, 8, 16, 3] with dropout: Lin, Act, Drop, Lin, Act, Drop, Lin
    mlp = MLP([4, 8, 16, 3], activation="relu", dropout=0.3, seed=0)
    layers = mlp.net.layers
    assert isinstance(layers[0], Linear)
    assert isinstance(layers[1], ReLU)
    assert isinstance(layers[2], Dropout)
    assert isinstance(layers[3], Linear)
    assert isinstance(layers[4], ReLU)
    assert isinstance(layers[5], Dropout)
    assert isinstance(layers[6], Linear)


def test_mlp_dropout_layer_count():
    # Two hidden layers → two Dropout layers
    mlp = MLP([4, 8, 16, 3], activation="relu", dropout=0.5, seed=0)
    dropout_layers = [
        lay for lay in mlp.net.layers if isinstance(lay, Dropout)
    ]
    assert len(dropout_layers) == 2


# ---------------------------------------------------------------------------
# Train / eval mode propagation
# ---------------------------------------------------------------------------


def test_mlp_train_propagates_to_children():
    mlp = MLP([4, 8, 3], activation="relu", dropout=0.5, seed=0)
    mlp.eval()
    mlp.train()
    for layer in mlp.net.layers:
        assert layer.training is True


def test_mlp_eval_propagates_to_children():
    mlp = MLP([4, 8, 3], activation="relu", dropout=0.5, seed=0)
    mlp.eval()
    for layer in mlp.net.layers:
        assert layer.training is False

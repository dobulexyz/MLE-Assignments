"""Grader implementation for A4-neural-networks-in-numpy.

NOT FOR DISTRIBUTION."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def _normalize(obj, _d=4):
    if isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return f"{obj:.{_d}f}"
    if isinstance(obj, np.ndarray):
        return _normalize(obj.tolist(), _d)
    if isinstance(obj, (list, tuple)):
        return [_normalize(x, _d) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _normalize(v, _d) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return f"{float(obj):.{_d}f}"
    return obj


def _hash(result):
    try:
        text = json.dumps(
            _normalize(result), sort_keys=True, ensure_ascii=True
        )
    except Exception as e:
        text = f"EXCEPTION:{type(e).__name__}"
    return hashlib.sha256(text.encode()).hexdigest()


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return f"EXCEPTION:{type(e).__name__}"


def _R(raw):
    return {k: _hash(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Trusted SGD helpers — used in tasks b and f so those tasks do NOT
# depend on the student's optim.SGD (task g).
# Note: Linear.backward ACCUMULATES (dW +=), so call _ref_zero_grad
# before each forward/backward.
# ---------------------------------------------------------------------------


def _ref_zero_grad(params):
    for _, grad in params:
        grad[...] = 0.0


def _ref_sgd_step(params, lr):
    for param, grad in params:
        param -= lr * grad


# ---------------------------------------------------------------------------
# Shared synthetic dataset helper
# ---------------------------------------------------------------------------


def _make_synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n_per_class = 60
    means = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
    X = np.concatenate(
        [rng.normal(loc=mu, scale=0.6, size=(n_per_class, 2)) for mu in means]
    ).astype(np.float64)
    y = np.concatenate([np.full(n_per_class, i) for i in range(3)]).astype(
        np.int64
    )
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Task b — Linear  (src/nn/linear.py)
# ---------------------------------------------------------------------------


def _run_b():
    from nn.linear import Linear

    def _shapes():
        layer = Linear(4, 3, seed=0)
        return [list(layer.W.shape), list(layer.b.shape)]

    def _bias_zero():
        return Linear(8, 5, seed=0).b.tolist()

    def _seed_reproducible():
        a = Linear(5, 3, seed=7)
        b = Linear(5, 3, seed=7)
        return bool(np.array_equal(a.W, b.W))

    def _he_variance():
        layer = Linear(200, 200, seed=42)
        ratio = layer.W.var() / (2.0 / 200)
        return 0.85 < ratio < 1.15

    def _forward():
        layer = Linear(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return layer.forward(x).tolist()

    def _forward_caches_input():
        layer = Linear(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0]])
        layer.forward(x)
        return layer.x.tolist()

    def _callable_equals_forward():
        layer = Linear(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0]])
        return bool(np.array_equal(layer(x), layer.forward(x)))

    def _backward_dW():
        layer = Linear(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dout = np.array([[1.0, 0.0], [0.0, 1.0]])
        layer.forward(x)
        layer.backward(dout)
        return layer.dW.tolist()

    def _backward_db():
        layer = Linear(3, 2, seed=0)
        layer.forward(np.ones((4, 3)))
        dout = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        layer.backward(dout)
        return layer.db.tolist()

    def _backward_dx():
        layer = Linear(3, 2, seed=0)
        x = np.array([[1.0, 2.0, 3.0]])
        dout = np.array([[1.0, 1.0]])
        layer.forward(x)
        return layer.backward(dout).tolist()

    def _backward_without_forward():
        layer = Linear(3, 2, seed=0)
        return _safe(lambda: layer.backward(np.zeros((1, 2))))

    def _parameters_count():
        layer = Linear(3, 2, seed=0)
        return len(list(layer.parameters()))

    def _parameters_order():
        layer = Linear(3, 2, seed=0)
        pairs = list(layer.parameters())
        return [
            bool(pairs[0][0] is layer.W and pairs[0][1] is layer.dW),
            bool(pairs[1][0] is layer.b and pairs[1][1] is layer.db),
        ]

    def _parameters_step():
        # Use trusted SGD — must NOT use student optim.SGD
        layer = Linear(2, 1, seed=0)
        params = list(layer.parameters())
        _ref_zero_grad(params)
        layer.forward(np.array([[1.0, 1.0]]))
        layer.backward(np.array([[1.0]]))
        W_before = layer.W.copy()
        b_before = layer.b.copy()
        _ref_sgd_step(params, lr=0.5)
        w_changed = bool(not np.array_equal(layer.W, W_before))
        b_changed = bool(not np.array_equal(layer.b, b_before))
        # Check that the step magnitude matches -lr*grad
        expected_W = W_before - 0.5 * layer.dW
        expected_b = b_before - 0.5 * layer.db
        w_correct = bool(np.allclose(layer.W, expected_W))
        b_correct = bool(np.allclose(layer.b, expected_b))
        return [w_changed, b_changed, w_correct, b_correct]

    return _R(
        {
            "linear_init": [
                _safe(_shapes),
                _safe(_bias_zero),
                _safe(_seed_reproducible),
                _safe(_he_variance),
            ],
            "linear_forward": [
                _safe(_forward),
                _safe(_forward_caches_input),
                _safe(_callable_equals_forward),
            ],
            "linear_backward": [
                _safe(_backward_dW),
                _safe(_backward_db),
                _safe(_backward_dx),
                _backward_without_forward(),
            ],
            "linear_parameters": [
                _safe(_parameters_count),
                _safe(_parameters_order),
                _safe(_parameters_step),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task c — Activations  (src/nn/activations.py)
# ---------------------------------------------------------------------------


def _run_c():
    from nn.activations import GELU, LeakyReLU, ReLU

    def _relu_forward():
        relu = ReLU()
        return relu.forward(np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])).tolist()

    def _relu_backward_negative():
        relu = ReLU()
        relu.forward(np.array([[-1.0, 1.0, -2.0, 2.0]]))
        return relu.backward(np.array([[5.0, 5.0, 5.0, 5.0]])).tolist()

    def _relu_backward_zero():
        relu = ReLU()
        relu.forward(np.array([[0.0, 0.0]]))
        return relu.backward(np.array([[7.0, 7.0]])).tolist()

    def _relu_callable():
        relu = ReLU()
        x = np.array([[-1.0, 2.0]])
        return bool(np.array_equal(relu(x), relu.forward(x)))

    def _relu_no_forward():
        return _safe(lambda: ReLU().backward(np.zeros((1, 2))))

    def _leaky_forward():
        leaky = LeakyReLU()
        x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
        return leaky.forward(x).tolist()

    def _leaky_backward_negative():
        leaky = LeakyReLU()
        leaky.forward(np.array([[-1.0, 1.0, -2.0, 2.0]]))
        return leaky.backward(np.array([[5.0, 5.0, 5.0, 5.0]])).tolist()

    def _leaky_no_forward():
        return _safe(lambda: LeakyReLU().backward(np.zeros((1, 2))))

    def _leaky_custom_slope():
        leaky = LeakyReLU(negative_slope=0.2)
        x = np.array([[-1.0, 2.0]])
        fwd = leaky.forward(x).tolist()
        bwd = leaky.backward(np.ones_like(x)).tolist()
        return [fwd, bwd]

    def _gelu_forward():
        gelu = GELU()
        return gelu.forward(np.array([[-1.0, 0.0, 0.5, 2.0]])).tolist()

    def _gelu_backward():
        gelu = GELU()
        gelu.forward(np.array([[-1.0, 0.0, 0.5, 2.0]]))
        return gelu.backward(np.array([[1.0, 1.0, 1.0, 1.0]])).tolist()

    def _gelu_no_forward():
        return _safe(lambda: GELU().backward(np.zeros((1, 2))))

    return _R(
        {
            "relu": [
                _safe(_relu_forward),
                _safe(_relu_backward_negative),
                _safe(_relu_backward_zero),
                _safe(_relu_callable),
                _relu_no_forward(),
            ],
            "leaky_relu": [
                _safe(_leaky_forward),
                _safe(_leaky_backward_negative),
                _leaky_no_forward(),
                _safe(_leaky_custom_slope),
            ],
            "gelu": [
                _safe(_gelu_forward),
                _safe(_gelu_backward),
                _gelu_no_forward(),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task d — Loss  (src/nn/loss.py)
# ---------------------------------------------------------------------------


def _run_d():
    from nn.loss import CrossEntropy, MSE

    # --- CrossEntropy ---

    def _ce_uniform():
        loss = CrossEntropy()
        return loss.forward(np.zeros((4, 5)), np.array([0, 1, 2, 3]))

    def _ce_perfect():
        loss = CrossEntropy()
        logits = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
        return loss.forward(logits, np.array([0, 2]))

    def _ce_stability():
        loss = CrossEntropy()
        logits = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        value = loss.forward(logits, np.array([0, 1]))
        return bool(np.isfinite(value))

    def _ce_backward_sum_zero():
        loss = CrossEntropy()
        logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, -0.5]])
        loss.forward(logits, np.array([0, 2]))
        sums = loss.backward().sum(axis=1)
        return bool(np.allclose(sums, 0.0))

    def _ce_backward_values():
        loss = CrossEntropy()
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        loss.forward(logits, np.array([2, 0]))
        return loss.backward().tolist()

    def _ce_no_forward():
        return _safe(CrossEntropy().backward)

    # --- MSE ---

    def _mse_known():
        loss = MSE()
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = np.array([[1.0, 1.0], [1.0, 1.0]])
        # (0+1+4+9)/4 = 3.5
        return loss.forward(pred, target)

    def _mse_zero():
        loss = MSE()
        x = np.array([[1.0, 2.0, 3.0]])
        return loss.forward(x, x.copy())

    def _mse_backward_values():
        loss = MSE()
        pred = np.array([[1.0, 3.0]])
        target = np.array([[0.0, 0.0]])
        loss.forward(pred, target)
        # (2/N)*(pred-target); N=2 → [1.0, 3.0]
        return loss.backward().tolist()

    def _mse_no_forward():
        return _safe(MSE().backward)

    return _R(
        {
            "cross_entropy_forward": [
                _safe(_ce_uniform),
                _safe(_ce_perfect),
                _safe(_ce_stability),
            ],
            "cross_entropy_backward": [
                _safe(_ce_backward_sum_zero),
                _safe(_ce_backward_values),
            ],
            "cross_entropy_errors": [
                _ce_no_forward(),
            ],
            "mse_forward": [
                _safe(_mse_known),
                _safe(_mse_zero),
            ],
            "mse_backward": [
                _safe(_mse_backward_values),
            ],
            "mse_errors": [
                _mse_no_forward(),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task e — Dropout  (src/nn/dropout.py)
# ---------------------------------------------------------------------------


def _run_e():
    from nn.dropout import Dropout

    def _zeros_fraction():
        p = 0.4
        d = Dropout(p=p, seed=42)
        x = np.ones((1000, 10))
        out = d.forward(x)
        frac = float((out == 0.0).mean())
        return abs(frac - p) < 0.03

    def _survivors_scaled():
        p = 0.3
        d = Dropout(p=p, seed=7)
        x = np.ones((100, 20))
        out = d.forward(x)
        kept = out[out != 0.0]
        expected = 1.0 / (1.0 - p)
        return bool(np.allclose(kept, expected))

    def _same_seed_same_mask():
        x = np.ones((5, 5))
        d1 = Dropout(p=0.5, seed=99)
        d2 = Dropout(p=0.5, seed=99)
        return bool(np.array_equal(d1.forward(x), d2.forward(x)))

    def _diff_seeds_diff_masks():
        x = np.ones((20, 20))
        d1 = Dropout(p=0.5, seed=0)
        d2 = Dropout(p=0.5, seed=1)
        return bool(not np.array_equal(d1.forward(x), d2.forward(x)))

    def _eval_forward_identity():
        d = Dropout(p=0.5, seed=0)
        d.eval()
        x = np.arange(12, dtype=np.float64).reshape(3, 4)
        return bool(np.array_equal(d.forward(x), x))

    def _eval_backward_identity():
        d = Dropout(p=0.5, seed=0)
        d.eval()
        dout = np.arange(6, dtype=np.float64).reshape(2, 3)
        return bool(np.array_equal(d.backward(dout), dout))

    def _mask_reused_backward():
        d = Dropout(p=0.5, seed=0)
        x = np.ones((10, 10))
        out = d.forward(x)
        dout = np.ones_like(x)
        dx = d.backward(dout)
        return bool(np.array_equal(dx == 0.0, out == 0.0))

    def _backward_gradient_values():
        p = 0.5
        d = Dropout(p=p, seed=3)
        x = np.ones((4, 4))
        d.forward(x)
        dout = np.full((4, 4), 2.0)
        dx = d.backward(dout)
        kept = dx[dx != 0.0]
        return bool(np.allclose(kept, 2.0 / (1.0 - p)))

    def _p_neg_raises():
        return _safe(lambda: Dropout(p=-0.1))

    def _p_one_raises():
        return _safe(lambda: Dropout(p=1.0))

    def _p_above_one_raises():
        return _safe(lambda: Dropout(p=1.5))

    def _p_zero_valid():
        d = Dropout(p=0.0, seed=0)
        x = np.ones((10, 10))
        out = d.forward(x)
        return bool(np.all(out == 1.0))

    return _R(
        {
            "dropout_train": [
                _safe(_zeros_fraction),
                _safe(_survivors_scaled),
                _safe(_same_seed_same_mask),
                _safe(_diff_seeds_diff_masks),
            ],
            "dropout_eval": [
                _safe(_eval_forward_identity),
                _safe(_eval_backward_identity),
            ],
            "dropout_backward": [
                _safe(_mask_reused_backward),
                _safe(_backward_gradient_values),
            ],
            "dropout_init": [
                _p_neg_raises(),
                _p_one_raises(),
                _p_above_one_raises(),
                _safe(_p_zero_valid),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task f — Models  (src/nn/sequential.py, src/nn/mlp.py)
# ---------------------------------------------------------------------------


def _run_f():
    from nn.activations import GELU, LeakyReLU, ReLU
    from nn.dropout import Dropout
    from nn.linear import Linear
    from nn.loss import CrossEntropy
    from nn.mlp import MLP
    from nn.sequential import Sequential

    def _build_model():
        return Sequential(
            [
                Linear(4, 5, seed=0),
                ReLU(),
                Linear(5, 3, seed=1),
            ]
        )

    # -- sequential_forward --

    def _forward_shape():
        out = _build_model().forward(np.zeros((2, 4)))
        return list(out.shape)

    def _callable_equals_forward():
        model = _build_model()
        x = np.ones((2, 4))
        return bool(np.array_equal(model(x), model.forward(x)))

    def _forward_matches_manual():
        model_a = Sequential(
            [Linear(4, 5, seed=0), ReLU(), Linear(5, 3, seed=1)]
        )
        x = np.array([[1.0, -1.0, 0.5, -0.5]])
        manual = model_a.layers[0].forward(x.copy())
        manual = model_a.layers[1].forward(manual)
        manual = model_a.layers[2].forward(manual)
        model_b = Sequential(
            [Linear(4, 5, seed=0), ReLU(), Linear(5, 3, seed=1)]
        )
        return bool(np.allclose(model_b.forward(x), manual))

    # -- sequential_backward --

    def _backward_dx_shape():
        model = _build_model()
        model.forward(np.ones((2, 4)))
        return list(model.backward(np.ones((2, 3))).shape)

    def _backward_updates_grads():
        model = _build_model()
        model.forward(np.ones((2, 4)))
        model.backward(np.ones((2, 3)))
        return [
            bool(np.any(model.layers[0].dW != 0)),
            bool(np.any(model.layers[2].dW != 0)),
        ]

    # -- sequential_parameters --

    def _parameters_count():
        return len(list(_build_model().parameters()))

    def _parameters_skips_relu():
        return len(list(Sequential([ReLU(), ReLU()]).parameters()))

    def _empty_sequential_raises():
        return _safe(lambda: Sequential([]))

    def _parameters_step_changes_weights():
        # Use trusted SGD — must NOT use student optim.SGD
        model = _build_model()
        loss = CrossEntropy()
        x = np.ones((2, 4))
        y = np.array([0, 1])
        params = list(model.parameters())
        _ref_zero_grad(params)
        logits = model.forward(x)
        loss.forward(logits, y)
        model.backward(loss.backward())
        W0_before = model.layers[0].W.copy()
        W2_before = model.layers[2].W.copy()
        _ref_sgd_step(params, lr=0.1)
        return [
            bool(not np.array_equal(model.layers[0].W, W0_before)),
            bool(not np.array_equal(model.layers[2].W, W2_before)),
        ]

    # -- sequential_overfits --

    def _overfits():
        # Use trusted SGD loop — must NOT use student optim.SGD
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 4)).astype(np.float64)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        model = Sequential(
            [Linear(4, 16, seed=0), ReLU(), Linear(16, 3, seed=1)]
        )
        loss = CrossEntropy()
        params = list(model.parameters())
        for _ in range(400):
            _ref_zero_grad(params)
            logits = model.forward(X)
            loss.forward(logits, y)
            model.backward(loss.backward())
            _ref_sgd_step(params, lr=0.1)
        preds = model.forward(X).argmax(axis=1)
        return bool(np.array_equal(preds, y))

    # -- mlp_structure --

    def _mlp_forward_shape():
        mlp = MLP([4, 8, 3], activation="relu", seed=0)
        out = mlp.forward(np.zeros((2, 4)))
        return list(out.shape)

    def _mlp_layer_count_one_hidden():
        mlp = MLP([784, 128, 10], activation="relu", seed=0)
        return len(mlp.net.layers)

    def _mlp_layer_count_two_hidden():
        mlp = MLP([4, 8, 16, 3], activation="relu", seed=0)
        return len(mlp.net.layers)

    def _mlp_linear_widths():
        mlp = MLP([4, 8, 3], activation="relu", seed=0)
        linears = [lay for lay in mlp.net.layers if isinstance(lay, Linear)]
        return [list(lay.W.shape) for lay in linears]

    def _mlp_activation_placement():
        mlp = MLP([4, 8, 16, 3], activation="relu", seed=0)
        layers = mlp.net.layers
        return [
            isinstance(layers[0], Linear),
            isinstance(layers[1], ReLU),
            isinstance(layers[2], Linear),
            isinstance(layers[3], ReLU),
            isinstance(layers[4], Linear),
        ]

    def _mlp_leaky_relu():
        mlp = MLP([4, 8, 3], activation="leaky_relu", seed=0)
        acts = [lay for lay in mlp.net.layers if not isinstance(lay, Linear)]
        return [len(acts) == 1, isinstance(acts[0], LeakyReLU)]

    def _mlp_gelu():
        mlp = MLP([4, 8, 3], activation="gelu", seed=0)
        acts = [lay for lay in mlp.net.layers if not isinstance(lay, Linear)]
        return [len(acts) == 1, isinstance(acts[0], GELU)]

    def _mlp_unknown_raises():
        return _safe(lambda: MLP([4, 8, 3], activation="sigmoid", seed=0))

    # -- mlp_dropout --

    def _mlp_no_dropout_default():
        mlp = MLP([4, 8, 3], activation="relu", seed=0)
        drops = [lay for lay in mlp.net.layers if isinstance(lay, Dropout)]
        return len(drops) == 0

    def _mlp_dropout_injected():
        mlp = MLP([4, 8, 16, 3], activation="relu", dropout=0.3, seed=0)
        layers = mlp.net.layers
        return [
            isinstance(layers[0], Linear),
            isinstance(layers[1], ReLU),
            isinstance(layers[2], Dropout),
            isinstance(layers[3], Linear),
            isinstance(layers[4], ReLU),
            isinstance(layers[5], Dropout),
            isinstance(layers[6], Linear),
        ]

    def _mlp_dropout_count():
        mlp = MLP([4, 8, 16, 3], activation="relu", dropout=0.5, seed=0)
        drops = [lay for lay in mlp.net.layers if isinstance(lay, Dropout)]
        return len(drops)

    # -- mlp_mode --

    def _mlp_train_propagates():
        mlp = MLP([4, 8, 3], activation="relu", dropout=0.5, seed=0)
        mlp.eval()
        mlp.train()
        return [bool(lay.training) for lay in mlp.net.layers]

    def _mlp_eval_propagates():
        mlp = MLP([4, 8, 3], activation="relu", dropout=0.5, seed=0)
        mlp.eval()
        return [bool(lay.training) for lay in mlp.net.layers]

    return _R(
        {
            "sequential_forward": [
                _safe(_forward_shape),
                _safe(_callable_equals_forward),
                _safe(_forward_matches_manual),
            ],
            "sequential_backward": [
                _safe(_backward_dx_shape),
                _safe(_backward_updates_grads),
            ],
            "sequential_parameters": [
                _safe(_parameters_count),
                _safe(_parameters_skips_relu),
                _empty_sequential_raises(),
                _safe(_parameters_step_changes_weights),
            ],
            "sequential_overfits": [
                _safe(_overfits),
            ],
            "mlp_structure": [
                _safe(_mlp_forward_shape),
                _safe(_mlp_layer_count_one_hidden),
                _safe(_mlp_layer_count_two_hidden),
                _safe(_mlp_linear_widths),
                _safe(_mlp_activation_placement),
                _safe(_mlp_leaky_relu),
                _safe(_mlp_gelu),
                _mlp_unknown_raises(),
            ],
            "mlp_dropout": [
                _safe(_mlp_no_dropout_default),
                _safe(_mlp_dropout_injected),
                _safe(_mlp_dropout_count),
            ],
            "mlp_mode": [
                _safe(_mlp_train_propagates),
                _safe(_mlp_eval_propagates),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task g — Optim  (src/optim/sgd.py, src/optim/adam.py)
# ---------------------------------------------------------------------------


def _run_g():
    from optim.adam import Adam
    from optim.sgd import SGD

    # -- sgd_vanilla --

    def _vanilla_update():
        p = np.array([1.0, 2.0])
        g = np.array([0.5, -1.0])
        SGD([(p, g)], lr=0.1).step()
        expected = np.array([1.0 - 0.1 * 0.5, 2.0 - 0.1 * -1.0])
        return bool(np.allclose(p, expected))

    def _vanilla_all_params():
        p1 = np.array([1.0])
        g1 = np.array([1.0])
        p2 = np.array([2.0])
        g2 = np.array([2.0])
        SGD([(p1, g1), (p2, g2)], lr=0.1).step()
        return [
            bool(abs(p1[0] - 0.9) < 1e-9),
            bool(abs(p2[0] - 1.8) < 1e-9),
        ]

    def _vanilla_zero_grad():
        p = np.array([1.0])
        g = np.array([5.0])
        SGD([(p, g)], lr=0.1).zero_grad()
        return bool(g[0] == 0.0)

    # -- sgd_momentum --

    def _momentum_accumulates():
        p = np.array([0.0])
        g = np.array([1.0])
        opt = SGD([(p, g)], lr=0.1, momentum=0.9)
        opt.step()  # v=1, p=-0.1
        g[...] = 1.0
        opt.step()  # v=0.9+1=1.9, p=-0.1-0.19
        return bool(abs(p[0] - (-0.29)) < 1e-9)

    def _nesterov_update():
        p = np.array([0.0])
        g = np.array([1.0])
        SGD([(p, g)], lr=0.1, momentum=0.9, nesterov=True).step()
        # v=1, effective=1+0.9=1.9, p=-0.19
        return bool(abs(p[0] - (-0.19)) < 1e-9)

    def _nesterov_requires_momentum():
        return _safe(
            lambda: SGD([(np.zeros(1), np.zeros(1))], lr=0.1, nesterov=True)
        )

    # -- sgd_weight_decay --

    def _wd_update():
        p = np.array([2.0])
        g = np.array([0.0])
        # grad' = 0 + 0.5*2 = 1.0; p = 2 - 0.1*1.0 = 1.9
        SGD([(p, g)], lr=0.1, weight_decay=0.5).step()
        return bool(abs(p[0] - 1.9) < 1e-9)

    def _wd_preserves_grad():
        p = np.array([2.0])
        g = np.array([0.0])
        SGD([(p, g)], lr=0.1, weight_decay=0.5).step()
        return bool(g[0] == 0.0)

    def _wd_negative_raises():
        return _safe(
            lambda: SGD(
                [(np.zeros(1), np.zeros(1))], lr=0.1, weight_decay=-1.0
            )
        )

    # -- adam --

    def _adam_first_step():
        p = np.array([0.0])
        g = np.array([7.0])
        Adam([(p, g)], lr=0.1).step()
        return bool(abs(p[0] - (-0.1)) < 1e-6)

    def _adam_converges():
        p = np.array([0.0])
        g = np.zeros_like(p)
        opt = Adam([(p, g)], lr=0.1)
        for _ in range(1000):
            g[...] = 2.0 * (p - 3.0)
            opt.step()
        return bool(abs(p[0] - 3.0) < 0.01)

    def _adam_wd_shrinks():
        p = np.array([5.0])
        g = np.zeros_like(p)
        opt = Adam([(p, g)], lr=0.1, weight_decay=0.1)
        before = p[0]
        for _ in range(50):
            g[...] = 0.0
            opt.step()
        return bool(abs(p[0]) < before)

    def _adam_preserves_grad():
        p = np.array([2.0])
        g = np.array([0.0])
        Adam([(p, g)], lr=0.1, weight_decay=0.5).step()
        return bool(g[0] == 0.0)

    def _adam_neg_wd_raises():
        return _safe(
            lambda: Adam([(np.zeros(1), np.zeros(1))], weight_decay=-1.0)
        )

    def _adam_zero_grad():
        p = np.array([1.0])
        g = np.array([5.0])
        Adam([(p, g)]).zero_grad()
        return bool(g[0] == 0.0)

    return _R(
        {
            "sgd_vanilla": [
                _safe(_vanilla_update),
                _safe(_vanilla_all_params),
                _safe(_vanilla_zero_grad),
            ],
            "sgd_momentum": [
                _safe(_momentum_accumulates),
                _safe(_nesterov_update),
                _nesterov_requires_momentum(),
            ],
            "sgd_weight_decay": [
                _safe(_wd_update),
                _safe(_wd_preserves_grad),
                _wd_negative_raises(),
            ],
            "adam": [
                _safe(_adam_first_step),
                _safe(_adam_converges),
                _safe(_adam_wd_shrinks),
                _safe(_adam_preserves_grad),
                _adam_neg_wd_raises(),
                _safe(_adam_zero_grad),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task h — Data  (src/data/dataset.py, src/data/dataloader.py)
# ---------------------------------------------------------------------------


def _run_h():
    from data.dataloader import DataLoader
    from data.dataset import ArrayDataset, Dataset

    def _make_ds(n=12):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 4)).astype(np.float32)
        y = np.arange(n, dtype=np.int64)
        return ArrayDataset(X, y)

    # -- dataset --

    def _abstract_raises():
        return _safe(Dataset)

    def _array_len():
        return len(_make_ds(7))

    def _array_getitem_x():
        X = np.arange(20, dtype=np.float32).reshape(4, 5)
        y = np.array([10, 20, 30, 40], dtype=np.int64)
        ds = ArrayDataset(X, y)
        x0, _ = ds[0]
        return bool(np.array_equal(x0, X[0]))

    def _array_getitem_y():
        X = np.arange(20, dtype=np.float32).reshape(4, 5)
        y = np.array([10, 20, 30, 40], dtype=np.int64)
        ds = ArrayDataset(X, y)
        _, y2 = ds[2]
        return bool(y2 == y[2])

    def _length_mismatch_raises():
        return _safe(lambda: ArrayDataset(np.zeros((5, 3)), np.zeros(6)))

    # -- dataloader_batching --

    def _len_exact():
        return len(DataLoader(_make_ds(12), batch_size=4))

    def _len_remainder():
        return len(DataLoader(_make_ds(10), batch_size=4))

    def _full_batches_size():
        loader = DataLoader(_make_ds(12), batch_size=4)
        return [b[0].shape[0] for b in loader]

    def _last_batch_smaller():
        loader = DataLoader(_make_ds(10), batch_size=4)
        batches = list(loader)
        return batches[-1][0].shape[0]

    def _covers_all():
        ds = _make_ds(11)
        loader = DataLoader(ds, batch_size=3)
        all_y = np.concatenate([yb for _, yb in loader])
        return bool(np.array_equal(np.sort(all_y), np.arange(11)))

    def _yields_ndarray_pairs():
        loader = DataLoader(_make_ds(), batch_size=4)
        X_batch, y_batch = next(iter(loader))
        return [
            isinstance(X_batch, np.ndarray),
            isinstance(y_batch, np.ndarray),
        ]

    def _invalid_batch_size_raises():
        return _safe(lambda: DataLoader(_make_ds(), batch_size=0))

    # -- dataloader_shuffle --

    def _no_shuffle_order():
        ds = _make_ds(8)
        loader = DataLoader(ds, batch_size=8)
        _, y_batch = next(iter(loader))
        return bool(np.array_equal(y_batch, np.arange(8)))

    def _shuffle_seed_repro():
        ds = _make_ds(20)
        la = DataLoader(ds, batch_size=20, shuffle=True, seed=42)
        lb = DataLoader(ds, batch_size=20, shuffle=True, seed=42)
        _, ya = next(iter(la))
        _, yb = next(iter(lb))
        return bool(np.array_equal(ya, yb))

    def _shuffle_changes_order():
        ds = _make_ds(20)
        loader = DataLoader(ds, batch_size=20, shuffle=True, seed=0)
        _, y_batch = next(iter(loader))
        return bool(not np.array_equal(y_batch, np.arange(20)))

    def _shuffle_advances_rng():
        ds = _make_ds(20)
        loader = DataLoader(ds, batch_size=20, shuffle=True, seed=0)
        _, y1 = next(iter(loader))
        _, y2 = next(iter(loader))
        return bool(not np.array_equal(y1, y2))

    return _R(
        {
            "dataset": [
                _abstract_raises(),
                _safe(_array_len),
                _safe(_array_getitem_x),
                _safe(_array_getitem_y),
                _length_mismatch_raises(),
            ],
            "dataloader_batching": [
                _safe(_len_exact),
                _safe(_len_remainder),
                _safe(_full_batches_size),
                _safe(_last_batch_smaller),
                _safe(_covers_all),
                _safe(_yields_ndarray_pairs),
                _invalid_batch_size_raises(),
            ],
            "dataloader_shuffle": [
                _safe(_no_shuffle_order),
                _safe(_shuffle_seed_repro),
                _safe(_shuffle_changes_order),
                _safe(_shuffle_advances_rng),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task i — Metrics  (src/evaluation/accuracy.py, confusion_matrix.py)
# ---------------------------------------------------------------------------


def _run_i():
    from evaluation.accuracy import accuracy
    from evaluation.confusion_matrix import (
        ConfusionMatrix,
        f1,
        precision,
        recall,
    )

    # -- accuracy --

    def _acc_perfect():
        return accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))

    def _acc_half():
        return accuracy(np.array([0, 1, 1, 1]), np.array([0, 1, 0, 0]))

    def _acc_zero():
        return accuracy(np.array([1, 2, 0]), np.array([0, 1, 2]))

    def _acc_shape_mismatch():
        return _safe(lambda: accuracy(np.array([0, 1]), np.array([0, 1, 2])))

    def _acc_empty():
        return _safe(
            lambda: accuracy(np.array([], dtype=int), np.array([], dtype=int))
        )

    # -- confusion_matrix_update --

    def _cm_known_entries():
        cm = ConfusionMatrix(num_classes=3)
        cm.update(
            np.array([[0.1, 0.9, 0.0], [0.1, 0.8, 0.1], [0.9, 0.0, 0.1]]),
            np.array([0, 1, 2]),
        )
        expected = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
        return bool(np.array_equal(cm.compute(), expected))

    def _cm_perfect():
        cm = ConfusionMatrix(num_classes=3)
        cm.update(np.eye(3), np.array([0, 1, 2]))
        return bool(np.array_equal(cm.compute(), np.eye(3, dtype=np.int64)))

    def _cm_all_wrong():
        cm = ConfusionMatrix(num_classes=2)
        cm.update(
            np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]),
            np.array([0, 0, 0]),
        )
        expected = np.array([[0, 3], [0, 0]], dtype=np.int64)
        return bool(np.array_equal(cm.compute(), expected))

    def _cm_accumulates():
        cm = ConfusionMatrix(num_classes=2)
        cm.update(np.array([[0.9, 0.1]]), np.array([0]))
        cm.update(np.array([[0.1, 0.9]]), np.array([1]))
        cm.update(np.array([[0.1, 0.9]]), np.array([0]))
        expected = np.array([[1, 1], [0, 1]], dtype=np.int64)
        return bool(np.array_equal(cm.compute(), expected))

    def _cm_shape():
        cm = ConfusionMatrix(num_classes=4)
        cm.update(np.eye(4), np.arange(4))
        return list(cm.compute().shape)

    # -- confusion_matrix_metrics --
    # _CM_MIXED = [[3,1],[2,4]]
    # precision: [3/5, 4/5]  recall: [3/4, 4/6]

    _CM_MIXED = np.array([[3, 1], [2, 4]], dtype=np.int64)

    def _precision_known():
        p = precision(_CM_MIXED)
        return bool(np.allclose(p, [3 / 5, 4 / 5]))

    def _precision_zero_col():
        cm = np.array([[2, 0], [3, 0]], dtype=np.int64)
        return float(precision(cm, class_label=1))

    def _recall_known():
        r = recall(_CM_MIXED)
        return bool(np.allclose(r, [3 / 4, 4 / 6]))

    def _recall_zero_row():
        cm = np.array([[2, 1], [0, 0]], dtype=np.int64)
        return float(recall(cm, class_label=1))

    def _f1_known():
        p = np.array([3 / 5, 4 / 5])
        r = np.array([3 / 4, 4 / 6])
        expected = 2 * p * r / (p + r)
        result = f1(_CM_MIXED)
        return bool(np.allclose(result, expected))

    def _f1_both_zero():
        cm = np.array([[0, 0], [0, 0]], dtype=np.int64)
        return bool(np.allclose(f1(cm), [0.0, 0.0]))

    def _single_class_precision():
        return float(precision(_CM_MIXED, class_label=0))

    def _single_class_recall():
        return float(recall(_CM_MIXED, class_label=1))

    def _single_class_f1():
        result = f1(_CM_MIXED, class_label=0)
        p, r = 3 / 5, 3 / 4
        return bool(abs(result - 2 * p * r / (p + r)) < 1e-9)

    return _R(
        {
            "accuracy": [
                _safe(_acc_perfect),
                _safe(_acc_half),
                _safe(_acc_zero),
                _acc_shape_mismatch(),
                _acc_empty(),
            ],
            "confusion_matrix_update": [
                _safe(_cm_known_entries),
                _safe(_cm_perfect),
                _safe(_cm_all_wrong),
                _safe(_cm_accumulates),
                _safe(_cm_shape),
            ],
            "confusion_matrix_metrics": [
                _safe(_precision_known),
                _safe(_precision_zero_col),
                _safe(_recall_known),
                _safe(_recall_zero_row),
                _safe(_f1_known),
                _safe(_f1_both_zero),
                _safe(_single_class_precision),
                _safe(_single_class_recall),
                _safe(_single_class_f1),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Task j — Training  (src/training/train_ffnn.py) + answers
# ---------------------------------------------------------------------------


def _run_j():
    from data.dataloader import DataLoader
    from data.dataset import ArrayDataset
    from evaluation.accuracy import Accuracy
    from evaluation.metric import MetricCollection
    from nn.loss import CrossEntropy
    from nn.mlp import MLP
    from optim.sgd import SGD
    from training.train_ffnn import run_epoch, train_ffnn

    def _make_ds_j(n=32, seed=0):
        X, y = _make_synthetic(seed=seed)
        return ArrayDataset(X[:n], y[:n])

    def _make_model_j(seed=0):
        return MLP([2, 16, 3], activation="relu", seed=seed)

    # -- run_epoch --

    def _epoch_train_changes_params():
        ds = _make_ds_j()
        model = _make_model_j(seed=0)
        loader = DataLoader(ds, batch_size=8)
        loss_fn = CrossEntropy()
        optimizer = SGD(model.parameters(), lr=0.1)
        W_before = model.net.layers[0].W.copy()
        run_epoch(model, loader, loss_fn, optimizer=optimizer)
        return bool(not np.array_equal(model.net.layers[0].W, W_before))

    def _epoch_eval_no_change():
        ds = _make_ds_j()
        model = _make_model_j(seed=0)
        loader = DataLoader(ds, batch_size=8)
        loss_fn = CrossEntropy()
        W_before = model.net.layers[0].W.copy()
        run_epoch(model, loader, loss_fn, optimizer=None)
        return bool(np.array_equal(model.net.layers[0].W, W_before))

    def _epoch_train_sets_mode():
        ds = _make_ds_j()
        model = _make_model_j()
        loader = DataLoader(ds, batch_size=32)
        loss_fn = CrossEntropy()
        optimizer = SGD(model.parameters(), lr=0.01)
        model.eval()
        run_epoch(model, loader, loss_fn, optimizer=optimizer)
        return bool(model.training is True)

    def _epoch_eval_sets_mode():
        ds = _make_ds_j()
        model = _make_model_j()
        loader = DataLoader(ds, batch_size=32)
        loss_fn = CrossEntropy()
        model.train()
        run_epoch(model, loader, loss_fn, optimizer=None)
        return bool(model.training is False)

    def _epoch_loss_finite():
        ds = _make_ds_j()
        model = _make_model_j()
        loader = DataLoader(ds, batch_size=8)
        loss_fn = CrossEntropy()
        result = run_epoch(model, loader, loss_fn)
        return bool(np.isfinite(result["loss"]))

    def _epoch_returns_loss_key():
        ds = _make_ds_j()
        model = _make_model_j()
        loader = DataLoader(ds, batch_size=8)
        loss_fn = CrossEntropy()
        result = run_epoch(model, loader, loss_fn)
        return "loss" in result

    # -- train_loop --

    def _make_history(epochs=3, batch_size=16, seed=0):
        train_ds = _make_ds_j(n=120, seed=seed)
        val_ds = _make_ds_j(n=60, seed=seed + 1)
        model = _make_model_j(seed=seed)
        loss_fn = CrossEntropy()
        metrics = MetricCollection([Accuracy()])
        optimizer = SGD(model.parameters(), lr=0.1)
        return train_ffnn(
            model,
            train_ds,
            val_ds,
            optimizer,
            epochs=epochs,
            batch_size=batch_size,
            loss_fn=loss_fn,
            metrics=metrics,
            seed=seed,
        )

    def _loop_has_keys():
        h = _make_history()
        return [
            "train_loss" in h,
            "val_loss" in h,
            "train_acc" in h,
            "val_acc" in h,
        ]

    def _loop_lists_length():
        epochs = 5
        h = _make_history(epochs=epochs)
        return [len(v) == epochs for v in h.values()]

    def _loop_values_finite():
        h = _make_history()
        return all(bool(np.isfinite(v)) for vals in h.values() for v in vals)

    def _loop_uneven_batches():
        # 180 samples; batch_size=50 → 3 full + 1 tail of 30
        X, y = _make_synthetic(seed=0)
        train_ds = ArrayDataset(X, y)
        val_ds = ArrayDataset(X[:60], y[:60])
        model = _make_model_j()
        loss_fn = CrossEntropy()
        metrics = MetricCollection([Accuracy()])
        optimizer = SGD(model.parameters(), lr=0.1)
        h = train_ffnn(
            model,
            train_ds,
            val_ds,
            optimizer,
            epochs=3,
            batch_size=50,
            loss_fn=loss_fn,
            metrics=metrics,
            seed=0,
        )
        return [
            len(h["train_loss"]) == 3,
            all(bool(np.isfinite(v)) for v in h["train_loss"]),
        ]

    # -- train_convergence --

    def _convergence_val_improves():
        from evaluation.accuracy import accuracy as acc_fn

        X, y = _make_synthetic(seed=0)
        train_ds = ArrayDataset(X[:120], y[:120])
        val_ds = ArrayDataset(X[120:], y[120:])
        model = _make_model_j(seed=0)
        initial_pred = model.forward(X[120:]).argmax(axis=1)
        initial = acc_fn(initial_pred, y[120:])
        loss_fn = CrossEntropy()
        metrics = MetricCollection([Accuracy()])
        optimizer = SGD(model.parameters(), lr=0.1)
        h = train_ffnn(
            model,
            train_ds,
            val_ds,
            optimizer,
            epochs=20,
            batch_size=16,
            loss_fn=loss_fn,
            metrics=metrics,
            seed=0,
        )
        return bool(h["val_acc"][-1] > initial + 0.2)

    def _convergence_overfits():
        X, y = _make_synthetic(seed=0)
        train_ds = ArrayDataset(X, y)
        val_ds = ArrayDataset(X[:60], y[:60])
        model = _make_model_j(seed=0)
        loss_fn = CrossEntropy()
        metrics = MetricCollection([Accuracy()])
        optimizer = SGD(model.parameters(), lr=0.1)
        h = train_ffnn(
            model,
            train_ds,
            val_ds,
            optimizer,
            epochs=40,
            batch_size=16,
            loss_fn=loss_fn,
            metrics=metrics,
            seed=0,
        )
        return bool(h["train_acc"][-1] > 0.95)

    def _convergence_deterministic():
        X, y = _make_synthetic(seed=0)
        train_ds = ArrayDataset(X[:120], y[:120])
        val_ds = ArrayDataset(X[120:], y[120:])

        def _run():
            model = _make_model_j(seed=42)
            loss_fn = CrossEntropy()
            metrics = MetricCollection([Accuracy()])
            optimizer = SGD(model.parameters(), lr=0.1)
            h = train_ffnn(
                model,
                train_ds,
                val_ds,
                optimizer,
                epochs=3,
                batch_size=16,
                loss_fn=loss_fn,
                metrics=metrics,
                seed=7,
            )
            return h["train_loss"]

        ha = _run()
        hb = _run()
        return bool(all(abs(a - b) < 1e-9 for a, b in zip(ha, hb)))

    # answer keys
    try:
        import answers

        j_best_lr = answers.j_best_lr
        j_lr_001_underfits = answers.j_lr_001_underfits
        j_lr_10_unstable = answers.j_lr_10_unstable
    except Exception:
        j_best_lr = j_lr_001_underfits = j_lr_10_unstable = None

    return _R(
        {
            "run_epoch": [
                _safe(_epoch_train_changes_params),
                _safe(_epoch_eval_no_change),
                _safe(_epoch_train_sets_mode),
                _safe(_epoch_eval_sets_mode),
                _safe(_epoch_loss_finite),
                _safe(_epoch_returns_loss_key),
            ],
            "train_loop": [
                _safe(_loop_has_keys),
                _safe(_loop_lists_length),
                _safe(_loop_values_finite),
                _safe(_loop_uneven_batches),
            ],
            "train_convergence": [
                _safe(_convergence_val_improves),
                _safe(_convergence_overfits),
                _safe(_convergence_deterministic),
            ],
            "j_best_lr": [j_best_lr],
            "j_lr_001_underfits": [j_lr_001_underfits],
            "j_lr_10_unstable": [j_lr_10_unstable],
        }
    )


# ---------------------------------------------------------------------------
# Tasks k–p — observe-only answer-key runners
# ---------------------------------------------------------------------------


def _run_k():
    try:
        import answers

        k_observations = answers.k_observations
        k_num = answers.k_num_datasets_linear_solves
        k_smallest = answers.k_smallest_arch_solving_checkerboard
    except Exception:
        k_observations = k_num = k_smallest = None
    return _R(
        {
            "k_observations": [k_observations],
            "k_num_datasets_linear_solves": [k_num],
            "k_smallest_arch_solving_checkerboard": [k_smallest],
        }
    )


def _run_l():
    try:
        import answers

        l_observations = answers.l_observations
        l_highest = answers.l_condition_with_highest_peak_grad_norm
        l_lowest_upd = answers.l_condition_with_lowest_update_magnitude
        l_lowest_gn = answers.l_condition_with_lowest_gradient_norm
    except Exception:
        l_observations = l_highest = l_lowest_upd = l_lowest_gn = None
    return _R(
        {
            "l_observations": [l_observations],
            "l_condition_with_highest_peak_grad_norm": [l_highest],
            "l_condition_with_lowest_update_magnitude": [l_lowest_upd],
            "l_condition_with_lowest_gradient_norm": [l_lowest_gn],
        }
    )


def _run_m():
    try:
        import answers

        m_observations = answers.m_observations
        m_best = answers.m_best_generalizing_config
        m_lowest = answers.m_lowest_train_mse_config
    except Exception:
        m_observations = m_best = m_lowest = None
    return _R(
        {
            "m_observations": [m_observations],
            "m_best_generalizing_config": [m_best],
            "m_lowest_train_mse_config": [m_lowest],
        }
    )


def _run_n():
    try:
        import answers

        n_observations = answers.n_observations
        n_best_lr = answers.n_best_lr_for_raw
        n_best_scaling = answers.n_best_scaling
    except Exception:
        n_observations = n_best_lr = n_best_scaling = None
    return _R(
        {
            "n_observations": [n_observations],
            "n_best_lr_for_raw": [n_best_lr],
            "n_best_scaling": [n_best_scaling],
        }
    )


def _run_o():
    try:
        import answers

        o_observations = answers.o_observations
        o_fastest = answers.o_fastest_optimizer
        o_best_reg = answers.o_best_regularizer_for_val
    except Exception:
        o_observations = o_fastest = o_best_reg = None
    return _R(
        {
            "o_observations": [o_observations],
            "o_fastest_optimizer": [o_fastest],
            "o_best_regularizer_for_val": [o_best_reg],
        }
    )


def _run_p():
    try:
        import answers

        p_observations = answers.p_observations
        p_side = answers.p_filter_image_side
        p_more = answers.p_more_units_higher_test_accuracy
    except Exception:
        p_observations = p_side = p_more = None
    return _R(
        {
            "p_observations": [p_observations],
            "p_filter_image_side": [p_side],
            "p_more_units_higher_test_accuracy": [p_more],
        }
    )


# ---------------------------------------------------------------------------
# Registry and entry points
# ---------------------------------------------------------------------------


_RUNNERS = {
    "b": _run_b,
    "c": _run_c,
    "d": _run_d,
    "e": _run_e,
    "f": _run_f,
    "g": _run_g,
    "h": _run_h,
    "i": _run_i,
    "j": _run_j,
    "k": _run_k,
    "l": _run_l,
    "m": _run_m,
    "n": _run_n,
    "o": _run_o,
    "p": _run_p,
}

_SCHEMA_VERSION = 1
_HOMEWORK_ID = "a4-neural-networks-in-numpy"

_ANSWERS_KEYS = frozenset(
    {
        "j_best_lr",
        "j_lr_001_underfits",
        "j_lr_10_unstable",
        "k_observations",
        "k_num_datasets_linear_solves",
        "k_smallest_arch_solving_checkerboard",
        "l_observations",
        "l_condition_with_highest_peak_grad_norm",
        "l_condition_with_lowest_update_magnitude",
        "l_condition_with_lowest_gradient_norm",
        "m_observations",
        "m_best_generalizing_config",
        "m_lowest_train_mse_config",
        "n_observations",
        "n_best_lr_for_raw",
        "n_best_scaling",
        "o_observations",
        "o_fastest_optimizer",
        "o_best_regularizer_for_val",
        "p_observations",
        "p_filter_image_side",
        "p_more_units_higher_test_accuracy",
    }
)


def _is_exempt(key):
    return key in _ANSWERS_KEYS


def run_puzzles(keys):
    _ref_mode = False
    try:
        import nn.linear as _m

        if getattr(_m, "_c4f7b2a1", None) is True:
            _ref_mode = True
    except Exception:
        pass

    tests = {}
    for k in keys:
        try:
            runner_results = _RUNNERS[k]().items()
        except Exception as e:
            exc_str = f"EXCEPTION:{type(e).__name__}"
            runner_results = {f"{k}--runner-exception": _hash(exc_str)}.items()
        for test_id, hash_val in runner_results:
            key = test_id.replace("_", "-")
            if _ref_mode and not _is_exempt(test_id):
                tests[key] = "REFERENCE_MODE"
            else:
                tests[key] = f"sha256:{hash_val}"
    return {
        "schema_version": _SCHEMA_VERSION,
        "homework_id": _HOMEWORK_ID,
        "tests": tests,
    }


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

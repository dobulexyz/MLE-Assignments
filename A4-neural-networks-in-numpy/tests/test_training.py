"""Tests for src/training/train_ffnn.py."""

import numpy as np
import pytest

from data.dataset import ArrayDataset
from nn.loss import CrossEntropy
from nn.mlp import MLP
from optim.sgd import SGD
from training.train_ffnn import run_epoch, train_ffnn
from data.dataloader import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n: int = 32, in_features: int = 4, n_classes: int = 3, seed: int = 0
) -> ArrayDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_features)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n)
    return ArrayDataset(X, y)


def _make_model(seed: int = 0) -> MLP:
    return MLP([4, 8, 3], activation="relu", seed=seed)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_run_epoch_train_smoke():
    ds = _make_dataset()
    model = _make_model()
    loader = DataLoader(ds, batch_size=8)
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)
    result = run_epoch(model, loader, loss_fn, optimizer=optimizer)
    assert isinstance(result, dict)
    assert "loss" in result


def test_run_epoch_eval_smoke():
    ds = _make_dataset()
    model = _make_model()
    loader = DataLoader(ds, batch_size=8)
    loss_fn = CrossEntropy()
    result = run_epoch(model, loader, loss_fn, optimizer=None)
    assert isinstance(result, dict)
    assert "loss" in result


def test_train_ffnn_smoke():
    train_ds = _make_dataset(n=20, seed=0)
    val_ds = _make_dataset(n=8, seed=1)
    model = _make_model()
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)
    history = train_ffnn(
        model,
        train_ds,
        val_ds,
        optimizer,
        epochs=2,
        batch_size=8,
        loss_fn=loss_fn,
        seed=0,
    )
    assert isinstance(history, dict)


# ---------------------------------------------------------------------------
# run_epoch: training pass changes parameters
# ---------------------------------------------------------------------------


def test_run_epoch_train_changes_parameters():
    ds = _make_dataset()
    model = _make_model(seed=0)
    loader = DataLoader(ds, batch_size=8)
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.1)

    W_before = model.net.layers[0].W.copy()  # type: ignore[attr-defined]
    run_epoch(model, loader, loss_fn, optimizer=optimizer)
    W_after = model.net.layers[0].W  # type: ignore[attr-defined]

    assert not np.array_equal(W_before, W_after)


def test_run_epoch_eval_does_not_change_parameters():
    ds = _make_dataset()
    model = _make_model(seed=0)
    loader = DataLoader(ds, batch_size=8)
    loss_fn = CrossEntropy()

    W_before = model.net.layers[0].W.copy()  # type: ignore[attr-defined]
    run_epoch(model, loader, loss_fn, optimizer=None)
    W_after = model.net.layers[0].W  # type: ignore[attr-defined]

    assert np.array_equal(W_before, W_after)


# ---------------------------------------------------------------------------
# run_epoch: loss value is finite and positive
# ---------------------------------------------------------------------------


def test_run_epoch_loss_is_finite():
    ds = _make_dataset()
    model = _make_model()
    loader = DataLoader(ds, batch_size=8)
    loss_fn = CrossEntropy()
    result = run_epoch(model, loader, loss_fn)
    assert np.isfinite(result["loss"])
    assert result["loss"] > 0.0


# ---------------------------------------------------------------------------
# run_epoch: model mode is set correctly
# ---------------------------------------------------------------------------


def test_run_epoch_train_sets_training_mode():
    ds = _make_dataset()
    model = _make_model()
    loader = DataLoader(ds, batch_size=32)
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)
    model.eval()
    run_epoch(model, loader, loss_fn, optimizer=optimizer)
    assert model.training is True


def test_run_epoch_eval_sets_eval_mode():
    ds = _make_dataset()
    model = _make_model()
    loader = DataLoader(ds, batch_size=32)
    loss_fn = CrossEntropy()
    model.train()
    run_epoch(model, loader, loss_fn, optimizer=None)
    assert model.training is False


# ---------------------------------------------------------------------------
# train_ffnn: history shape and keys
# ---------------------------------------------------------------------------


def test_train_ffnn_history_has_train_val_loss():
    train_ds = _make_dataset(n=24, seed=0)
    val_ds = _make_dataset(n=8, seed=1)
    model = _make_model()
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)
    history = train_ffnn(
        model,
        train_ds,
        val_ds,
        optimizer,
        epochs=3,
        batch_size=8,
        loss_fn=loss_fn,
        seed=0,
    )
    assert "train_loss" in history
    assert "val_loss" in history


def test_train_ffnn_history_lists_have_length_epochs():
    epochs = 5
    train_ds = _make_dataset(n=24, seed=0)
    val_ds = _make_dataset(n=8, seed=1)
    model = _make_model()
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)
    history = train_ffnn(
        model,
        train_ds,
        val_ds,
        optimizer,
        epochs=epochs,
        batch_size=8,
        loss_fn=loss_fn,
        seed=0,
    )
    for key, values in history.items():
        assert len(values) == epochs, (
            f"{key!r} has {len(values)} entries, expected {epochs}"
        )


def test_train_ffnn_history_values_are_finite():
    train_ds = _make_dataset(n=24, seed=0)
    val_ds = _make_dataset(n=8, seed=1)
    model = _make_model()
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)
    history = train_ffnn(
        model,
        train_ds,
        val_ds,
        optimizer,
        epochs=3,
        batch_size=8,
        loss_fn=loss_fn,
        seed=0,
    )
    for key, values in history.items():
        for v in values:
            assert np.isfinite(v), f"{key!r} contains non-finite value {v}"


# ---------------------------------------------------------------------------
# train_ffnn: determinism under fixed seed
# ---------------------------------------------------------------------------


def test_train_ffnn_deterministic_under_fixed_seed():
    train_ds = _make_dataset(n=24, seed=0)
    val_ds = _make_dataset(n=8, seed=1)

    def _run() -> list[float]:
        model = _make_model(seed=42)
        loss_fn = CrossEntropy()
        optimizer = SGD(model.parameters(), lr=0.01)
        history = train_ffnn(
            model,
            train_ds,
            val_ds,
            optimizer,
            epochs=3,
            batch_size=8,
            loss_fn=loss_fn,
            seed=7,
        )
        return history["train_loss"]

    assert _run() == pytest.approx(_run())


# ---------------------------------------------------------------------------
# train_ffnn: training loss generally decreases over many epochs
# ---------------------------------------------------------------------------


def test_train_ffnn_loss_decreases():
    """Final training loss should be lower than initial on an easy dataset."""
    train_ds = _make_dataset(n=64, seed=0)
    val_ds = _make_dataset(n=16, seed=1)
    model = _make_model(seed=0)
    loss_fn = CrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.1)
    history = train_ffnn(
        model,
        train_ds,
        val_ds,
        optimizer,
        epochs=20,
        batch_size=16,
        loss_fn=loss_fn,
        seed=0,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]

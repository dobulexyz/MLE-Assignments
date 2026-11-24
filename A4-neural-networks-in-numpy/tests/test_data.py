import numpy as np
import pytest

from data.dataloader import DataLoader
from data.dataset import ArrayDataset, Dataset
from data.utils import Subset, random_split


def _make_dataset(n: int = 12, features: int = 4) -> ArrayDataset:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, features)).astype(np.float32)
    y = np.arange(n, dtype=np.int64)
    return ArrayDataset(X, y)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_is_abstract():
    with pytest.raises(TypeError):
        Dataset()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ArrayDataset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_array_dataset_smoke():
    ds = _make_dataset()
    assert isinstance(ds, Dataset)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_array_dataset_len():
    assert len(_make_dataset(n=7)) == 7


def test_array_dataset_getitem_x():
    X = np.arange(20, dtype=np.float32).reshape(4, 5)
    y = np.array([10, 20, 30, 40], dtype=np.int64)
    ds = ArrayDataset(X, y)
    x0, _ = ds[0]
    assert np.array_equal(x0, X[0])


def test_array_dataset_getitem_y():
    X = np.arange(20, dtype=np.float32).reshape(4, 5)
    y = np.array([10, 20, 30, 40], dtype=np.int64)
    ds = ArrayDataset(X, y)
    _, y2 = ds[2]
    assert y2 == y[2]


def test_array_dataset_length_mismatch_raises():
    with pytest.raises(ValueError):
        ArrayDataset(np.zeros((5, 3)), np.zeros(6))


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_dataloader_smoke():
    loader = DataLoader(_make_dataset(), batch_size=4)
    batches = list(loader)
    assert len(batches) > 0


def test_dataloader_yields_ndarray_pairs():
    loader = DataLoader(_make_dataset(), batch_size=4)
    X_batch, y_batch = next(iter(loader))
    assert isinstance(X_batch, np.ndarray)
    assert isinstance(y_batch, np.ndarray)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_dataloader_len_exact_division():
    loader = DataLoader(_make_dataset(n=12), batch_size=4)
    assert len(loader) == 3


def test_dataloader_len_with_remainder():
    loader = DataLoader(_make_dataset(n=10), batch_size=4)
    assert len(loader) == 3


def test_dataloader_full_batches_have_correct_size():
    loader = DataLoader(_make_dataset(n=12), batch_size=4)
    for X_batch, y_batch in loader:
        assert X_batch.shape[0] == 4
        assert y_batch.shape[0] == 4


def test_dataloader_last_batch_smaller():
    loader = DataLoader(_make_dataset(n=10), batch_size=4)
    batches = list(loader)
    assert batches[-1][0].shape[0] == 2


def test_dataloader_covers_all_samples():
    ds = _make_dataset(n=11)
    loader = DataLoader(ds, batch_size=3)
    all_y = np.concatenate([yb for _, yb in loader])
    assert np.array_equal(np.sort(all_y), np.arange(11))


def test_dataloader_no_shuffle_preserves_order():
    ds = _make_dataset(n=8)
    loader = DataLoader(ds, batch_size=8)
    _, y_batch = next(iter(loader))
    assert np.array_equal(y_batch, np.arange(8))


def test_dataloader_shuffle_seed_reproducible():
    ds = _make_dataset(n=20)
    loader_a = DataLoader(ds, batch_size=20, shuffle=True, seed=42)
    loader_b = DataLoader(ds, batch_size=20, shuffle=True, seed=42)
    _, ya = next(iter(loader_a))
    _, yb = next(iter(loader_b))
    assert np.array_equal(ya, yb)


def test_dataloader_shuffle_changes_order():
    ds = _make_dataset(n=20)
    loader = DataLoader(ds, batch_size=20, shuffle=True, seed=0)
    _, y_batch = next(iter(loader))
    assert not np.array_equal(y_batch, np.arange(20))


def test_dataloader_shuffle_advances_rng_each_epoch():
    ds = _make_dataset(n=20)
    loader = DataLoader(ds, batch_size=20, shuffle=True, seed=0)
    _, y_epoch1 = next(iter(loader))
    _, y_epoch2 = next(iter(loader))
    assert not np.array_equal(y_epoch1, y_epoch2)


def test_dataloader_invalid_batch_size_raises():
    with pytest.raises(ValueError):
        DataLoader(_make_dataset(), batch_size=0)


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------


def test_subset_len():
    ds = _make_dataset(n=10)
    sub = Subset(ds, [3, 1, 4, 1, 5])
    assert len(sub) == 5


def test_subset_getitem_maps_through_indices():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.int64)
    ds = ArrayDataset(X, y)
    sub = Subset(ds, [3, 1, 4])
    for pos, orig in enumerate([3, 1, 4]):
        x_sub, y_sub = sub[pos]
        assert np.array_equal(x_sub, X[orig])
        assert y_sub == y[orig]


def test_subset_is_dataset_instance():
    ds = _make_dataset(n=5)
    assert isinstance(Subset(ds, [0, 1]), Dataset)


def test_subset_works_with_dataloader():
    ds = _make_dataset(n=10)
    sub = Subset(ds, list(range(6)))
    loader = DataLoader(sub, batch_size=3)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0].shape[0] == 3


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------


def test_random_split_lengths():
    ds = _make_dataset(n=10)
    a, b = random_split(ds, [7, 3], seed=0)
    assert len(a) == 7
    assert len(b) == 3


def test_random_split_covers_all_indices():
    ds = _make_dataset(n=20)
    a, b = random_split(ds, [12, 8], seed=0)
    all_idx = sorted(a.indices + b.indices)
    assert all_idx == list(range(20))


def test_random_split_seeded_reproducible():
    ds = _make_dataset(n=20)
    a1, _ = random_split(ds, [15, 5], seed=42)
    a2, _ = random_split(ds, [15, 5], seed=42)
    assert a1.indices == a2.indices


def test_random_split_different_seeds_differ():
    ds = _make_dataset(n=20)
    a1, _ = random_split(ds, [15, 5], seed=0)
    a2, _ = random_split(ds, [15, 5], seed=1)
    assert a1.indices != a2.indices


def test_random_split_sum_mismatch_raises():
    ds = _make_dataset(n=10)
    with pytest.raises(ValueError):
        random_split(ds, [6, 6], seed=0)


def test_random_split_zero_length_raises():
    ds = _make_dataset(n=10)
    with pytest.raises(ValueError):
        random_split(ds, [0, 10], seed=0)


def test_random_split_three_way():
    ds = _make_dataset(n=12)
    a, b, c = random_split(ds, [6, 4, 2], seed=0)
    assert len(a) == 6
    assert len(b) == 4
    assert len(c) == 2
    all_idx = sorted(a.indices + b.indices + c.indices)
    assert all_idx == list(range(12))


# ---------------------------------------------------------------------------
# Toy dataset class smoke tests
# ---------------------------------------------------------------------------


def test_toy_moons():
    from datasets.toy import Moons

    ds = Moons(n=20, seed=0)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (2,)
    assert y.dtype == np.int64


def test_toy_circles():
    from datasets.toy import Circles

    ds = Circles(n=20, seed=0)
    assert len(ds) == 20
    x, _ = ds[0]
    assert x.shape == (2,)


def test_toy_xor():
    from datasets.toy import XOR

    ds = XOR(n=20, seed=0)
    assert len(ds) == 20


def test_toy_linear():
    from datasets.toy import Linear

    ds = Linear(n=20, seed=0)
    assert len(ds) == 20


def test_toy_checkerboard():
    from datasets.toy import Checkerboard

    ds = Checkerboard(n=20, seed=0)
    assert len(ds) == 20


def test_toy_spiral():
    from datasets.toy import Spiral

    ds = Spiral(n=20, seed=0)
    assert len(ds) == 20


def test_toy_blobs():
    from datasets.toy import Blobs

    ds = Blobs(n=20, seed=0)
    assert len(ds) == 20


def test_toy_curve():
    from datasets.toy import Curve

    ds = Curve(n=20, seed=0)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (1,)
    assert y.shape == (1,)

"""
DataLoader: wraps a Dataset and yields mini-batches.

Mirrors the interface of `torch.utils.data.DataLoader`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import math
from collections.abc import Iterator

import numpy as np

from .dataset import ArrayDataset, Dataset


class DataLoader:
    """
    Iterates over a `Dataset` in mini-batches.

    Constructor arguments:
        dataset    -- any Dataset instance
        batch_size -- number of samples per batch; must be >= 1
        shuffle    -- if True, reshuffle indices at the start of each
                      iteration
        seed       -- seed for the RNG used when shuffling

    Raises ValueError if batch_size < 1.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng: np.random.Generator | None = (
            np.random.default_rng(seed) if shuffle else None
        )

    def __len__(self) -> int:
        """Returns the number of batches."""
        ### BEGIN SOLUTION
        return math.ceil(len(self.dataset) / self.batch_size)
        ### END SOLUTION

    def __iter__(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yields (X_batch, y_batch) pairs, one mini-batch at a time."""
        ### BEGIN SOLUTION
        indices = np.arange(len(self.dataset))
        if self.shuffle and self.rng is not None:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if isinstance(self.dataset, ArrayDataset):
                # Fast path: vectorized indexing avoids a Python loop
                # over the batch.
                yield self.dataset.X[batch_idx], self.dataset.y[batch_idx]
                continue
            samples = [self.dataset[int(i)] for i in batch_idx]
            xs = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            yield np.stack(xs), np.stack(ys)
        ### END SOLUTION

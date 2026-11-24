"""
Dataset abstractions.

`Dataset` is an abstract base class that defines the two-method
interface every dataset must satisfy: `__len__` and `__getitem__`.
`ArrayDataset` is a concrete subclass backed by a pair of NumPy arrays.

Mirrors `torch.utils.data.Dataset` and `torch.utils.data.TensorDataset`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the sample at position `idx` as (x, y)."""
        ...


class ArrayDataset(Dataset):
    """
    Dataset backed by two NumPy arrays.

    Constructor arguments:
        X -- feature array of shape (n, ...)
        y -- label array of shape (n,)

    Raises ValueError if len(X) != len(y).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        ### BEGIN SOLUTION
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length, got {len(X)} and {len(y)}"
            )
        self.X = X
        self.y = y
        ### END SOLUTION

    def __len__(self) -> int:
        ### BEGIN SOLUTION
        return len(self.X)
        ### END SOLUTION

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        ### BEGIN SOLUTION
        return self.X[idx], self.y[idx]
        ### END SOLUTION

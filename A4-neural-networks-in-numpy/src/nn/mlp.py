"""
Multi-layer perceptron (MLP) convenience wrapper.

Builds a fully-connected feed-forward network from a list of layer
sizes and an activation module, then delegates all forward/backward
logic to a `Sequential`.

    mlp = MLP([784, 128, 64, 10], ReLU())
    logits = mlp(x)
"""

from collections.abc import Iterable

import numpy as np

from typing import Callable

from .activations import GELU, LeakyReLU, ReLU
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .sequential import Sequential


class _Identity(Module):
    """Pass-through activation for linear (no-activation) stacks."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout


class MLP(Module):
    """
    Fully-connected network defined by a list of layer widths and a
    unique activation function instance inserted between every pair of
    Linear layers.

    Constructor arguments:
        layer_sizes -- list of integers [in, h1, h2, ..., out];
                       must have at least two elements
        activation  -- string name of the activation function
                       (e.g. "relu", "leaky_relu", "gelu", "identity");
                       no activation is applied after the final Linear layer
        dropout     -- drop probability applied after each hidden-layer
                       activation; 0.0 disables dropout entirely
        seed        -- optional base seed; each Linear layer receives
                       seed + i, and each Dropout layer receives
                       seed + len(layer_sizes) + i, so initializations
                       and dropout masks are independent and reproducible
    """

    _ACTIVATION_FACTORIES: dict[str, Callable[[], Module]] = {
        "relu": ReLU,
        "leaky_relu": LeakyReLU,
        "gelu": GELU,
        "identity": _Identity,
    }

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        ### BEGIN SOLUTION
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must have at least two elements "
                "(input size and output size)"
            )

        factory = self._ACTIVATION_FACTORIES.get(activation.lower())
        if factory is None:
            raise ValueError(f"Unknown activation: {activation!r}")

        layers: list[Module] = []
        for i, (in_f, out_f) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layers.append(
                Linear(in_f, out_f, seed=None if seed is None else seed + i)
            )
            if i < len(layer_sizes) - 2:
                layers.append(factory())
                if dropout > 0.0:
                    dropout_seed = (
                        None if seed is None else seed + len(layer_sizes) + i
                    )
                    layers.append(Dropout(dropout, seed=dropout_seed))
        self.net = Sequential(layers)
        ### END SOLUTION

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.net.forward(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.net.backward(dout)

    def parameters(
        self,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        return self.net.parameters()

    def train(self) -> None:
        super().train()
        self.net.train()

    def eval(self) -> None:
        super().eval()
        self.net.eval()

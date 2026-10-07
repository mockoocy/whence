"""What the harness and the strategies need from a model."""

from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F  # noqa: N812 -- such is the convention for torch.

if TYPE_CHECKING:
    from pathlib import Path

    from whence.forecaster import LstmForecaster


class Model(Protocol):
    def errors(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, sample_len, n_features) -> (B, n_scored, n_targets), non-negative."""
        ...

    def fit(self, windows: torch.Tensor) -> None:
        """One round of learning from `windows`, e.g. one epoch. The first call
        may also fit what the model derives from its data, such as input scaling.
        """
        ...

    def loss(self, windows: torch.Tensor) -> float:
        """Held-out quality on `windows`, lower is better. Early stopping and the
        validation gate compare models by it.
        """
        ...

    def state(self) -> object: ...

    def restore(self, state: object) -> None: ...

    def save(self, path: Path) -> None:
        """Writes what it takes to rebuild the model to `path`."""
        ...

    def load(self, path: Path) -> None:
        """Restores what `save` wrote into a model built the same way."""
        ...


class Objective(Protocol):
    def __call__(self, model: LstmForecaster, windows: torch.Tensor) -> torch.Tensor:
        """Differentiable training loss on a batch of raw windows."""
        ...


class Mse:
    def __call__(self, model: LstmForecaster, windows: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(*model.predict(windows))

"""What the harness and the strategies need from a model."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    import torch


class Model(Protocol):
    def outputs(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(output, target): what the model produced for `windows` and what it
        should have produced, e.g. a forecast and the rows that followed, or a
        reconstruction and its input.
        """
        ...

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        """Anomaly scores per channel, non-negative:
        (B, sample_len, n_features) -> (B, n_scored, n_targets).
        """
        ...

    def fit(self, windows: torch.Tensor) -> None:
        """One round of learning from `windows`, e.g. one epoch."""
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

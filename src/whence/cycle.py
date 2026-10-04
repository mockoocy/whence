"""One adaptation cycle: which windows are held out for validation (Holdout) and
how the model is trained on the rest (Fit). Initial training is the first cycle.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from whence.events import Window
    from whence.model import Model


class Holdout:
    """Holds out the newest `val_fraction` of the rows each cycle covers, and keeps
    every window held out so far. Windows that only overlap a held-out region are
    dropped, since training on them would leak validation rows. With
    `val_fraction=0` nothing new is held out, but the starting set stays.
    """

    def __init__(self, val_fraction: float) -> None:
        self._val_fraction = val_fraction
        self._chunks: list[torch.Tensor] = []
        self._regions: list[tuple[int, int]] = []  # inclusive row ranges
        self._interval_start = 0

    def start(self, validation: torch.Tensor) -> None:
        self._chunks = [validation] if len(validation) else []

    def split(self, windows: list[Window], now: int) -> list[Window]:
        """Of the windows admitted this cycle, the ones to train on. `now` is the
        newest row streamed so far.
        """
        start = self._interval_start
        boundary = start + int((now - start + 1) * (1 - self._val_fraction))
        if boundary <= now:
            self._regions.append((boundary, now))
        self._interval_start = now + 1

        train, validation = [], []
        for window in windows:
            if any(
                a <= window.first_row and window.last_row <= b for a, b in self._regions
            ):
                validation.append(window)
            elif not any(
                window.first_row <= b and a <= window.last_row for a, b in self._regions
            ):
                train.append(window)
        if validation:
            self._chunks.append(torch.stack([w.data for w in validation]))
        return train

    def windows(self) -> torch.Tensor:
        """Every window held out so far."""
        if not self._chunks:
            msg = "no window has been held out"
            raise ValueError(msg)
        return torch.cat(self._chunks)


class Fit(Protocol):
    def __call__(
        self,
        model: Model,
        sample: Callable[[], torch.Tensor],
        validation: torch.Tensor,
    ) -> bool:
        """Trains `model` on windows from `sample`; returns whether the result was
        kept.
        """
        ...


@dataclass(frozen=True)
class Epochs:
    """A fixed number of epochs on one sample, always kept."""

    epochs: int = 1

    def __call__(
        self,
        model: Model,
        sample: Callable[[], torch.Tensor],
        validation: torch.Tensor,
    ) -> bool:
        windows = sample()
        for _ in range(self.epochs):
            model.fit(windows)
        return True


@dataclass(frozen=True)
class EarlyStopping:
    """Trains on a fresh sample each epoch until the validation loss has not
    improved by `delta` for `patience` epochs, then restores the best weights.

    Gated, the loss before training is the one to beat, as in validation-gated-cl:
    if no epoch beats it, the old weights come back and the update is rejected.
    Ungated (initial training), any first epoch counts as an improvement.
    """

    max_epochs: int
    patience: int
    delta: float
    gated: bool = True
    desc: str | None = None  # progress bar label; no bar without one

    def __call__(
        self,
        model: Model,
        sample: Callable[[], torch.Tensor],
        validation: torch.Tensor,
    ) -> bool:
        best_loss = model.loss(validation) if self.gated else math.inf
        best_state = model.state()
        improved, stale = False, 0
        epochs = range(self.max_epochs)
        for _ in tqdm(epochs, desc=self.desc, unit="epoch", disable=self.desc is None):
            model.fit(sample())
            loss = model.loss(validation)
            if best_loss - loss > self.delta:
                best_loss, best_state, improved, stale = loss, model.state(), True, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        model.restore(best_state)
        return improved

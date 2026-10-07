"""One adaptation cycle: which windows are held out for validation (Holdout) and
how the model is trained on the rest (EarlyStopping).
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from whence.events import Window
    from whence.model import Model


class Holdout:
    """Holds out the newest `val_fraction` of the rows each cycle covers, and keeps
    every window held out so far. Windows that only overlap a held-out region are
    dropped, since training on them would leak validation rows.
    """

    def __init__(self, val_fraction: float) -> None:
        self._val_fraction = val_fraction
        self._chunks: list[torch.Tensor] = []
        self._regions: list[tuple[int, int]] = []  # inclusive row ranges
        self._interval_start = 0

    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self._chunks)

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


@dataclass(frozen=True)
class EarlyStopping:
    """Trains on a fresh sample each epoch until the validation loss has not
    improved by `delta` for `patience` epochs, then restores the best weights.
    Any first epoch counts as an improvement; whether the result beats the weights
    before training is the strategy's call (see Replay's gate).
    """

    max_epochs: int
    patience: int
    delta: float
    desc: str | None = None  # progress bar label; no bar without one

    def __call__(
        self,
        model: Model,
        sample: Callable[[], torch.Tensor],
        validation: torch.Tensor,
    ) -> None:
        best_loss, best_state, stale = math.inf, model.state(), 0
        epochs = range(self.max_epochs)
        for _ in tqdm(epochs, desc=self.desc, unit="epoch", disable=self.desc is None):
            model.fit(sample())
            loss = model.loss(validation)
            if best_loss - loss > self.delta:
                best_loss, best_state, stale = loss, model.state(), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        model.restore(best_state)

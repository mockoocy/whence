"""Learning from one concept: which windows are held out for validation (split)
and how the model is trained on the rest (EarlyStopping).
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from whence.events import Window
    from whence.model import Model


def split(
    windows: list[Window],
    val_fraction: float,
    *,
    after: int = -1,
) -> tuple[list[Window], list[Window]]:
    """(train, validation) of time-ordered stride-1 `windows`. Validation tiles the
    newest `val_fraction` of the timesteps they cover with non-overlapping windows;
    training is every window that ends before that and starts after timestep
    `after` (the last one validated before), so no validated timestep is trained on.
    """
    first, last = windows[0].first_row, windows[-1].last_row
    cut = first + int((last - first + 1) * (1 - val_fraction))
    validation, end = [], cut - 1
    for window in windows:
        if window.first_row > end:
            validation.append(window)
            end = window.last_row
    train = [w for w in windows if after < w.first_row and w.last_row < cut]
    return train, validation


@dataclass(frozen=True)
class EarlyStopping:
    """Trains on a fresh sample each epoch until the validation loss has not
    improved by `delta` for `patience` epochs, then restores the best weights.
    Any first epoch counts as an improvement; whether the result beats the weights
    before training is the strategy's call (see ValidationGate).
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

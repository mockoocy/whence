from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 -- such is the convention for torch.
from scipy.signal import lfilter

if TYPE_CHECKING:
    from pathlib import Path

# A channel constant where the statistics come from would otherwise divide by zero.
MIN_STD = 1e-6


class ZScore:
    def __init__(self, *, n_targets: int, alpha: float | None = None) -> None:
        self._n_targets = n_targets
        self._alpha = alpha
        self._mean: torch.Tensor | None = None
        self._var: torch.Tensor | None = None

    def _rows(self, windows: torch.Tensor) -> torch.Tensor:
        return windows[:, -1, : self._n_targets].double()

    def outputs(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self._rows(windows)
        return self._mean.expand_as(rows)[:, None], rows[:, None]

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        assert self._var is not None
        assert self._mean is not None
        rows = self._rows(windows)
        if self._alpha is None:
            mean, var = self._mean, self._var
        else:
            mean, var = self._follow(rows)
        std = var.clamp(min=MIN_STD**2).sqrt()
        return ((rows - mean).abs() / std).float()[:, None]

    def _follow(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._var is not None
        assert self._mean is not None
        assert self._alpha is not None

        start = torch.cat([self._mean, self._var + self._mean**2]).numpy()
        rows_arr = rows.numpy()
        # lfilter is here to vectorize computing emwa statistics.
        ewma = lfilter(
            [self._alpha],
            [1, self._alpha - 1],
            np.hstack([rows_arr, rows_arr**2]),
            axis=0,
            zi=[(1 - self._alpha) * start],
        )[0]
        before = torch.from_numpy(np.vstack([start, ewma[:-1]]))
        mean, square = before.tensor_split(2, dim=1)
        last_mean, last_square = torch.from_numpy(ewma[-1].copy()).tensor_split(2)
        self._mean, self._var = last_mean, last_square - last_mean**2
        return mean, square - mean**2

    def fit(self, windows: torch.Tensor) -> None:
        """Sets the mean and variance to those of the windows' rows."""
        rows = self._rows(windows)
        self._mean, self._var = rows.mean(0), rows.var(0, correction=0)

    def loss(self, windows: torch.Tensor) -> float:
        return F.mse_loss(*self.outputs(windows)).item()

    def state(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._mean, self._var

    def restore(self, state: tuple[torch.Tensor | None, torch.Tensor | None]) -> None:
        self._mean, self._var = state

    def save(self, path: Path) -> None:
        torch.save(self.state(), path)

    def load(self, path: Path) -> None:
        self.restore(torch.load(path))

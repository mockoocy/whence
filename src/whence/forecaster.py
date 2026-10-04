from typing import TYPE_CHECKING

import torch

from whence.lstm import LstmForecasterModule
from whence.model import Mse, Objective

if TYPE_CHECKING:
    from whence.aggregator import RobustScaler


class LstmForecaster:
    def __init__(  # noqa: PLR0913 - keyword-only, mirrors LstmForecasterConfig
        self,
        *,
        n_features: int,
        n_targets: int,
        window_size: int,
        horizon: int,
        hidden_size: tuple[int, int] = (16, 16),
        dropout: float = 0.5,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        self._window_size = window_size
        self._horizon = horizon
        self._n_targets = n_targets
        self._device = torch.device(device)
        self._module = LstmForecasterModule(
            n_features=n_features,
            hidden_size=hidden_size,
            dropout=dropout,
            horizon=horizon,
            n_target=n_targets,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        # what `fit` minimises; a strategy such as DER++ may replace it
        self.objective: Objective = Mse()
        self.input_scaler: RobustScaler | None = None
        # per-channel variance of the scaled training targets, set by initial training
        self.target_variance: torch.Tensor | None = None

    @property
    def sample_len(self) -> int:
        return self._window_size + self._horizon

    @property
    def n_scored(self) -> int:
        return self._horizon

    def _split(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits a sample into context and targets."""
        if self.input_scaler is not None:
            windows = self.input_scaler(windows)
        context = windows[:, : self._window_size, :]
        targets = windows[:, self._window_size :, : self._n_targets]
        return context, targets

    def predict(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(predictions, targets) for raw windows, both (B, horizon, n_targets) in
        scaled units. Runs in whatever mode the module is in: training inside `fit`,
        evaluation elsewhere.
        """
        context, targets = self._split(windows.to(self._device))
        return self._module(context), targets

    def fit(self, windows: torch.Tensor, *, batch_size: int = 64) -> None:
        """One epoch over `windows` in random order, minimising `objective`."""
        self._module.train()
        order = torch.randperm(len(windows))
        for start in range(0, len(windows), batch_size):
            loss = self.objective(self, windows[order[start : start + batch_size]])
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    @torch.no_grad()
    def loss(self, windows: torch.Tensor, *, batch_size: int = 1024) -> float:
        self._module.eval()
        scale = (
            torch.ones(self._n_targets)
            if self.target_variance is None
            else self.target_variance
        ).to(self._device)
        total, count = 0.0, 0
        for start in range(0, len(windows), batch_size):
            predictions, targets = self.predict(windows[start : start + batch_size])
            squared = (predictions - targets) ** 2  # (B, horizon, C)
            total += (squared.sum(dim=(0, 1)) / scale).sum().item()
            count += targets.numel()
        return total / count

    def state(self) -> dict[str, torch.Tensor]:
        """Snapshot of the weights only, like the repo's checkpoint: restoring it
        does not rewind the optimizer's moments.
        """
        return {k: v.detach().clone() for k, v in self._module.state_dict().items()}

    def restore(self, state: dict[str, torch.Tensor]) -> None:
        self._module.load_state_dict(state)

    @torch.no_grad()
    def errors(self, windows: torch.Tensor) -> torch.Tensor:
        """Returns per-channel anomaly scores for a horizon.

        Shapes are transformed:
            (batch, window_size + horizon, features) -> (batch, horizon, channels)
        """
        self._module.eval()
        predictions, targets = self.predict(windows)
        return (predictions - targets).abs().cpu()

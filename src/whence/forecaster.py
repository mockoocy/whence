from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812 -- such is the convention for torch.

from whence.lstm import LstmForecasterModule

if TYPE_CHECKING:
    from pathlib import Path


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
        batch_size: int = 64,
        eval_batch_size: int = 1024,
        device: torch.device | str = "cpu",
    ) -> None:
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._window_size = window_size
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

    def outputs(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(forecasts, targets), both (B, horizon, n_targets). Runs in whatever mode
        the module is in: training inside `fit`, evaluation elsewhere.
        """
        windows = windows.to(self._device)
        context = windows[:, : self._window_size, :]
        targets = windows[:, self._window_size :, : self._n_targets]
        return self._module(context), targets

    @torch.no_grad()
    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        """Per-channel absolute forecast errors:
        (B, window_size + horizon, n_features) -> (B, horizon, n_targets).
        """
        self._module.eval()
        forecasts, targets = self.outputs(windows)
        return (forecasts - targets).abs().cpu()

    def fit(self, windows: torch.Tensor) -> None:
        """One epoch over `windows` in random order, minimising the MSE."""
        self._module.train()
        order = torch.randperm(len(windows))
        for start in range(0, len(windows), self._batch_size):
            batch = windows[order[start : start + self._batch_size]]
            loss = F.mse_loss(*self.outputs(batch))
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    @torch.no_grad()
    def loss(self, windows: torch.Tensor) -> float:
        """Mean squared forecast error over every window, horizon step and channel."""
        self._module.eval()
        total, count = 0.0, 0
        for start in range(0, len(windows), self._eval_batch_size):
            forecasts, targets = self.outputs(
                windows[start : start + self._eval_batch_size],
            )
            total += ((forecasts - targets) ** 2).sum().item()
            count += targets.numel()
        return total / count

    def state(self) -> dict[str, torch.Tensor]:
        """Snapshot of the weights only, like the repo's checkpoint: restoring it
        does not rewind the optimizer's moments.
        """
        return {k: v.detach().clone() for k, v in self._module.state_dict().items()}

    def restore(self, state: dict[str, torch.Tensor]) -> None:
        self._module.load_state_dict(state)

    def save(self, path: Path) -> None:
        torch.save(
            {"weights": self.state(), "optimizer": self._optimizer.state_dict()},
            path,
        )

    def load(self, path: Path) -> None:
        saved = torch.load(path, map_location=self._device)
        self.restore(saved["weights"])
        self._optimizer.load_state_dict(saved["optimizer"])

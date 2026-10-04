from dataclasses import dataclass
from typing import Literal, Self

import torch

Method = Literal["first", "mean", "ewma"]


def horizon_weights(method: Method, horizon: int) -> torch.Tensor:
    """Weight per horizon, index 0 being the one-step-ahead forecast."""
    match method:
        case "first":
            weights = torch.zeros(horizon)
            weights[0] = 1.0
        case "mean":
            weights = torch.ones(horizon)
        case "ewma":
            # favours the near forecasts, which saw the most recent context
            weights = torch.exp(torch.arange(horizon, 0, -1, dtype=torch.float32))
        case _:
            msg = f"unknown aggregation method: {method}"
            raise ValueError(msg)
    return weights


class HorizonAggregator:
    def __init__(
        self,
        *,
        horizon: int,
        n_targets: int,
        method: Method = "first",
    ) -> None:
        self._weights = horizon_weights(method, horizon)
        self._pending = torch.zeros(horizon, n_targets)
        self._weight_sum = torch.zeros(horizon)

    def push(self, error: torch.Tensor) -> torch.Tensor:
        """error: (horizon, n_targets) for the timesteps this sample forecasts.
        Returns (n_targets,) for the oldest of them, which no later sample covers.
        """
        self._pending += self._weights[:, None] * error
        self._weight_sum += self._weights

        # At the start of a stream the earlier samples that would have covered a
        # timestep do not exist. Normalise by the weight that actually arrived.
        completed = self._pending[0] / self._weight_sum[0].clamp(min=1e-12)

        # roll is circular, so the emitted slot wraps to the end; zero it to open
        # a clean slot for the timestep entering the buffer
        self._pending = torch.roll(self._pending, -1, dims=0)
        self._weight_sum = torch.roll(self._weight_sum, -1, dims=0)
        self._pending[-1] = 0.0
        self._weight_sum[-1] = 0.0
        return completed


@dataclass(frozen=True)
class RobustScaler:
    """Per-column robust z-score: (x - median) / (1.4826 * MAD). On model inputs it
    puts every channel on a comparable scale; on errors it keeps channels whose
    errors are typically larger from dominating a row's score.
    """

    median: torch.Tensor  # (columns,)
    scale: torch.Tensor  # (columns,)

    @classmethod
    def fit(cls, values: torch.Tensor) -> Self:
        """values: (rows, columns) of anomaly-free data, e.g. aggregated errors or
        model inputs.
        """
        median = values.median(dim=0).values
        scale = 1.4826 * (values - median).abs().median(dim=0).values
        # forward-filled columns can be mostly constant, i.e. MAD = 0: fall back to
        # the std, and leave a column that never changes unscaled
        scale = torch.where(scale > 0, scale, values.std(dim=0))
        return cls(median, torch.where(scale > 0, scale, torch.ones_like(scale)))

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Scales the last dimension, so (..., columns) of any leading shape."""
        return (values - self.median) / self.scale

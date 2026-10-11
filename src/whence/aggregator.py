from typing import Literal

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
        # the last horizon - 1 samples' errors, which still cover rows to come
        self._tail = torch.zeros(horizon - 1, horizon, n_targets)
        self._seen = torch.zeros(horizon - 1)  # 0 for samples before the stream

    def push(self, errors: torch.Tensor) -> torch.Tensor:
        """errors: (B, horizon, n_targets) of consecutive samples, each forecasting the
        rows from its own on. Returns (B, n_targets): row j combines horizon h of
        sample j - h, so a row is complete once its own sample arrives.
        """
        horizon, n = len(self._weights), len(errors)
        errors = torch.cat([self._tail, errors])
        seen = torch.cat([self._seen, torch.ones(n)])
        total, weight_sum = torch.zeros(n, errors.shape[2]), torch.zeros(n)
        for h, w in enumerate(self._weights):
            sample = slice(horizon - 1 - h, horizon - 1 - h + n)  # j - h, for each j
            # At the start of a stream the samples that would have covered a row do
            # not exist; normalising by the weight that arrived leaves them out.
            weight = w * seen[sample]
            total += weight[:, None] * errors[sample, h]
            weight_sum += weight
        # errors[-0:] would keep every sample when horizon is 1
        self._tail = errors[len(errors) - (horizon - 1) :]
        self._seen = seen[len(seen) - (horizon - 1) :]
        return total / weight_sum.clamp(min=1e-12)[:, None]

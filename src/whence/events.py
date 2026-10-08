from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Window:
    """One stream window, created only after it has been scored, so a strategy that
    learns from it can only affect windows that arrive later.
    """

    step: int
    data: torch.Tensor  # (sample_len, n_features), the shape the replay buffer stores
    labels: torch.Tensor  # (sample_len, n_targets) label codes; 0 is nominal

    @property
    def first_row(self) -> int:
        return self.step  # the stream advances one row per window

    @property
    def last_row(self) -> int:
        return self.first_row + len(self.data) - 1

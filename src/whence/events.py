"""Event-like objects sent to a strategy, to allow for adaptation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Window:
    """Sent only after the window has been scored, so a strategy that trains on it
    can only affect windows that arrive later.
    """

    step: int
    data: torch.Tensor  # (sample_len, n_features), the shape the replay buffer stores
    # (n_targets,) the completed row's errors, e.g. for drift triggers; None for
    # windows nothing scored, such as those of initial training
    scores: torch.Tensor | None = None

    @property
    def first_row(self) -> int:
        return self.step  # the stream advances one row per window

    @property
    def last_row(self) -> int:
        return self.first_row + len(self.data) - 1


@dataclass(frozen=True)
class Label:
    """The label for one row has become available to strategies."""

    row: int
    codes: torch.Tensor  # (n_targets,) per-channel label codes

    @property
    def dirty(self) -> bool:
        # any non-zero code keeps the row out of training, as in validation-gated-cl
        return bool((self.codes > 0).any())

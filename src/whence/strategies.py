from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from whence.admission import Admission
    from whence.aggregator import RobustScaler
    from whence.cycle import Fit, Holdout
    from whence.events import Label, Window
    from whence.model import Model
    from whence.triggers import Trigger
    from whence.window_reservoir import WindowReservoir


@dataclass
class Initial:
    """What initial training hands to every arm."""

    model: Model
    buffer: WindowReservoir  # the reservoir the model was trained from
    validation: torch.Tensor  # V(0): held-out, anomaly-free tail of initial training
    scaler: RobustScaler  # fitted on the model's errors over V(0), shared by every arm


class Strategy:
    model: Model
    refits: list[tuple[int, bool]]  # (step, kept)

    def on_start(self, initial: Initial) -> None:
        self.model = initial.model
        self.refits = []

    def on_window(self, window: Window) -> None:
        pass

    def on_label(self, label: Label) -> None:
        pass


class Frozen(Strategy):
    """Never adapts: the lower bound every other arm has to beat."""


class Replay(Strategy):
    """Reservoir replay: refits on buffer samples whenever the trigger fires."""

    def __init__(  # noqa: PLR0913 - one argument per composed part, plus tuning
        self,
        trigger: Trigger,
        admission: Admission,
        holdout: Holdout,
        fit: Fit,
        *,
        insert_every: int = 32,
        sample_size: int = 4096,
    ) -> None:
        self._trigger = trigger
        self._admission = admission
        self._holdout = holdout
        self._fit = fit
        self._insert_every = insert_every
        self._sample_size = sample_size

    def on_start(self, initial: Initial) -> None:
        super().on_start(initial)
        self._buffer = initial.buffer  # carry on from the initial-training reservoir
        self._recent: list[Window] = []  # admitted since the last refit
        self._holdout.start(initial.validation)

    def on_window(self, window: Window) -> None:
        if window.step % self._insert_every == 0:
            self._recent += self._admission.offer(window)
        if self._trigger(window):
            self._refit(window)

    def on_label(self, label: Label) -> None:
        self._recent += self._admission.label(label)

    def _refit(self, window: Window) -> None:
        train, self._recent = self._holdout.split(self._recent, window.last_row), []
        for admitted in train:
            self._buffer.add(admitted.data.unsqueeze(0))
        kept = self._fit(
            self.model,
            lambda: self._buffer.sample(self._sample_size),
            self._holdout.windows(),
        )
        self.refits.append((window.step, kept))

from typing import TYPE_CHECKING

from whence.window_reservoir import WindowReservoir

if TYPE_CHECKING:
    from whence.admission import Admission
    from whence.cycle import EarlyStopping, Holdout
    from whence.events import Label, Window
    from whence.model import Model
    from whence.triggers import Trigger


class Strategy:
    model: Model
    refits: list[tuple[int, bool]]  # (step, kept)

    def on_start(self, model: Model) -> None:
        """`model` comes pre-trained; the strategy starts with no data of its own."""
        self.model = model
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
        early_stopping: EarlyStopping,
        *,
        insert_every: int = 32,
        sample_size: int = 4096,
        capacity: int = 70_000,
        gate_delta: float | None = None,
    ) -> None:
        self._trigger = trigger
        self._admission = admission
        self._holdout = holdout
        self._early_stopping = early_stopping
        self._insert_every = insert_every
        self._sample_size = sample_size
        self._buffer = WindowReservoir(capacity)
        self._recent: list[Window] = []  # admitted since the last refit
        # None: keep every refit; else a refit has to lower the validation loss by
        # more than this, as in validation-gated-cl
        self._gate_delta = gate_delta

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
        if not self._buffer or not self._holdout:
            return  # nothing to train on or validate against yet
        # ponytail: the gate validates on the same windows early stopping picked
        # the best epoch on, so it leans towards keeping; a separate split fixes it
        validation = self._holdout.windows()
        before, state = self.model.loss(validation), self.model.state()
        self._early_stopping(
            self.model,
            lambda: self._buffer.sample(self._sample_size),
            validation,
        )
        kept = (
            self._gate_delta is None
            or before - self.model.loss(validation) > self._gate_delta
        )
        if not kept:
            self.model.restore(state)
        self.refits.append((window.step, kept))

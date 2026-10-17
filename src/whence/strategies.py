from typing import TYPE_CHECKING, Protocol

import torch

from whence.cycle import split
from whence.window_reservoir import WindowReservoir

if TYPE_CHECKING:
    from whence.cycle import EarlyStopping
    from whence.events import Window
    from whence.model import Model


class Strategy(Protocol):
    """What an arm runs. It owns the model; the harness only asks it to predict,
    and tells it when to learn.
    """

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, sample_len, n_features) -> (B, n_scored, n_targets) errors."""
        ...

    def learn(self, concept: list[Window]) -> bool | None:
        """Called when the arm's admission starts a new concept, with its windows.
        Returns whether the update was kept, None if there was none.
        """
        ...


class Frozen:
    """Never learns: the lower bound every other arm has to beat."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        return self._model.predict(windows)

    def learn(self, concept: list[Window]) -> bool | None:
        return None


class Replay:
    def __init__(
        self,
        model: Model,
        *,
        capacity: int = 70_000,
        sample_size: int = 4096,
        epochs: int = 1,
    ) -> None:
        self._model = model
        self._buffer = WindowReservoir(capacity)
        self._sample_size = sample_size
        self._epochs = epochs

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        return self._model.predict(windows)

    def learn(self, concept: list[Window]) -> bool | None:
        self._buffer.add(w.data for w in concept)
        if not self._buffer:
            return None
        windows = self._buffer.sample(self._sample_size)
        for _ in range(self._epochs):
            self._model.fit(windows)
        return True


class ValidationGate:
    def __init__(  # noqa: PLR0913 - keyword-only tuning
        self,
        model: Model,
        *,
        val_fraction: float,
        early_stopping: EarlyStopping,
        capacity: int = 70_000,
        sample_size: int = 4096,
        gate_delta: float = 1e-5,
    ) -> None:
        self._model = model
        self._val_fraction = val_fraction
        self._validation: list[torch.Tensor] = []  # every concept's, so far
        self._validated_until = -1  # last timestep any validation window covers
        self._early_stopping = early_stopping
        self._buffer = WindowReservoir(capacity)
        self._sample_size = sample_size
        self._gate_delta = gate_delta

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        return self._model.predict(windows)

    def learn(self, concept: list[Window]) -> bool | None:
        if concept:
            train, validation = split(
                concept,
                self._val_fraction,
                after=self._validated_until,
            )
            if validation:
                self._validated_until = validation[-1].last_row
            # labelled (dirty) windows are never trained or validated on, as in
            # validation-gated-cl
            # copies: a view would keep the stream's whole read chunk alive
            self._validation += [w.data.clone() for w in validation if _clean(w)]
            self._buffer.add(w.data for w in train if _clean(w))
        if not self._buffer or not self._validation:
            return None  # nothing to train on or validate against yet
        validation = torch.stack(self._validation)
        before, state = self._model.loss(validation), self._model.state()
        self._early_stopping(
            self._model,
            lambda: self._buffer.sample(self._sample_size),
            validation,
        )
        if before - self._model.loss(validation) > self._gate_delta:
            return True
        self._model.restore(state)
        return False


def _clean(window: Window) -> bool:
    return not (window.labels > 0).any()

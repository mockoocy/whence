from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from whence.events import Window


class Admission(Protocol):
    def offer(self, window: Window) -> list[Window] | None:
        """The windows of a new concept, if `window` completes one, else None."""
        ...


class Every:
    """Starts a new concept every `timesteps` timesteps, handing over every window
    since the last one. Their labels are assumed known by then, as in
    validation-gated-cl.
    """

    def __init__(self, timesteps: int) -> None:
        self._timesteps = timesteps
        self._windows: list[Window] = []  # since the last concept

    def offer(self, window: Window) -> list[Window] | None:
        self._windows.append(window)
        if window.step == 0 or window.step % self._timesteps:
            return None
        concept, self._windows = self._windows, []
        return concept

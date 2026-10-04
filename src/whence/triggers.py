"""Triggers: when a strategy should adapt."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from whence.events import Window


class Trigger(Protocol):
    def __call__(self, window: Window) -> bool: ...


@dataclass(frozen=True)
class Every:
    """Adapts every `windows` windows."""

    windows: int

    def __call__(self, window: Window) -> bool:
        return window.step > 0 and window.step % self.windows == 0

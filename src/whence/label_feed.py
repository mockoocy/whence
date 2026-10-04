"""Holds labels back until operators would have confirmed them.

The delay is measured in rows, i.e. in stream time, so how long a label takes to
arrive does not depend on when a strategy chooses to adapt.
"""

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whence.events import Label


class LabelFeed:
    """Holds each row's label until `delay` rows after that row streamed past.
    `delay=None` never releases anything: the unsupervised setting.
    """

    def __init__(self, delay: int | None = 0) -> None:
        self._delay = delay
        self._queue: deque[Label] = deque()

    def observe(self, label: Label) -> None:
        if self._delay is not None:
            self._queue.append(label)

    def release(self, now: int) -> list[Label]:
        """Labels of rows at least `delay` rows before `now`, oldest first."""
        if self._delay is None:
            return []
        released = []
        while self._queue and self._queue[0].row + self._delay <= now:
            released.append(self._queue.popleft())
        return released

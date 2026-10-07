from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whence.events import Label, Window

# An admission rule's `offer` and `label` both return the windows that become
# admitted as a result.


class AdmitAll:
    """No filtering: every window goes in immediately. The unsupervised setting."""

    def offer(self, window: Window) -> list[Window]:
        return [window]

    def label(self, label: Label) -> list[Window]:
        return []


class CleanWindows:
    def __init__(self) -> None:
        self._pending: deque[Window] = deque()
        self._last_dirty = -1

    def offer(self, window: Window) -> list[Window]:
        self._pending.append(window)
        return []

    def label(self, label: Label) -> list[Window]:
        if label.dirty:
            self._last_dirty = label.row
        clean = []
        while self._pending and self._pending[0].last_row <= label.row:
            window = self._pending.popleft()
            if self._last_dirty < window.first_row:
                clean.append(window)
        return clean

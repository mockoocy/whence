from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


class WindowReservoir:
    """Vitter's algorithm R over fixed-shape windows. Storage is allocated by the
    first window added, on its device. Draws from torch's global generator, which
    `run_arm` seeds per arm.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._buffer: torch.Tensor | None = None
        self._size: int = 0
        self._seen: int = 0
        self._capacity = capacity

    def __len__(self) -> int:
        return self._size

    def add(self, windows: Iterable[torch.Tensor]) -> None:
        """`windows`: same-shape windows, e.g. an (N, ...) tensor or a generator,
        so a large set need not be stacked first.
        """
        for window in windows:
            if self._buffer is None:
                self._buffer = window.new_empty((self._capacity, *window.shape))
            if self._size < self._capacity:
                self._buffer[self._size] = window
                self._size += 1
            else:
                index_candidate = torch.randint(0, self._seen + 1, size=())
                if index_candidate < self._capacity:
                    self._buffer[index_candidate] = window
            self._seen += 1

    def sample(self, k: int = 1) -> torch.Tensor:
        if self._buffer is None:
            msg = "nothing has been added to the reservoir"
            raise ValueError(msg)
        indices = torch.randperm(self._size)[:k]
        return self._buffer[indices]

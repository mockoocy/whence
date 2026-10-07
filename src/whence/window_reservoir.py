import torch


class WindowReservoir:
    """Vitter's algorithm R over fixed-shape windows. Storage is allocated by the
    first `add`, on that batch's device. Draws from torch's global generator, which
    `run` seeds per arm.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._buffer: torch.Tensor | None = None
        self._size: int = 0
        self._seen: int = 0
        self._capacity = capacity

    def __len__(self) -> int:
        return self._size

    def add(self, windows: torch.Tensor) -> None:
        if self._buffer is None:
            self._buffer = windows.new_empty((self._capacity, *windows.shape[1:]))
        for window in windows:
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

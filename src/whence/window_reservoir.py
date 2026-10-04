import torch


class WindowReservoir:
    """Vitter's algorithm R over fixed-shape windows."""

    def __init__(
        self,
        *,
        window_shape: tuple[int, ...],
        capacity: int = 512,
        device: torch.Device = "cpu",
        seed: int | None = None,
    ) -> None:
        self._buffer = torch.empty((capacity, *window_shape), device=device)
        self._size: int = 0
        self._seen: int = 0
        self._capacity = capacity
        rng = torch.Generator()
        rng_seed = seed if seed is not None else rng.seed()
        self._rng = rng.manual_seed(rng_seed)

    def add(self, windows: torch.Tensor) -> None:
        for window in windows:
            if self._size < self._capacity:
                self._buffer[self._size] = window
                self._size += 1
            else:
                index_candidate = torch.randint(
                    0,
                    self._seen + 1,
                    size=(),
                    generator=self._rng,
                )
                if index_candidate < self._capacity:
                    self._buffer[index_candidate] = window
            self._seen += 1

    def sample(self, k: int = 1) -> torch.Tensor:
        indices = torch.randperm(self._size, generator=self._rng)[:k]
        return self._buffer[indices]

from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


class Stream[T]:
    """A lazy iterable that also carries its length when the source knows it."""

    def __init__(self, iterable: Iterable[T], length: int | None = None) -> None:
        self._iterable = iterable
        self._length = length

    def map[U](self, fn: Callable[[T], U]) -> Stream[U]:
        return Stream(map(fn, self._iterable), self._length)

    def take(self, n: int | None) -> Stream[T]:
        if n is None:
            return self
        length = None if self._length is None else min(self._length, n)
        return Stream(islice(self._iterable, n), length)

    def __len__(self) -> int:
        # TypeError is what len(), list() and tqdm expect for "no length"
        if self._length is None:
            msg = "stream length is unknown"
            raise TypeError(msg)
        return self._length

    def __iter__(self) -> Iterator[T]:
        return iter(self._iterable)

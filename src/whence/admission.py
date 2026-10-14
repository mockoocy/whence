import math
from collections import deque
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from whence.events import Window


class Admission(Protocol):
    def offer(self, window: Window, score: float) -> list[Window] | None:
        """The windows of a new concept, if `window` completes one, else None.
        `score` is the window's anomaly score (the largest channel error of the row
        it scores), from the model as it was when the window arrived.
        """
        ...


class Every:
    """Starts a new concept every `timesteps` timesteps, handing over every window
    since the last one. Their labels are assumed known by then, as in
    validation-gated-cl.
    """

    def __init__(self, timesteps: int) -> None:
        self._timesteps = timesteps
        self._windows: list[Window] = []  # since the last concept

    def offer(self, window: Window, score: float) -> list[Window] | None:
        self._windows.append(window)
        if window.step == 0 or window.step % self._timesteps:
            return None
        concept, self._windows = self._windows, []
        return concept


class Adwin:
    """Start a new concept whenever ADWIN2 detects a change in the scores.

    ADWIN2 from Bifet & Gavalda (2007), "Learning from Time-Changing Data with
    Adaptive Windowing", monitoring the model's error. Hands over every window since
    the last concept, labels assumed known as in `Every`. A change seen fewer than
    `min_timesteps` after the last concept is deferred until that many have passed,
    which also folds the run of shrinks that follows one change into one concept.

    Not corrected here: the paper assumes independent scores within [0, 1], but
    forecasts are unbounded and scores autocorrelated, so `delta` does not bound the
    false alarm rate. The window keeps the old model's scores after an update, so an
    update that shifts the error level can itself be detected.
    """

    def __init__(
        self,
        delta: float,
        *,
        max_buckets: int = 5,
        min_timesteps: int = 1,
    ) -> None:
        self._delta = delta
        self._max_buckets = max_buckets  # the paper used M = 5 throughout
        self._min_timesteps = min_timesteps
        self._pending = False  # a change seen since the last concept
        # level i: buckets of 2**i scores as (sum, sum of squares), oldest first;
        # each level holds older scores than the one below it
        self._levels: list[deque[tuple[float, float]]] = []
        self._n = 0  # scores in the window
        self._sum = 0.0
        self._sum_sq = 0.0
        self._windows: list[Window] = []  # since the last concept

    def offer(self, window: Window, score: float) -> list[Window] | None:
        self._windows.append(window)  # one per timestep
        self._pending |= self.add(score)
        if not self._pending or len(self._windows) < self._min_timesteps:
            return None
        concept, self._windows, self._pending = self._windows, [], False
        return concept

    def add(self, score: float) -> bool:
        """Add a score; return whether the window shrank, i.e. a change was detected."""
        self._insert(score)
        changed = False
        while self._cut():
            self._drop_oldest()
            changed = True
        return changed

    def _insert(self, score: float) -> None:
        self._n += 1
        self._sum += score
        self._sum_sq += score * score
        carry = (score, score * score)
        for level in self._levels:
            level.append(carry)
            if len(level) <= self._max_buckets:
                return
            # M + 1 buckets of this size: the two oldest merge into the next level
            (s1, q1), (s2, q2) = level.popleft(), level.popleft()
            carry = (s1 + s2, q1 + q2)
        self._levels.append(deque([carry]))

    def _drop_oldest(self) -> None:
        oldest = self._levels[-1]
        bucket_sum, bucket_sum_sq = oldest.popleft()
        self._n -= 1 << (len(self._levels) - 1)
        self._sum -= bucket_sum
        self._sum_sq -= bucket_sum_sq
        if not oldest:
            self._levels.pop()

    def _cut(self) -> bool:
        """Check whether a bucket border splits the window into old and new at a change.

        A change: |mean(old) - mean(new)| >= eps_cut, the paper's eq. 3.1.
        """
        n = self._n
        if n < 2:  # noqa: PLR2004 - a split needs two scores
            return False
        mean = self._sum / n
        variance = max(self._sum_sq / n - mean * mean, 0.0)  # of the whole window
        log_term = math.log(2 * math.log(n) / self._delta)
        old_n, old_sum = 0, 0.0
        for size_log, level in reversed(list(enumerate(self._levels))):
            for bucket_sum, _ in level:
                old_n += 1 << size_log
                old_sum += bucket_sum
                new_n = n - old_n
                if new_n == 0:
                    return False
                inverse_m = 1 / old_n + 1 / new_n  # m = 1 / (1/n0 + 1/n1)
                eps_cut = (
                    math.sqrt(2 * inverse_m * variance * log_term)
                    + 2 / 3 * inverse_m * log_term
                )
                if abs(old_sum / old_n - (self._sum - old_sum) / new_n) >= eps_cut:
                    return True
        return False

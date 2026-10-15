import math
from collections import deque
from typing import TYPE_CHECKING, Protocol

import numpy as np
import ot

if TYPE_CHECKING:
    from whence.events import Window


class Admission(Protocol):
    """Decides which windows form a concept to learn from."""

    def offer(self, window: Window, score: float) -> list[Window] | None:
        """Return the windows of a concept if `window` completes one, else None.

        `score` is the window's anomaly score (the largest channel error of the row
        it scores), from the model as it was when the window arrived.
        """
        ...


class Every:
    """Start a new concept every `timesteps` timesteps.

    Hands over every window since the last concept, as in validation-gated-cl.
    """

    def __init__(self, timesteps: int) -> None:
        self._timesteps = timesteps
        self._windows: list[Window] = []  # since the last concept

    def offer(self, window: Window, score: float) -> list[Window] | None:
        """Return the windows since the last concept every `timesteps`, else None."""
        self._windows.append(window)
        if window.step == 0 or window.step % self._timesteps:
            return None
        concept, self._windows = self._windows, []
        return concept


class _AfterChange:
    """Start a new concept once `after_change` rows have followed a detected change.

    Hands over every window since the last concept, as `Every` does, so the new
    data is learned along with what came before it. Changes detected while
    waiting do not restart the wait. With `max_interval`, a concept also starts
    that many timesteps after the last one, as VLAD's forced update does.
    """

    def __init__(self, after_change: int, max_interval: int | None = None) -> None:
        self._after_change = after_change
        self._max_interval = max_interval
        self._due: int | None = None  # row that completes the concept, once detected
        self._windows: list[Window] = []  # since the last concept

    def offer(self, window: Window, score: float) -> list[Window] | None:
        """Return the windows since the last concept if `window` completes one."""
        self._windows.append(window)
        # the detector sees every window, also while waiting
        if self._detect(window, score) and self._due is None:
            self._due = window.last_row + self._after_change
        overdue = (
            self._max_interval is not None
            and window.step - self._windows[0].step >= self._max_interval
        )
        if not overdue and (self._due is None or window.last_row < self._due):
            return None
        concept, self._windows, self._due = self._windows, [], None
        return concept

    def _detect(self, window: Window, score: float) -> bool:
        raise NotImplementedError


class Adwin(_AfterChange):
    """Start a new concept after ADWIN2 detects a change in the scores.

    ADWIN2 is from Bifet & Gavalda (2007), "Learning from Time-Changing Data with
    Adaptive Windowing". Not corrected here: the paper assumes independent scores
    within [0, 1], but forecasts are unbounded and scores autocorrelated, so `delta`
    does not bound the false alarm rate. The window keeps the old model's scores
    after an update, so an update that shifts the error level is itself a change.
    """

    def __init__(
        self,
        delta: float,
        *,
        after_change: int,
        max_interval: int | None = None,
        max_buckets: int = 5,
    ) -> None:
        super().__init__(after_change, max_interval)
        self._delta = delta
        self._max_buckets = max_buckets  # the paper used M = 5 throughout
        # level i: buckets of 2**i scores as (sum, sum of squares), oldest first;
        # each level holds older scores than the one below it
        self._levels: list[deque[tuple[float, float]]] = []
        self._n = 0  # scores in the window
        self._sum = 0.0
        self._sum_sq = 0.0

    def _detect(self, window: Window, score: float) -> bool:
        return self.add(score)

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


class Wasserstein(_AfterChange):
    """Start a new concept after WATCH detects a change in the input rows.

    WATCH is from Faber et al. (2021), "WATCH: Wasserstein Change Point Detection
    for High-Dimensional Time Series Data", the detector VLAD extends. It reads each
    window's newest row in mini-batches of `batch_size`. Once the current
    distribution holds `min_points` rows, a mini-batch farther from it than `ratio`
    times the farthest of its own mini-batches is a change and starts the next one.
    Up to `max_points` rows, other mini-batches join it and the threshold is
    recomputed; then both freeze, as in the paper's pseudo-code.

    The distance is the exact 1-Wasserstein distance with Euclidean ground cost,
    where WATCH used an approximation. Not corrected here: rows are autocorrelated,
    and the features with the largest spread dominate the cost.
    """

    def __init__(  # noqa: PLR0913 - keyword-only tuning
        self,
        ratio: float,
        *,
        after_change: int,
        max_interval: int | None = None,
        batch_size: int = 4,
        min_points: int = 1024,
        max_points: int = 1024,
    ) -> None:
        super().__init__(after_change, max_interval)
        self._ratio = ratio
        self._batch_size = batch_size
        self._min_points = min_points
        self._max_points = max_points
        self._batch: list[np.ndarray] = []  # rows of the mini-batch being filled
        self._batches: list[np.ndarray] = []  # the current distribution
        self._threshold = math.inf

    def _detect(self, window: Window, score: float) -> bool:
        return self.add(window.data[-1].numpy())

    def add(self, row: np.ndarray) -> bool:
        """Add a row; return whether it completed a mini-batch that is a change."""
        self._batch.append(row)
        if len(self._batch) < self._batch_size:
            return False
        batch, self._batch = np.stack(self._batch), []
        points = len(self._batches) * self._batch_size
        if points >= self._min_points:  # detecting
            distance = _wasserstein(batch, np.concatenate(self._batches))
            if distance > self._threshold:
                self._batches = [batch]
                return True
        if points < self._max_points:
            self._batches.append(batch)
            if points + self._batch_size >= self._min_points:
                distribution = np.concatenate(self._batches)
                self._threshold = self._ratio * max(
                    _wasserstein(b, distribution) for b in self._batches
                )
        return False


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """Return the exact 1-Wasserstein distance between two uniformly weighted sets."""
    return float(ot.emd2([], [], ot.dist(a, b, metric="euclidean")))

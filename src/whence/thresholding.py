"""Thresholders: per-channel errors to one detection flag per row."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import lfilter


class Thresholder(Protocol):
    def detect(self, errors: np.ndarray) -> np.ndarray:
        """(rows, channels) errors -> (rows,) bool, True where any channel is
        anomalous.
        """
        ...


@dataclass(frozen=True)
class Rolling:
    """validation-gated-cl's rolling threshold (detector.py: compute_errors_telemanom,
    then rolling_window_thresholding), defaults included. Per channel, a row is
    anomalous where the errors' EWMA exceeds mean_factor x mean + std_factor x sd of
    the lowest `quantile` of the `window` rows up to the block of `stride` rows it is
    in, that block included; each detection also marks the buffer / 2 rows on either
    side. The block and the widening reach into later rows, as theirs do.
    """

    window: int = 5000
    stride: int = 70
    quantile: float = 0.99
    mean_factor: float = 1.3
    std_factor: float = 8.5
    buffer: int = 200
    span: int = 39  # of the EWMA; theirs is int(70 * 30 * 0.019)
    warmup: int = 250  # their model's window_size: see _smooth and _thresholds

    def detect(self, errors: np.ndarray) -> np.ndarray:
        flags = np.zeros(len(errors), np.bool_)
        # a channel at a time: in float64, as theirs, all at once would take twice
        # the memory of the errors
        for channel in errors.T:
            smoothed = self._smooth(channel)
            flags |= smoothed > self._thresholds(smoothed)
        # widening commutes with "any channel", so it runs once
        return maximum_filter1d(flags, self.buffer + 1, mode="constant")

    def _smooth(self, errors: np.ndarray) -> np.ndarray:
        """pandas' ewm(span).mean() (adjust=True): the mean of the rows so far, row
        i rows back weighing decay ** i.
        """
        decay = 1 - 2 / (self.span + 1)
        weights = (1 - decay ** np.arange(1, len(errors) + 1)) / (1 - decay)
        smoothed = lfilter([1], [1, -decay], errors.astype(np.float64)) / weights
        if len(smoothed) > 2 * self.warmup:  # the first rows, before it settles
            smoothed[: self.warmup] = smoothed[: 2 * self.warmup].mean()
        return smoothed

    def _thresholds(self, smoothed: np.ndarray) -> np.ndarray:
        n = len(smoothed)
        thresholds = np.empty(n)
        for start in range(0, n, self.stride):
            block = smoothed[max(0, start - self.window + 1) : start + self.stride]
            k = int(self.quantile * len(block)) or len(block)
            lowest = np.partition(block, k - 1)[:k]  # theirs sorts: the same values
            thresholds[start : start + self.stride] = (
                self.mean_factor * lowest.mean() + self.std_factor * lowest.std()
            )
        if n > 2 * self.warmup:
            thresholds[: 2 * self.warmup] = thresholds[2 * self.warmup]
        return thresholds

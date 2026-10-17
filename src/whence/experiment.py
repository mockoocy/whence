import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from whence.aggregator import HorizonAggregator
from whence.cycle import EarlyStopping, split
from whence.esa_adb import EsaAdbDataset, row_at
from whence.events import Window
from whence.time_eval_metrics.range_metrics import RangePrAUC

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from whence.admission import Admission
    from whence.config import ArmConfig, Config
    from whence.esa_adb import Sample, Scaler
    from whence.model import Model
    from whence.stream import Stream

ANOMALY = 1.0
LEARNED = {True: "update kept", False: "update rejected", None: "nothing learned"}

logger = logging.getLogger(__name__)


@dataclass
class Result:
    errors: np.ndarray
    labels: np.ndarray  # (timesteps,) True where any target channel is anomalous
    refits: list[tuple[int, bool]]  # (step, kept)

    @property
    def adapted_at(self) -> list[int]:
        return [step for step, kept in self.refits if kept]

    @property
    def rejected_at(self) -> list[int]:
        return [step for step, kept in self.refits if not kept]

    def scores(self) -> np.ndarray:
        """Per row anomaly score: the largest error over channels."""
        # largest channels may dominate tho, by having errors with highest scale.
        return self.errors.max(axis=1)


def device_name(device: str) -> str:
    """The GPU model for a CUDA device, else `device` as given."""
    return torch.cuda.get_device_name(device) if device.startswith("cuda") else device


def build_model(config: Config) -> Model:
    return config.model.build(config.data.columns, config.device)


def initial_training(config: Config, scaler: Scaler, seed: int) -> Model:
    """Fits `scaler` in place on the rows before `until`, then trains on them."""
    cols = config.data.columns
    training = config.training

    torch.manual_seed(seed)  # weight init draws from the global generator
    model = build_model(config)
    until = row_at(config.data.path, training.until)
    raw = torch.cat(list(EsaAdbDataset(config.data.path, cols=cols, rows=slice(until))))
    scaler.fit(raw[:, : len(cols.features)].numpy())
    dataset = EsaAdbDataset(
        config.data.path,
        cols=cols,
        rows=slice(until),
        scaler=scaler,
    )
    samples = dataset.stream(config.model.sample_len, limit=config.limit)
    logger.info(
        "initial training on %d windows before %s",
        len(samples),
        training.until,
    )
    windows = [
        Window(i, sample.data, sample.labels)
        for i, sample in enumerate(tqdm(samples, desc="initial training", unit="step"))
    ]
    train, validation = split(windows, training.val_fraction)
    train_data = torch.stack([w.data for w in train])
    fit = EarlyStopping(
        max_epochs=training.max_epochs,
        patience=training.patience,
        delta=training.delta,
        desc="initial early stopping",
    )
    fit(model, lambda: train_data, torch.stack([w.data for w in validation]))
    return model


def run_arm(  # noqa: PLR0913 - keyword-only
    config: Config,
    arm: ArmConfig,
    initial: Path,
    scaler: Scaler,
    *,
    seed: int,
    desc: str,
    end: datetime | None = None,
) -> Result:
    """Streams every row from `until` past one arm, scoring those before `end` (all
    of them if None). Each window is scored by its strategy, then offered to the
    arm's admission rule with its score; once the labels of a concept it handed
    over have arrived (`label_delay`), the strategy learns from it before the next
    window is scored.
    """
    torch.manual_seed(seed)
    model = build_model(config)
    model.load(initial)
    strategy = arm.strategy.build(model)
    admission = arm.admission.build() if arm.admission else None
    refits: list[tuple[int, bool]] = []
    n_scored = config.model.n_scored
    aggregator = HorizonAggregator(
        horizon=n_scored,
        n_targets=len(config.data.columns.channels),
        method=config.model.aggregation,
    )

    waiting: deque[list[Window]] = deque()  # concepts whose labels have not arrived
    samples = _stream(config, scaler, end)
    logger.info(
        "arm %s: %d windows from %s; model %s; strategy %s; admission %s",
        desc,
        len(samples),
        config.training.until,
        config.model,
        arm.strategy,
        arm.admission,
    )
    # preallocated: a list of per-row arrays costs ~800 B/row, 12 GB on Mission 1
    errors = np.empty((len(samples), len(config.data.columns.channels)), np.float32)
    labels = np.empty(len(samples), np.bool_)
    scored = 0  # rows scored so far

    windows = enumerate(tqdm(samples, desc=desc, unit="step"))
    # Windows after a concept were scored by the model before it learned, so they
    # are scored again: learning reaches the very next window, whatever the batch
    # size.
    stale: list[tuple[int, Sample]] = []
    while batch := stale + list(
        islice(windows, config.score_batch_size - len(stale)),
    ):
        sample_errors = strategy.predict(_windows(batch, config.device))
        scores = aggregator.peek(sample_errors).max(dim=1).values  # as Result.scores
        n, concept = (
            _offer(admission, batch, scores.tolist(), waiting, config.label_delay)
            if admission
            else (len(batch), None)
        )
        rows = slice(scored, scored + n)
        errors[rows] = aggregator.push(sample_errors[:n]).numpy()
        first_scored = torch.stack([s.labels[-n_scored] for _, s in batch[:n]])
        labels[rows] = (first_scored == ANOMALY).any(dim=1).numpy()
        scored = rows.stop
        stale = batch[n:]
        if concept is not None:
            step = batch[n - 1][0]
            logger.info(
                "arm %s, step %d: learning from %d windows",
                desc,
                step,
                len(concept),
            )
            kept = strategy.learn(concept)
            logger.info("arm %s, step %d: %s", desc, step, LEARNED[kept])
            if kept is not None:
                refits.append((step, kept))

    assert scored == len(samples), f"stream yielded {scored} of {len(samples)} windows"
    return Result(errors, labels, refits)


def _windows(batch: list[tuple[int, Sample]], device: str) -> torch.Tensor:
    """Stack consecutive windows by moving the rows they span to `device`."""
    first = batch[0][1].data
    rows = torch.cat([first, *(s.data[-1:] for _, s in batch[1:])])
    return rows.to(device).unfold(0, len(first), 1).transpose(1, 2)


def _offer(
    admission: Admission,
    batch: list[tuple[int, Sample]],
    scores: list[float],
    waiting: deque[list[Window]],
    label_delay: int,
) -> tuple[int, list[Window] | None]:
    """Offer the windows in order, up to the first at which a concept is learnable.

    Concepts the admission completes join `waiting`; the oldest is learnable once
    the stream is `label_delay` rows past its last row. Returns how many windows
    were offered, and that concept.
    """
    for n, ((step, sample), score) in enumerate(zip(batch, scores, strict=True), 1):
        window = Window(step, sample.data, sample.labels)
        concept = admission.offer(window, score)
        if concept is not None:
            waiting.append(concept)
        if waiting and waiting[0][-1].last_row + label_delay <= window.last_row:
            return n, waiting.popleft()
    return len(batch), None


def _stream(config: Config, scaler: Scaler, end: datetime | None) -> Stream[Sample]:
    spec = config.model
    path = config.data.path
    until = row_at(path, config.training.until)
    # window j scores row `until` + j first and reaches n_scored - 1 rows past it,
    # so the last window read scores the row before `end`
    stop = None if end is None else row_at(path, end) + spec.n_scored - 1
    rows = slice(until - (spec.sample_len - spec.n_scored), stop)
    cols = config.data.columns
    dataset = EsaAdbDataset(path, cols=cols, rows=rows, scaler=scaler)
    return dataset.stream(spec.sample_len, limit=config.limit)


def report(
    result: Result,
    periods: dict[str, slice],
    *,
    max_buffer_size: int = 6,  # slopes of at most 3 rows, ESA-ADB's point anomalies
) -> dict[str, dict[str, float]]:
    """Metrics of each period (a slice of the scored rows) by its name; periods
    without rows are left out.
    """
    n = len(result.labels)
    return {
        name: _metrics(result, rows, max_buffer_size)
        for name, rows in periods.items()
        if len(range(n)[rows])
    }


def _metrics(result: Result, rows: slice, max_buffer_size: int) -> dict[str, float]:
    span = range(len(result.labels))[rows]
    y_true = result.labels[rows].astype(np.int_)
    # a refit at step k follows the window that scores row k
    kept = [kept for step, kept in result.refits if step in span]
    metrics = {
        "timesteps": len(span),
        "anomalous": int(y_true.sum()),
        "adaptations": sum(kept),
        "rejected": len(kept) - sum(kept),
        "TRIMMED_MAE": trimmed_mae(result.errors[rows]),
    }
    if not 0 < y_true.sum() < len(y_true):
        logger.warning("rows %s: no label variety, skipping VUS", rows)
        return metrics
    logger.info(
        "evaluating %d rows: Range PR/ROC VUS, buffer sizes 0..%d",
        len(y_true),
        max_buffer_size,
    )
    pr, roc = range_vus(y_true, result.scores()[rows], max_buffer_size)
    return metrics | {"RANGE_PR_VOLUME": pr, "RANGE_ROC_VOLUME": roc}


def trimmed_mae(errors: np.ndarray, cutoff: float = 95) -> float:
    """Compute the MAE of the errors at or below the `cutoff`th percentile."""
    return float(np.mean([c[c <= np.percentile(c, cutoff)].mean() for c in errors.T]))


def range_vus(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_buffer_size: int,
) -> tuple[float, float]:
    """TimeEval's RangePrVUS and RangeRocVUS scores. Both average, over buffer sizes,
    one AUC of the same TimeEval computation, which yields both AUCs at once; here it
    runs once per buffer size instead of once per metric, the sizes in threads.
    """

    def aucs(buffer_size: int) -> tuple[float, float]:
        metric = RangePrAUC(buffer_size=buffer_size)
        pr, roc, _ = metric._range_pr_roc_auc_support(y_true, y_score)  # noqa: SLF001
        return pr, roc

    sizes = range(max_buffer_size + 1)
    # numpy releases the GIL in the array passes that take the time; on a laptop
    # memory bandwidth, not the GIL, capped processes at the same speed-up
    with ThreadPoolExecutor() as pool:
        prs, rocs = map(np.array, zip(*pool.map(aucs, sizes), strict=True))
    # averaged as RangePrVUS/RangeRocVUS.score do, so the results are identical
    return float(np.sum(prs) / len(sizes)), float(np.sum(rocs) / len(sizes))

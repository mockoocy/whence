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


def initial_training(config: Config, scaler: Scaler) -> Model:
    """Fits `scaler` in place on the rows before `until`, then trains on them."""
    cols = config.data.columns
    training = config.training

    if config.seed is not None:
        torch.manual_seed(config.seed)  # weight init draws from the global generator
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


def run_arm(
    config: Config,
    arm: ArmConfig,
    initial: Path,
    scaler: Scaler,
    *,
    desc: str,
) -> Result:
    """Streams every row from `until` past one arm. Each window is scored by its
    strategy, then offered to the arm's admission rule with its score; once the
    labels of a concept it handed over have arrived (`label_delay`), the strategy
    learns from it before the next window is scored.
    """
    if config.seed is not None:
        torch.manual_seed(config.seed)
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
    samples = _stream(config, scaler)
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
        sample_errors = strategy.predict(torch.stack([s.data for _, s in batch]))
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


def _stream(config: Config, scaler: Scaler) -> Stream[Sample]:
    spec = config.model
    until = row_at(config.data.path, config.training.until)
    rows = slice(until - (spec.sample_len - spec.n_scored), None)
    cols = config.data.columns
    dataset = EsaAdbDataset(config.data.path, cols=cols, rows=rows, scaler=scaler)
    return dataset.stream(spec.sample_len, limit=config.limit)


# max_buffer_size 6: slopes of at most 3 rows, ESA-ADB's point anomaly length
def report(result: Result, *, max_buffer_size: int = 6) -> dict[str, float]:
    y_true = result.labels.astype(np.int_)
    y_score = result.scores()
    print(
        f"timesteps={len(y_true)} anomalous={y_true.sum()} "
        f"adaptations={len(result.adapted_at)} rejected={len(result.rejected_at)}",
    )
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("no usable label variety in this slice, skipping VUS")
        return {}

    logger.info(
        "evaluating %d rows: Range PR/ROC VUS, buffer sizes 0..%d",
        len(y_true),
        max_buffer_size,
    )
    pr, roc = range_vus(y_true, y_score, max_buffer_size)
    metrics = {"RANGE_PR_VOLUME": pr, "RANGE_ROC_VOLUME": roc}
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    return metrics


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

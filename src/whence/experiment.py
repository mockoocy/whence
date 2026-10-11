from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from whence.aggregator import HorizonAggregator
from whence.cycle import EarlyStopping, split
from whence.esa_adb import EsaAdbDataset, row_at
from whence.events import Window
from whence.time_eval_metrics.range_metrics import RangePrVUS, RangeRocVUS

if TYPE_CHECKING:
    from pathlib import Path

    from whence.config import ArmConfig, Config
    from whence.esa_adb import Sample, Scaler
    from whence.model import Model
    from whence.stream import Stream

ANOMALY = 1.0


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
    """Streams every row from `until` past one arm. Each window is offered to the
    arm's admission rule and scored by its strategy; when the admission hands over
    a new concept, the strategy learns from it once all of it has been scored.
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

    samples = _stream(config, scaler)
    # preallocated: a list of per-row arrays costs ~800 B/row, 12 GB on Mission 1
    errors = np.empty((len(samples), len(config.data.columns.channels)), np.float32)
    labels = np.empty(len(samples), np.bool_)
    scored = 0  # rows scored so far

    def score(batch: list[Sample]) -> None:
        nonlocal scored
        rows = slice(scored, scored + len(batch))
        errors[rows] = aggregator.push(
            strategy.predict(torch.stack([s.data for s in batch])),
        ).numpy()
        first_scored = torch.stack([s.labels[-n_scored] for s in batch])
        labels[rows] = (first_scored == ANOMALY).any(dim=1).numpy()
        scored = rows.stop

    batch: list[Sample] = []
    for step, sample in enumerate(tqdm(samples, desc=desc, unit="step")):
        batch.append(sample)
        # Offered before it is scored, which is safe because the admission only sees
        # the window, never its errors (a detector that needs them would have to be
        # offered after scoring, and the rest of the batch rescored after learning).
        # The batch closes at a concept, so learning reaches the very next window,
        # whatever the batch size.
        concept = (
            admission.offer(Window(step, sample.data, sample.labels))
            if admission
            else None
        )
        if concept is None and len(batch) < config.score_batch_size:
            continue
        score(batch)
        batch = []
        if concept is not None:
            kept = strategy.learn(concept)
            if kept is not None:
                refits.append((step, kept))
    if batch:
        score(batch)

    assert scored == len(samples), f"stream yielded {scored} of {len(samples)} windows"
    return Result(errors, labels, refits)


def _stream(config: Config, scaler: Scaler) -> Stream[Sample]:
    spec = config.model
    until = row_at(config.data.path, config.training.until)
    rows = slice(until - (spec.sample_len - spec.n_scored), None)
    cols = config.data.columns
    dataset = EsaAdbDataset(config.data.path, cols=cols, rows=rows, scaler=scaler)
    return dataset.stream(spec.sample_len, limit=config.limit)


def report(result: Result, *, max_buffer_size: int = 64) -> dict[str, float]:
    y_true = result.labels.astype(np.int_)
    y_score = result.scores()
    print(
        f"timesteps={len(y_true)} anomalous={y_true.sum()} "
        f"adaptations={len(result.adapted_at)} rejected={len(result.rejected_at)}",
    )
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("no usable label variety in this slice, skipping VUS")
        return {}

    metrics = {}
    for metric in (
        RangePrVUS(max_buffer_size=max_buffer_size),
        RangeRocVUS(max_buffer_size=max_buffer_size),
    ):
        metrics[metric.name] = metric.score(y_true, y_score)
        print(f"{metric.name}: {metrics[metric.name]:.4f}")
    return metrics

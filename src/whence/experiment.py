from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from whence.aggregator import HorizonAggregator
from whence.cycle import EarlyStopping, Holdout
from whence.esa_adb import EsaAdbDataset, row_at
from whence.events import Label, Window
from whence.forecaster import LstmForecaster
from whence.label_feed import LabelFeed
from whence.time_eval_metrics.range_metrics import RangePrVUS, RangeRocVUS

if TYPE_CHECKING:
    from whence.config import Config
    from whence.model import Model
    from whence.strategies import Strategy

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
        """Per row anomaly score: the largest error over channels. Errors are in
        the model's input-scaled units, so channels are comparable.
        """
        return self.errors.max(axis=1)


def device_name(device: str) -> str:
    """The GPU model for a CUDA device, else `device` as given."""
    return torch.cuda.get_device_name(device) if device.startswith("cuda") else device


def initial_training(config: Config) -> Model:
    """The model every arm starts from: trained on all rows before `until`,
    early-stopped on their newest `val_fraction`. No strategy sees either part.
    """
    cols = config.data.columns
    spec = config.model
    training = config.training

    if config.seed is not None:
        torch.manual_seed(config.seed)  # weight init draws from the global generator
    model = LstmForecaster(
        n_features=len(cols.features),
        n_targets=len(cols.channels),
        window_size=spec.window_size,
        horizon=spec.horizon,
        hidden_size=spec.hidden_size,
        dropout=spec.dropout,
        lr=spec.lr,
        device=config.device,
    )
    until = row_at(config.data.path, training.until)
    dataset = EsaAdbDataset(config.data.path, cols=cols, rows=slice(until))
    samples = dataset.stream(model.sample_len, limit=config.limit)
    windows = [
        Window(i, sample.data)
        for i, sample in enumerate(tqdm(samples, desc="initial training", unit="win"))
    ]
    if not windows:
        msg = f"the initial period before {training.until} has no window"
        raise ValueError(msg)
    holdout = Holdout(training.val_fraction)
    train = torch.stack(
        [w.data for w in holdout.split(windows, now=windows[-1].last_row)],
    )
    fit = EarlyStopping(
        max_epochs=training.max_epochs,
        patience=training.patience,
        delta=training.delta,
        desc="initial early stopping",
    )
    fit(model, lambda: train, holdout.windows())
    return model


def run(
    model: Model,
    strategy: Strategy,
    config: Config,
    *,
    label_delay: int | None = 0,
    desc: str = "stream",
) -> Result:
    if config.seed is not None:
        # training randomness then does not depend on which arms ran before this one
        torch.manual_seed(config.seed)
    n_scored = model.n_scored
    aggregator = HorizonAggregator(
        horizon=n_scored,
        n_targets=len(config.data.columns.channels),
        method=config.model.aggregation,
    )
    feed = LabelFeed(delay=label_delay)
    strategy.on_start(model)

    step = 0

    # The first scored rows are the first rows from `until`; the windows before
    # them reach back into the initial period for the unscored rest as context.
    until = row_at(config.data.path, config.training.until)
    rows = slice(until - (model.sample_len - n_scored), None)
    dataset = EsaAdbDataset(config.data.path, cols=config.data.columns, rows=rows)
    samples = dataset.stream(model.sample_len, limit=config.limit)
    # preallocated: a list of per-row arrays costs ~800 B/row, 12 GB on Mission 1
    errors = np.empty((len(samples), len(config.data.columns.channels)), np.float32)
    labels = np.empty(len(samples), np.bool_)
    stream = tqdm(samples, desc=desc, unit="win")
    for batch in batched(stream, config.score_batch_size, strict=False):
        batch_errors = model.errors(torch.stack([s.data for s in batch]))  # test...
        for sample, error in zip(batch, batch_errors, strict=False):
            completed = aggregator.push(error)  # (n_targets,)
            scored = sample.labels[-n_scored:]
            errors[step] = completed.numpy()
            # ground truth for the metrics, never delayed; scored[0] is the row
            # `completed` covers
            labels[step] = (scored[0] == ANOMALY).any()

            window = Window(step, sample.data, completed)
            # A row's label enters the feed when the row first streams past, i.e. as
            # the window's newest row - every scored row, for the first window.
            # Entering later, as the oldest scored row, would add n_scored - 1 rows
            # to every delay.
            arrived = scored if step == 0 else sample.labels[-1:]
            first_arrived = window.last_row - len(arrived) + 1
            for offset, c in enumerate(arrived):
                feed.observe(Label(first_arrived + offset, c))

            strategy.on_window(window)  # ...then adapt
            for label in feed.release(now=window.last_row):
                strategy.on_label(label)

            step += 1

    assert step == len(samples), f"stream yielded {step} of {len(samples)} windows"
    return Result(
        errors,
        labels,
        list(strategy.refits),
    )


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

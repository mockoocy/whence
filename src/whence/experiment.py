from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from whence.aggregator import HorizonAggregator, RobustScaler
from whence.cycle import EarlyStopping, Holdout
from whence.esa_adb import EsaAdbDataset, row_at
from whence.events import Label, Window
from whence.forecaster import LstmForecaster
from whence.label_feed import LabelFeed
from whence.strategies import Initial, Strategy
from whence.time_eval_metrics.range_metrics import RangePrVUS, RangeRocVUS
from whence.window_reservoir import WindowReservoir

if TYPE_CHECKING:
    from whence.config import Config
    from whence.model import Model

ANOMALY = 1.0


@dataclass
class Result:
    errors: np.ndarray
    labels: np.ndarray  # (timesteps,) True where any target channel is anomalous
    refits: list[tuple[int, bool]]  # (step, kept)
    # scaler instance with it's statistics being updated during different fits.
    scaler: RobustScaler

    @property
    def adapted_at(self) -> list[int]:
        return [step for step, kept in self.refits if kept]

    @property
    def rejected_at(self) -> list[int]:
        return [step for step, kept in self.refits if not kept]

    def scores(self) -> np.ndarray:
        """Per row anomaly labels."""
        return self.scaler(torch.from_numpy(self.errors)).amax(dim=1).numpy()


def initial_training(config: Config) -> Initial:
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
    buffer = WindowReservoir(
        window_shape=(model.sample_len, len(cols.features)),
        capacity=training.capacity,
        seed=config.seed,
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
    # the first cycle: its held-out tail is V(0), which no arm ever trains on
    holdout = Holdout(training.val_fraction)
    train = holdout.split(windows, now=windows[-1].last_row)
    buffer.add(torch.stack([w.data for w in train]))
    validation = holdout.windows()

    n_targets = len(cols.channels)
    clean = buffer.sample(training.capacity)
    model.input_scaler = RobustScaler.fit(clean.flatten(0, 1))
    targets = model.input_scaler(clean)[:, spec.window_size :, :n_targets]
    model.target_variance = targets.var(dim=(0, 1))
    fit = EarlyStopping(
        max_epochs=training.max_epochs,
        patience=training.patience,
        delta=training.delta,
        gated=False,
        desc="initial early stopping",
    )
    fit(model, lambda: buffer.sample(training.capacity), validation)
    scaler = RobustScaler.fit(_aggregated_errors(model, validation, config))
    return Initial(model, buffer, validation, scaler)


def _aggregated_errors(
    model: Model,
    windows: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    aggregator = HorizonAggregator(
        horizon=model.n_scored,
        n_targets=len(config.data.columns.channels),
        method=config.model.aggregation,
    )
    rows = []
    for batch in windows.split(config.score_batch_size):
        rows += [aggregator.push(error) for error in model.errors(batch)]
    return torch.stack(rows)


def run(
    initial: Initial,
    strategy: Strategy,
    config: Config,
    *,
    label_delay: int | None = 0,
    desc: str = "stream",
) -> Result:
    if config.seed is not None:
        # training randomness then does not depend on which arms ran before this one
        torch.manual_seed(config.seed)
    model = initial.model
    n_scored = model.n_scored
    aggregator = HorizonAggregator(
        horizon=n_scored,
        n_targets=len(config.data.columns.channels),
        method=config.model.aggregation,
    )
    feed = LabelFeed(delay=label_delay)
    strategy.on_start(initial)

    errors: list[np.ndarray] = []
    labels: list[bool] = []
    step = 0

    # The first scored rows are the first rows from `until`; the windows before
    # them reach back into the initial period for the unscored rest as context.
    until = row_at(config.data.path, config.training.until)
    rows = slice(until - (model.sample_len - n_scored), None)
    dataset = EsaAdbDataset(config.data.path, cols=config.data.columns, rows=rows)
    samples = dataset.stream(model.sample_len, limit=config.limit)
    stream = tqdm(samples, desc=desc, unit="win")
    for batch in batched(stream, config.score_batch_size, strict=False):
        batch_errors = model.errors(torch.stack([s.data for s in batch]))  # test...
        for sample, error in zip(batch, batch_errors, strict=False):
            completed = aggregator.push(error)  # (n_targets,)
            scored = sample.labels[-n_scored:]
            errors.append(completed.numpy())
            # ground truth for the metrics, never delayed; scored[0] is the row
            # `completed` covers
            labels.append(bool((scored[0] == ANOMALY).any()))

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

    return Result(
        np.stack(errors),
        np.array(labels),
        list(strategy.refits),
        initial.scaler,
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

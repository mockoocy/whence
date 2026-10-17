"""The experiment's config.toml: what one `whence run` executes.

Every `kind` field selects a variant of a tagged union. Config classes build the
object they describe, so the classes themselves stay free of config.
"""

import copy
import itertools
import tomllib
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NaiveDatetime,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    ValidationInfo,
    model_validator,
)
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from whence.admission import Admission, Adwin, Every, Oracle, Wasserstein
from whence.aggregator import Method
from whence.cycle import EarlyStopping
from whence.esa_adb import ColumnSet, Scaler
from whence.forecaster import LstmForecaster
from whence.strategies import Frozen, Replay, Strategy, ValidationGate
from whence.zscore import ZScore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from whence.model import Model


class _Strict(BaseModel):
    # a misspelt key in the TOML fails loudly instead of silently taking the default
    model_config = ConfigDict(extra="forbid", frozen=True)


def _under_data_root(path: Path, info: ValidationInfo) -> Path:
    """Relative paths are relative to --data-root; absolute ones stay as they are."""
    root = (info.context or {}).get("data_root", Path.cwd())
    resolved = (root / path).resolve()
    if not resolved.exists():
        msg = f"{resolved} does not exist"
        raise ValueError(msg)
    return resolved


DataPath = Annotated[Path, AfterValidator(_under_data_root)]
Fraction = Annotated[float, Field(gt=0, lt=1)]


class PaperChannels(_Strict):
    """channel_41..46, as in validation-gated-cl."""

    kind: Literal["paper"]

    def columns(self) -> ColumnSet:
        return ColumnSet.validation_gated_cl()


class TargetChannels(_Strict):
    """Every annotated channel (Target=YES in channels.csv)."""

    kind: Literal["targets"]
    channels_csv: DataPath

    def columns(self) -> ColumnSet:
        return ColumnSet.targets(self.channels_csv)


class IdentityScaling(_Strict):
    """Features as stored in the parquet."""

    kind: Literal["identity"]

    def build(self) -> Scaler:
        return FunctionTransformer()


class RobustScaling(_Strict):
    """Per feature: minus the median, over the interquartile range."""

    kind: Literal["robust"]

    def build(self) -> Scaler:
        return RobustScaler()


class DataConfig(_Strict):
    """For now there is one dataset: one file, split in time by `training.until`."""

    path: DataPath
    channels: PaperChannels | TargetChannels = Field(
        PaperChannels(kind="paper"),
        discriminator="kind",
    )
    # fitted on the rows before `training.until`
    scaling: IdentityScaling | RobustScaling = Field(
        IdentityScaling(kind="identity"),
        discriminator="kind",
    )

    @cached_property
    def columns(self) -> ColumnSet:
        return self.channels.columns()


class LstmForecasterConfig(_Strict):
    kind: Literal["lstm_forecaster"]
    window_size: PositiveInt = 32
    horizon: PositiveInt = 4
    hidden_size: tuple[PositiveInt, PositiveInt] = (16, 16)
    dropout: float = Field(0.5, ge=0, lt=1)
    lr: float = Field(1e-3, gt=0)
    batch_size: PositiveInt = 64  # windows per training step
    # windows per forward pass when computing the validation loss
    eval_batch_size: PositiveInt = 1024
    aggregation: Method = "first"

    @property
    def sample_len(self) -> int:
        """Rows the stream hands the model per window."""
        return self.window_size + self.horizon

    @property
    def n_scored(self) -> int:
        """Rows at the end of a window the model scores."""
        return self.horizon

    def build(self, cols: ColumnSet, device: str) -> Model:
        return LstmForecaster(
            n_features=len(cols.features),
            n_targets=len(cols.channels),
            window_size=self.window_size,
            horizon=self.horizon,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            lr=self.lr,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            device=device,
        )


class ZScoreConfig(_Strict):
    """Mean and variance of the initial training rows; with `alpha`, they then
    follow the stream as EWMAs (alpha = 1 - 0.5 ** (1 / half_life_in_timesteps)).
    """

    kind: Literal["zscore"]
    alpha: Fraction | None = None
    sample_len: ClassVar[int] = 1  # one row per window, scored as itself
    n_scored: ClassVar[int] = 1
    aggregation: ClassVar[Method] = "first"

    def build(self, cols: ColumnSet, device: str) -> Model:  # runs on the CPU
        return ZScore(n_targets=len(cols.channels), alpha=self.alpha)


ModelConfig = Annotated[
    LstmForecasterConfig | ZScoreConfig,
    Field(discriminator="kind"),
]


class TrainingConfig(_Strict):
    # The initial period is every row before `until`, the stream every row from it.
    # The initial model should train on a contiguous, label-free interval, so
    # `until` is meant to be the first labelled row; nothing checks it.
    until: NaiveDatetime
    val_fraction: Fraction = 0.2
    max_epochs: PositiveInt = 1000
    patience: PositiveInt = 20
    delta: NonNegativeFloat = 1e-5  # in MSE of the scaled values, see Model.loss


class EveryConfig(_Strict):
    kind: Literal["every"]
    timesteps: PositiveInt  # per concept

    def build(self) -> Admission:
        return Every(self.timesteps)


class OracleConfig(_Strict):
    """A new concept `after_change` rows after the label codes change."""

    kind: Literal["oracle"]
    after_change: PositiveInt  # rows of new data in the concept
    # timesteps after the last concept at which one starts anyway; None: never
    max_interval: PositiveInt | None = None

    def build(self) -> Admission:
        return Oracle(after_change=self.after_change, max_interval=self.max_interval)


class AdwinConfig(_Strict):
    """A new concept `after_change` rows after ADWIN2 sees the score change."""

    kind: Literal["adwin"]
    delta: Fraction  # confidence; the paper's experiments used 0.05 to 0.3
    after_change: PositiveInt  # rows of new data in the concept
    # timesteps after the last concept at which one starts anyway; None: never
    max_interval: PositiveInt | None = None
    max_buckets: PositiveInt = 5  # buckets of each size, the paper's M

    def build(self) -> Admission:
        return Adwin(
            self.delta,
            after_change=self.after_change,
            max_interval=self.max_interval,
            max_buckets=self.max_buckets,
        )


class WassersteinConfig(_Strict):
    """A new concept `after_change` rows after WATCH sees the input rows change."""

    kind: Literal["wasserstein"]
    # threshold over the largest in-distribution distance, the paper's epsilon;
    # VLAD tried 1.5 to 3
    ratio: float = Field(gt=0)
    after_change: PositiveInt  # rows of new data in the concept
    # timesteps after the last concept at which one starts anyway; None: never
    max_interval: PositiveInt | None = None
    batch_size: PositiveInt = 4  # rows per mini-batch, omega; VLAD used 4
    min_points: PositiveInt = 1024  # rows before detection starts, kappa
    # rows after which the distribution and threshold freeze, mu; None: min_points
    max_points: PositiveInt | None = None

    @model_validator(mode="after")
    def _max_not_below_min(self) -> Self:
        if self.max_points is not None and self.max_points < self.min_points:
            msg = "max_points must not be below min_points"
            raise ValueError(msg)
        return self

    def build(self) -> Admission:
        return Wasserstein(
            self.ratio,
            after_change=self.after_change,
            max_interval=self.max_interval,
            batch_size=self.batch_size,
            min_points=self.min_points,
            max_points=self.max_points or self.min_points,
        )


class EarlyStoppingConfig(_Strict):
    max_epochs: PositiveInt = 100
    patience: PositiveInt = 5
    delta: NonNegativeFloat = 1e-5

    def build(self) -> EarlyStopping:
        return EarlyStopping(
            max_epochs=self.max_epochs,
            patience=self.patience,
            delta=self.delta,
        )


class FrozenConfig(_Strict):
    kind: Literal["frozen"]

    def build(self, model: Model) -> Strategy:
        return Frozen(model)


class ReplayConfig(_Strict):
    kind: Literal["replay"]
    capacity: PositiveInt = 70_000  # windows the replay reservoir holds
    sample_size: PositiveInt = 4096
    epochs: PositiveInt = 1

    def build(self, model: Model) -> Strategy:
        return Replay(
            model,
            capacity=self.capacity,
            sample_size=self.sample_size,
            epochs=self.epochs,
        )


class ValidationGateConfig(_Strict):
    kind: Literal["validation_gate"]
    # newest share of each concept's rows that joins the validation set
    val_fraction: Fraction = 0.2
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    capacity: PositiveInt = 70_000
    sample_size: PositiveInt = 4096
    gate_delta: NonNegativeFloat = 1e-5  # in validation loss, see LstmForecaster.loss

    def build(self, model: Model) -> Strategy:
        return ValidationGate(
            model,
            val_fraction=self.val_fraction,
            early_stopping=self.early_stopping.build(),
            capacity=self.capacity,
            sample_size=self.sample_size,
            gate_delta=self.gate_delta,
        )


class ArmConfig(_Strict):
    """A strategy and its admission rule, which decides when a new concept starts."""

    strategy: FrozenConfig | ReplayConfig | ValidationGateConfig = Field(
        discriminator="kind",
    )
    # None: no concept ever starts
    admission: EveryConfig | OracleConfig | AdwinConfig | WassersteinConfig | None = (
        Field(None, discriminator="kind")
    )


# the name becomes the arm's directory in the run; dots for swept values, e.g. 1.5
ArmName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")]


class EvaluationConfig(_Strict):
    """Every arm but the baselines streams the development rows, before `dev_until`;
    the one with the best dev Range-PR-VUS then streams again from the start to the
    end of the file, as the baselines do, and is reported on the test rows.
    """

    dev_until: NaiveDatetime
    # arms that stream to the end whichever wins, to compare it with; not candidates
    baselines: tuple[ArmName, ...] = ()


class Config(_Strict):
    # timesteps streamed per part (initial training, stream), for development runs
    limit: PositiveInt | None = None
    device: str = "cpu"
    score_batch_size: PositiveInt = 512
    # timesteps until a row's labels are known (default: 1 week at 30 s); a concept
    # is learned only once all of its rows' labels are
    label_delay: NonNegativeInt = 20_160
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    # a list anywhere in an arm sweeps it, see _expand
    arms: dict[ArmName, ArmConfig] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _expand_sweeps(cls, data: Any) -> Any:  # noqa: ANN401 - the TOML as parsed
        if not isinstance(data, dict) or not isinstance(data.get("arms"), dict):
            return data  # the field validation reports what is wrong
        expanded = {name: _expand(name, arm) for name, arm in data["arms"].items()}
        arms: dict[str, Any] = {}
        for arm in expanded.values():
            if clash := arms.keys() & arm.keys():
                msg = f"swept arms collide: {', '.join(sorted(clash))}"
                raise ValueError(msg)
            arms |= arm
        data = data | {"arms": arms}
        evaluation = data.get("evaluation")
        baselines = (
            evaluation.get("baselines") if isinstance(evaluation, dict) else None
        )
        if isinstance(baselines, list):
            # a swept baseline is every arm it expands to
            names = [name for b in baselines for name in expanded.get(b, [b])]
            data["evaluation"] = evaluation | {"baselines": names}
        return data

    @model_validator(mode="after")
    def _evaluation_fits(self) -> Self:
        if self.evaluation.dev_until <= self.training.until:
            msg = "evaluation.dev_until must be after training.until"
            raise ValueError(msg)
        if unknown := set(self.evaluation.baselines) - set(self.arms):
            msg = f"evaluation.baselines names no arm: {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        return self


def _expand(name: str, arm: Any) -> dict[str, Any]:  # noqa: ANN401 - the TOML
    """The arms an arm with lists in it stands for, one per combination of their
    values, named `name` and -<key><value> for each list, in the order they appear:
    delta = [1e-5, 1e-30] in arm `adwin` makes `adwin-delta1e-05` and
    `adwin-delta1e-30`. No arm field takes a list, so a list always means a sweep.
    """
    swept = list(_lists(arm)) if isinstance(arm, dict) else []
    if not swept:
        return {name: arm}
    if empty := [path[-1] for path, values in swept if not values]:
        msg = f"arm {name} sweeps no values of {', '.join(empty)}"
        raise ValueError(msg)
    arms = {}
    for values in itertools.product(*(values for _, values in swept)):
        one = copy.deepcopy(arm)
        suffix = ""
        for (path, _), value in zip(swept, values, strict=True):
            *parents, key = path
            table = one
            for parent in parents:
                table = table[parent]
            table[key] = value
            suffix += f"-{key}{value}"
        arms[name + suffix] = one
    return arms


def _lists(table: dict, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple, list]]:
    """(path, values) of each list in `table` and the tables in it."""
    for key, value in table.items():
        if isinstance(value, list):
            yield (*path, key), value
        elif isinstance(value, dict):
            yield from _lists(value, (*path, key))


def load_config(path: Path, data_root: Path) -> Config:
    with path.open("rb") as f:
        return Config.model_validate(
            tomllib.load(f),
            context={"data_root": data_root.resolve()},
        )


TEMPLATE = """\
# limit = 3000                # timesteps streamed per part, for development runs
device = "cuda"
label_delay = 20_160           # timesteps until labels are known (1 week at 30 s)

[data]                         # relative paths are relative to --data-root
path = "ESA_Mission1.parquet"
# or { kind = "targets", channels_csv = "ESA-Mission1/channels.csv" }
channels = { kind = "paper" }
scaling = { kind = "identity" } # or "robust"; fitted on the rows before `until`

[model]
kind = "lstm_forecaster"
window_size = 32
horizon = 4
hidden_size = [16, 16]
aggregation = "first"          # "first", "mean" or "ewma"

[training]                     # the initial fit every arm starts from
# first label of any kind in any channel; the stream starts here
until = 2000-02-10T01:31:30
val_fraction = 0.2
max_epochs = 1000
patience = 20
delta = 1e-5                   # in MSE of the scaled values

[evaluation]
# arms but the baselines stream to here; the best by Range-PR-VUS on these rows
# then streams again to the end, and is reported on the rows from here on.
# 2007-01-01 is where ESA-ADB's Mission 1 test set starts.
dev_until = 2007-01-01T00:00:00
baselines = ["frozen"]         # stream to the end whichever wins; not candidates

[arms.frozen]
strategy = { kind = "frozen" }

[arms.gate]
admission = { kind = "every", timesteps = 322_560 }  # a new concept every 16 weeks
# or { kind = "adwin", delta = 0.05, after_change = 40_320 }: every window since the
# last concept, 2 weeks of rows after ADWIN2 sees the error level change; or
# { kind = "wasserstein", ratio = 2, after_change = 40_320 }: after WATCH sees the
# input rows change; either takes max_interval, e.g. 161_280 (8 weeks).
# A list sweeps: timesteps = [40_320, 322_560] makes arms gate-timesteps40320 and
# gate-timesteps322560 (several lists: every combination).
[arms.gate.strategy]
kind = "validation_gate"       # or "replay": no validation, no labels needed
val_fraction = 0.2             # newest rows of each concept, never trained on
early_stopping = { max_epochs = 100, patience = 5, delta = 1e-5 }
capacity = 70_000              # windows the replay reservoir holds
gate_delta = 1e-5              # keep an update only if it beats the old weights by this
"""

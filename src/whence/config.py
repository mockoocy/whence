"""The experiment's config.toml: what one `whence run` executes.

Every `kind` field selects a variant of a tagged union. Config classes build the
object they describe, so the classes themselves stay free of config.
"""

import tomllib
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NaiveDatetime,
    NonNegativeFloat,
    PositiveInt,
    StringConstraints,
    ValidationInfo,
)
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from whence.admission import Admission, Every
from whence.aggregator import Method
from whence.cycle import EarlyStopping
from whence.esa_adb import ColumnSet, Scaler
from whence.forecaster import LstmForecaster
from whence.strategies import Frozen, Replay, Strategy, ValidationGate
from whence.zscore import ZScore

if TYPE_CHECKING:
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
    admission: EveryConfig | None = None  # None: no concept ever starts


# the name becomes the arm's directory in the run
ArmName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]


class Config(_Strict):
    seed: int | None = None  # `whence run` draws and records one when omitted
    # timesteps streamed per part (initial training, stream), for development runs
    limit: PositiveInt | None = None
    device: str = "cpu"
    score_batch_size: PositiveInt = 512
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    arms: dict[ArmName, ArmConfig] = Field(min_length=1)


def load_config(path: Path, data_root: Path) -> Config:
    with path.open("rb") as f:
        return Config.model_validate(
            tomllib.load(f),
            context={"data_root": data_root.resolve()},
        )


TEMPLATE = """\
# seed omitted: each run draws one and records it in config.json
# seed = 0
# limit = 3000                 # timesteps streamed per part, for development runs
device = "cuda"

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

[arms.frozen]
strategy = { kind = "frozen" }

[arms.gate]
admission = { kind = "every", timesteps = 322_560 }  # a new concept every 16 weeks
[arms.gate.strategy]
kind = "validation_gate"       # or "replay": no validation, no labels needed
val_fraction = 0.2             # newest rows of each concept, never trained on
early_stopping = { max_epochs = 100, patience = 5, delta = 1e-5 }
capacity = 70_000              # windows the replay reservoir holds
gate_delta = 1e-5              # keep an update only if it beats the old weights by this
"""

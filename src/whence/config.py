"""The experiment's config.toml: what one `whence run` executes.

Every `kind` field selects a variant of a tagged union. Strategy-side variants
build the object they describe, so the classes themselves stay free of config.
"""

import tomllib
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

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
)

from whence.admission import AdmitAll, CleanWindows
from whence.cycle import EarlyStopping, Epochs, Holdout
from whence.esa_adb import ColumnSet
from whence.strategies import Frozen, Replay, Strategy
from whence.triggers import Every

if TYPE_CHECKING:
    from whence.aggregator import Method


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


class DataConfig(_Strict):
    """For now there is one dataset: one file, split in time by `training.until`."""

    path: DataPath
    channels: Annotated[PaperChannels | TargetChannels, Field(discriminator="kind")] = (
        PaperChannels(kind="paper")
    )

    @cached_property
    def columns(self) -> ColumnSet:
        return self.channels.columns()


class LstmForecasterConfig(_Strict):
    kind: Literal["lstm_forecaster"]
    window_size: PositiveInt = 32
    horizon: PositiveInt = 4
    hidden_size: tuple[PositiveInt, PositiveInt] = (16, 16)
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.5
    lr: Annotated[float, Field(gt=0)] = 1e-3
    aggregation: Method = "first"

    @property
    def sample_len(self) -> int:
        """Rows the stream hands the model per window."""
        return self.window_size + self.horizon

    @property
    def n_scored(self) -> int:
        """Rows at the end of a window the model scores."""
        return self.horizon


ModelConfig = Annotated[LstmForecasterConfig, Field(discriminator="kind")]


class TrainingConfig(_Strict):
    # The initial period is every row before `until`, the stream every row from it.
    # The run fails if a target channel carries any label before it: the initial
    # model is trained on a contiguous, label-free interval.
    until: NaiveDatetime
    capacity: PositiveInt = 70_000
    val_fraction: Fraction = 0.2
    max_epochs: PositiveInt = 1000
    patience: PositiveInt = 20
    delta: NonNegativeFloat = 3e-4  # in standardised MSE, see LstmForecaster.loss


class EveryConfig(_Strict):
    kind: Literal["every"]
    windows: PositiveInt

    def build(self) -> Every:
        return Every(self.windows)


class AdmitAllConfig(_Strict):
    kind: Literal["admit_all"]

    def build(self) -> AdmitAll:
        return AdmitAll()


class CleanWindowsConfig(_Strict):
    kind: Literal["clean_windows"]

    def build(self) -> CleanWindows:
        return CleanWindows()


class HoldoutConfig(_Strict):
    val_fraction: Annotated[float, Field(ge=0, lt=1)]  # 0: keep only V(0)

    def build(self) -> Holdout:
        return Holdout(self.val_fraction)


class EpochsConfig(_Strict):
    kind: Literal["epochs"]
    epochs: PositiveInt = 1

    def build(self) -> Epochs:
        return Epochs(self.epochs)


class EarlyStoppingConfig(_Strict):
    kind: Literal["early_stopping"]
    max_epochs: PositiveInt = 100
    patience: PositiveInt = 5
    delta: NonNegativeFloat = 3e-4
    gated: bool = True  # reject an update that does not beat the old weights

    def build(self) -> EarlyStopping:
        return EarlyStopping(
            max_epochs=self.max_epochs,
            patience=self.patience,
            delta=self.delta,
            gated=self.gated,
        )


class ThresholdConfig(_Strict):
    """No model and no strategy: the harness scores it with threshold_baseline."""

    kind: Literal["threshold"]


class FrozenConfig(_Strict):
    kind: Literal["frozen"]

    def build(self) -> Strategy:
        return Frozen()


class ReplayConfig(_Strict):
    kind: Literal["replay"]
    trigger: Annotated[EveryConfig, Field(discriminator="kind")]
    admission: Annotated[
        AdmitAllConfig | CleanWindowsConfig,
        Field(discriminator="kind"),
    ]
    holdout: HoldoutConfig
    fit: Annotated[EpochsConfig | EarlyStoppingConfig, Field(discriminator="kind")]
    insert_every: PositiveInt = 32
    sample_size: PositiveInt = 4096

    def build(self) -> Strategy:
        return Replay(
            self.trigger.build(),
            self.admission.build(),
            self.holdout.build(),
            self.fit.build(),
            insert_every=self.insert_every,
            sample_size=self.sample_size,
        )


class ArmConfig(_Strict):
    strategy: Annotated[
        ThresholdConfig | FrozenConfig | ReplayConfig,
        Field(discriminator="kind"),
    ]
    label_delay: NonNegativeInt | Literal["never"] = 0  # rows

    @property
    def delay(self) -> int | None:
        """As LabelFeed takes it: None means labels never arrive."""
        return None if self.label_delay == "never" else self.label_delay


# the name becomes the arm's directory in the run
ArmName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]


class Config(_Strict):
    seed: int | None = None  # `whence run` draws and records one when omitted
    limit: PositiveInt | None = None  # windows read per part, for development runs
    device: str = "cpu"
    score_batch_size: PositiveInt = 512
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    arms: Annotated[dict[ArmName, ArmConfig], Field(min_length=1)]


def load_config(path: Path, data_root: Path) -> Config:
    with path.open("rb") as f:
        return Config.model_validate(
            tomllib.load(f),
            context={"data_root": data_root.resolve()},
        )


TEMPLATE = """\
# seed omitted: each run draws one and records it in config.json
# seed = 0
# limit = 3000                 # windows read per part, for development runs
device = "cuda"

[data]                         # relative paths are relative to --data-root
path = "ESA_Mission1.parquet"
# or { kind = "targets", channels_csv = "ESA-Mission1/channels.csv" }
channels = { kind = "paper" }

[model]
kind = "lstm_forecaster"
window_size = 32
horizon = 4
hidden_size = [16, 16]
aggregation = "first"          # "first", "mean" or "ewma"

[training]                     # the initial fit every arm starts from
# first label of any kind in any channel; the stream starts here
until = 2000-02-10T01:31:30
capacity = 70_000
val_fraction = 0.2
max_epochs = 1000
patience = 20
delta = 3e-4

[arms.threshold]
strategy = { kind = "threshold" }

[arms.frozen]
strategy = { kind = "frozen" }

[arms.replay-gated]
label_delay = 0                # rows, or "never"
[arms.replay-gated.strategy]
kind = "replay"
trigger = { kind = "every", windows = 322_560 }  # 16 weeks at 30 s
admission = { kind = "clean_windows" }
holdout = { val_fraction = 0.2 }  # newest rows of each cycle, never trained on
fit = { kind = "early_stopping" } # or { kind = "epochs", epochs = 1 }
"""

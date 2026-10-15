"""One `whence run`: a new, never overwritten directory under <experiment>/runs/.

runs/<timestamp>/
  config.toml   as written       config.json   as resolved (absolute paths, seed)
  meta.json     status, code and package versions, data hashes, timings
  labels.npy    per scored row: any target channel anomalous, shared by every arm
  log.txt       stdout
  initial/model.pt
  <arm>/scores.npy  per scored row, as `report` scores it (Result.scores)
  <arm>/refits.json  metrics.json
"""

import contextlib
import hashlib
import io
import json
import logging
import multiprocessing
import secrets
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, TextIO

import numpy as np

from whence.config import TEMPLATE, Config, load_config
from whence.experiment import Result, device_name, initial_training, report, run_arm

if TYPE_CHECKING:
    from pathlib import Path

    from whence.config import ArmConfig
    from whence.esa_adb import Scaler

PACKAGES = ("numpy", "polars", "pyarrow", "pydantic", "torch")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def new_experiment(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.toml"
    with path.open("x") as f:  # never replace a config someone has edited
        f.write(TEMPLATE)
    return path


def run_experiment(directory: Path, data_root: Path, jobs: int = 1) -> Path:
    config = load_config(directory / "config.toml", data_root)
    if config.seed is None:
        config = config.model_copy(update={"seed": secrets.randbelow(2**31)})

    runs = directory / "runs"
    runs.mkdir(exist_ok=True)
    run_dir = runs / datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S")
    # fails rather than share a directory with a run started this second
    run_dir.mkdir()
    shutil.copyfile(directory / "config.toml", run_dir / "config.toml")
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))
    logger.info("run %s: seed %d, device %s", run_dir, config.seed, config.device)

    meta = {"status": "running", "started": _now()}
    try:
        meta |= _provenance(config)
        _write_json(run_dir / "meta.json", meta)
        with (
            (run_dir / "log.txt").open("w") as log,
            contextlib.redirect_stdout(_Tee(sys.stdout, log)),
        ):
            _run_arms(config, run_dir, jobs)
        meta["status"] = "finished"
    except BaseException:
        meta |= {"status": "failed", "traceback": traceback.format_exc()}
        raise
    finally:
        meta["ended"] = _now()
        _write_json(run_dir / "meta.json", meta)
    return run_dir


def _run_arms(config: Config, run_dir: Path, jobs: int) -> None:
    initial = run_dir / "initial" / "model.pt"
    initial.parent.mkdir()
    scaler = config.data.scaling.build()  # fitted by initial_training, shared by arms
    initial_training(config, scaler).save(initial)

    # each arm in its own process, `jobs` at a time; spawned, as CUDA cannot fork
    with ProcessPoolExecutor(
        jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=partial(logging.basicConfig, level=logging.INFO, format=LOG_FORMAT),
    ) as pool:
        futures = {
            name: pool.submit(_run_arm, config, arm, initial, scaler, run_dir / name)
            for name, arm in config.arms.items()
        }
        labels = None
        for name, future in futures.items():
            arm_labels, printed = future.result()
            if labels is None:
                labels = arm_labels
                np.save(run_dir / "labels.npy", labels)
            assert np.array_equal(labels, arm_labels), "arms scored different rows"
            print(f"── {name}\n{printed}", end="")


def _run_arm(
    config: Config,
    arm: ArmConfig,
    initial: Path,
    scaler: Scaler,
    arm_dir: Path,
) -> tuple[np.ndarray, str]:
    """Run one arm into `arm_dir`; return its labels and what it printed."""
    # every arm starts from the same pre-trained model
    result = run_arm(config, arm, initial, scaler, desc=arm_dir.name)
    # scores reach disk before the metrics, which can still fail
    _save_result(arm_dir, result)
    # captured, so the parent prints it into log.txt
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        _write_json(arm_dir / "metrics.json", _counts(result) | report(result))
    return result.labels, printed.getvalue()


def _save_result(arm_dir: Path, result: Result) -> None:
    arm_dir.mkdir()
    np.save(arm_dir / "scores.npy", result.scores().astype(np.float32))
    _write_json(
        arm_dir / "refits.json",
        [{"step": step, "kept": kept} for step, kept in result.refits],
    )


def _counts(result: Result) -> dict[str, int]:
    return {
        "timesteps": len(result.labels),
        "anomalous": int(result.labels.sum()),
        "adaptations": len(result.adapted_at),
        "rejected": len(result.rejected_at),
    }


def _provenance(config: Config) -> dict:
    """What, besides the config, decides the outcome: code, environment, data."""
    return {
        "command": sys.argv,
        "python": sys.version,
        "packages": {name: version(name) for name in PACKAGES},
        "device": device_name(config.device),
        "data": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (config.data.path,)
        },
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


class _Tee:
    """Writes to several streams at once, so stdout also lands in log.txt."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

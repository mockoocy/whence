"""One `whence run`: a new, never overwritten directory under <experiment>/runs/.

runs/<timestamp>/
  config.toml   as written       config.json   as resolved (absolute paths, seed)
  meta.json     status, code and package versions, data hashes, timings
  labels.npy    per scored row: any target channel anomalous, shared by every arm
  log.txt       stdout
  initial/model.pt
  <arm>/errors.npy  error_scaler.json  refits.json  metrics.json
"""

import contextlib
import copy
import hashlib
import json
import secrets
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

import numpy as np

from whence.config import TEMPLATE, Config, load_config
from whence.experiment import Result, device_name, initial_training, report, run

PACKAGES = ("numpy", "polars", "pyarrow", "pydantic", "torch")


def new_experiment(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.toml"
    with path.open("x") as f:  # never replace a config someone has edited
        f.write(TEMPLATE)
    return path


def run_experiment(directory: Path, data_root: Path) -> Path:
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

    meta = {"status": "running", "started": _now()}
    try:
        meta |= _provenance(config, run_dir)
        _write_json(run_dir / "meta.json", meta)
        with (
            (run_dir / "log.txt").open("w") as log,
            contextlib.redirect_stdout(_Tee(sys.stdout, log)),
        ):
            print(f"run {run_dir}, seed {config.seed}")
            _run_arms(config, run_dir)
        meta["status"] = "finished"
    except BaseException:
        meta |= {"status": "failed", "traceback": traceback.format_exc()}
        raise
    finally:
        meta["ended"] = _now()
        _write_json(run_dir / "meta.json", meta)
    return run_dir


def _run_arms(config: Config, run_dir: Path) -> None:
    initial = initial_training(config)
    (run_dir / "initial").mkdir()
    initial.model.save(run_dir / "initial" / "model.pt")

    labels = None
    for name, arm in config.arms.items():
        # every arm starts from the same model, buffer and validation set
        result = run(
            copy.deepcopy(initial),
            arm.strategy.build(),
            config,
            label_delay=arm.delay,
            desc=name,
        )
        if labels is None:
            labels = result.labels
            np.save(run_dir / "labels.npy", labels)
        assert np.array_equal(labels, result.labels), "arms scored different rows"

        # errors reach disk before the metrics, which can still fail
        _save_result(run_dir / name, result, config)
        print(f"── {name}")
        _write_json(run_dir / name / "metrics.json", _counts(result) | report(result))
        del result  # otherwise its errors stay alive through the next arm's run


def _save_result(arm_dir: Path, result: Result, config: Config) -> None:
    arm_dir.mkdir()
    np.save(arm_dir / "errors.npy", result.errors.astype(np.float32))
    _write_json(
        arm_dir / "error_scaler.json",
        {
            "channels": list(config.data.columns.channels),
            "median": result.scaler.median.tolist(),
            "scale": result.scaler.scale.tolist(),
        },
    )
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


def _provenance(config: Config, run_dir: Path) -> dict:
    """What, besides the config, decides the outcome: code, environment, data."""
    return {
        "command": sys.argv,
        "git": _git(run_dir),
        "python": sys.version,
        "packages": {name: version(name) for name in PACKAGES},
        "device": device_name(config.device),
        "data": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (config.data.path,)
        },
    }


def _git(run_dir: Path) -> dict | None:
    """The commit of the whence source tree. With uncommitted changes their diff
    goes to git.diff, since the commit alone would not reproduce the run.
    """

    def git(*args: str) -> str:
        # args are literals from this module; git is taken from PATH like python is
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return None  # not a repository yet, or no git
    dirty = bool(git("status", "--porcelain").strip())
    if dirty:
        (run_dir / "git.diff").write_text(git("diff", "HEAD"))
    return {"commit": commit, "dirty": dirty}


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

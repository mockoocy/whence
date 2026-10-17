"""One `whence run`: a new, never overwritten directory under <experiment>/runs/.

runs/<timestamp>-seed<seed>/
  config.toml   as written       config.json   as resolved (absolute paths)
  meta.json     status, seed, code and package versions, data hashes, timings
  log.txt       stdout, ending with the table of results.md
  initial/model.pt
  dev/<arm>/    every arm but the baselines, streamed to evaluation.dev_until
  full/<arm>/   the baselines and the selected arm, streamed to the end
  dev/labels.npy, full/labels.npy  per scored row: any target channel anomalous
  <dev|full>/<arm>/scores.npy  per scored row, as `report` scores it (Result.scores)
  <dev|full>/<arm>/refits.json, metrics.json (by period: dev, and test in full/)
  selection.json  each candidate's dev Range-PR-VUS, and the one selected
  results.csv, results.md  per arm: its role and its dev and test metrics
"""

import contextlib
import csv
import hashlib
import json
import logging
import multiprocessing
import secrets
import shutil
import statistics
import sys
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, TextIO

import numpy as np

from whence.config import TEMPLATE, Config, load_config
from whence.esa_adb import row_at
from whence.experiment import Result, device_name, initial_training, report, run_arm

if TYPE_CHECKING:
    from concurrent.futures import Future
    from pathlib import Path

    from whence.config import ArmConfig
    from whence.esa_adb import Scaler

PACKAGES = ("numpy", "polars", "pyarrow", "pydantic", "torch")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SELECT_BY = "RANGE_PR_VOLUME"  # the dev metric that picks the arm to test
TABLE = {
    "RANGE_PR_VOLUME": "PR-VUS",
    "RANGE_ROC_VOLUME": "ROC-VUS",
    "TRIMMED_MAE": "tMAE",
}

Metrics = dict[str, dict[str, float]]  # by period, as `report` returns them

logger = logging.getLogger(__name__)


def new_experiment(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.toml"
    with path.open("x") as f:  # never replace a config someone has edited
        f.write(TEMPLATE)
    return path


def run_experiment(
    directory: Path,
    data_root: Path,
    jobs: int = 1,
    seed: int | None = None,  # None: drawn, and recorded in meta.json
) -> Path:
    config = load_config(directory / "config.toml", data_root)
    seed = secrets.randbelow(2**31) if seed is None else seed

    runs = directory / "runs"
    runs.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S")
    # fails rather than share a directory with a run of this seed started this
    # second; runs of other seeds, e.g. array jobs, may start in the same one
    run_dir = runs / f"{stamp}-seed{seed}"
    run_dir.mkdir()
    shutil.copyfile(directory / "config.toml", run_dir / "config.toml")
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))
    logger.info("run %s: seed %d, device %s", run_dir, seed, config.device)

    meta = {"status": "running", "started": _now(), "seed": seed}
    try:
        meta |= _provenance(config)
        _write_json(run_dir / "meta.json", meta)
        with (
            (run_dir / "log.txt").open("w") as log,
            contextlib.redirect_stdout(_Tee(sys.stdout, log)),
        ):
            _run_arms(config, run_dir, jobs, seed)
        meta["status"] = "finished"
    except BaseException:
        meta |= {"status": "failed", "traceback": traceback.format_exc()}
        raise
    finally:
        meta["ended"] = _now()
        _write_json(run_dir / "meta.json", meta)
    return run_dir


def _run_arms(config: Config, run_dir: Path, jobs: int, seed: int) -> None:
    """The candidates stream the dev rows and the baselines the whole file; the
    candidate with the best dev SELECT_BY then streams the whole file too.
    """
    initial = run_dir / "initial" / "model.pt"
    initial.parent.mkdir()
    scaler = config.data.scaling.build()  # fitted by initial_training, shared by arms
    initial_training(config, scaler, seed).save(initial)
    evaluation = config.evaluation
    candidates = [name for name in config.arms if name not in evaluation.baselines]

    # each arm in its own process, `jobs` at a time; spawned, as CUDA cannot fork
    with ProcessPoolExecutor(
        jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=partial(logging.basicConfig, level=logging.INFO, format=LOG_FORMAT),
    ) as pool:

        def launch(names: list[str], stage: str, end: datetime | None) -> _Futures:
            (run_dir / stage).mkdir(exist_ok=True)
            return {
                name: pool.submit(
                    _run_arm,
                    config,
                    config.arms[name],
                    initial,
                    scaler,
                    seed=seed,
                    arm_dir=run_dir / stage / name,
                    end=end,
                )
                for name in names
            }

        dev = launch(candidates, "dev", evaluation.dev_until)
        # the baselines run alongside: they are not candidates
        full = launch(list(evaluation.baselines), "full", None)
        dev_metrics = _collect(dev, run_dir / "dev")
        dev_scores = {
            name: metrics["dev"][SELECT_BY]
            for name, metrics in dev_metrics.items()
            if SELECT_BY in metrics.get("dev", {})
        }
        # on a tie, the first in the config
        selected = max(dev_scores, key=dev_scores.__getitem__, default=None)
        selection = {"by": SELECT_BY, "dev": dev_scores, "selected": selected}
        _write_json(run_dir / "selection.json", selection)
        if selected is not None:
            full |= launch([selected], "full", None)
        full_metrics = _collect(full, run_dir / "full")
    _write_results(run_dir, _results(config, dev_metrics, full_metrics, selected))


_Futures = dict[str, "Future[tuple[np.ndarray, Metrics]]"]


def _collect(futures: _Futures, stage_dir: Path) -> dict[str, Metrics]:
    """Wait for the arms in order; save their labels, which must agree."""
    labels, metrics = None, {}
    for name, future in futures.items():
        arm_labels, metrics[name] = future.result()
        if labels is None:
            labels = arm_labels
            np.save(stage_dir / "labels.npy", labels)
        assert np.array_equal(labels, arm_labels), "arms scored different rows"
        print(f"── {stage_dir.name}/{name} finished")
    return metrics


def _run_arm(  # noqa: PLR0913 - what a spawned process needs
    config: Config,
    arm: ArmConfig,
    initial: Path,
    scaler: Scaler,
    *,
    seed: int,
    arm_dir: Path,
    end: datetime | None,
) -> tuple[np.ndarray, Metrics]:
    """Run one arm into `arm_dir`, scoring the rows before `end`; return its labels
    and metrics.
    """
    # every arm starts from the same pre-trained model
    result = run_arm(
        config,
        arm,
        initial,
        scaler,
        seed=seed,
        desc=arm_dir.name,
        end=end,
    )
    # scores reach disk before the metrics, which can still fail
    _save_result(arm_dir, result)
    metrics = report(result, _periods(config, end))
    _write_json(arm_dir / "metrics.json", metrics)
    return result.labels, metrics


def _periods(config: Config, end: datetime | None) -> dict[str, slice]:
    """The scored rows of each period of a run that stops before `end`."""
    if end is not None:  # a candidate's run, which ends at dev_until
        return {"dev": slice(None)}
    # scored row i is row `until` + i of the file
    path = config.data.path
    split = row_at(path, config.evaluation.dev_until) - row_at(
        path,
        config.training.until,
    )
    return {"dev": slice(None, split), "test": slice(split, None)}


def _save_result(arm_dir: Path, result: Result) -> None:
    arm_dir.mkdir()
    np.save(arm_dir / "scores.npy", result.scores().astype(np.float32))
    _write_json(
        arm_dir / "refits.json",
        [{"step": step, "kept": kept} for step, kept in result.refits],
    )


def _results(
    config: Config,
    dev: dict[str, Metrics],
    full: dict[str, Metrics],
    selected: str | None,
) -> list[dict]:
    """One row per arm, in config order: its role and its metrics as
    <period>.<metric>; the dev ones from its dev run when it had one.
    """
    rows = []
    for name in config.arms:
        role = (
            "baseline"
            if name in config.evaluation.baselines
            else "selected"
            if name == selected
            else ""
        )
        periods = full.get(name, {}) | dev.get(name, {})
        metrics = {
            f"{p}.{k}": v for p, values in periods.items() for k, v in values.items()
        }
        rows.append({"arm": name, "role": role} | metrics)
    return rows


def _write_results(run_dir: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    header = ["arm", "role"]
    for period in ("dev", "test"):
        header += [f"{period} {label}" for label in TABLE.values()]
        header.append(f"{period} kept/rejected")
    body = [
        [row["arm"], row["role"], *_cells(row, "dev"), *_cells(row, "test")]
        for row in rows
    ]
    table = _table(header, body)
    (run_dir / "results.md").write_text(table)
    print(table, end="")


def _cells(row: dict, period: str) -> list[str]:
    values = [row.get(f"{period}.{key}") for key in TABLE]
    kept = row.get(f"{period}.adaptations")
    refits = "" if kept is None else f"{kept:g}/{row[f'{period}.rejected']:g}"
    return [*("" if v is None else f"{v:.4f}" for v in values), refits]


def _table(header: list[str], rows: list[list[str]]) -> str:
    """A Markdown table, padded so it reads as plain text too."""
    widths = [max(map(len, column)) for column in zip(header, *rows, strict=True)]
    lines = [header, ["-" * width for width in widths], *rows]
    return "".join(
        "| "
        + " | ".join(c.ljust(w) for c, w in zip(line, widths, strict=True))
        + " |\n"
        for line in lines
    )


SELECTED = "(selected)"  # the summary row of whichever arm each run selected


def summarise(paths: list[Path], reference: str | None = None) -> str:
    """Mean ± sd over runs of one experiment of each arm's dev and test metrics,
    and how often it was selected; with `reference`, also each arm's test metrics
    minus the reference arm's of the same run.
    """
    runs = [_read_results(run_dir) for run_dir in _run_dirs(paths)]
    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, rows in enumerate(runs):
        for row in rows:
            groups.setdefault(row["arm"], []).append((i, row))
    groups[SELECTED] = [
        (i, row)
        for i, rows in enumerate(runs)
        for row in rows
        if row["role"] == "selected"
    ]
    refs = {
        i: row for i, rows in enumerate(runs) for row in rows if row["arm"] == reference
    }
    if reference is not None and not refs:
        msg = f"no arm {reference} in these runs"
        raise ValueError(msg)

    keys = [f"{period}.{key}" for period in ("dev", "test") for key in TABLE]
    deltas = [key for key in keys if key.startswith("test.")] if refs else []
    header = ["arm", "runs", "selected", *map(_label, keys)]
    header += [f"Δ {_label(key)}" for key in deltas]
    body = []
    for arm, members in groups.items():
        if not members:  # no run selected an arm
            continue
        selected = sum(row["role"] == "selected" for _, row in members)
        cells = [arm, str(len(members)), "" if arm == SELECTED else str(selected)]
        cells += [_stat([row[key] for _, row in members if key in row]) for key in keys]
        cells += [
            _stat(
                [
                    row[key] - refs[i][key]
                    for i, row in members
                    if key in row and key in refs.get(i, {})
                ],
            )
            for key in deltas
        ]
        body.append(cells)
    return _table(header, body)


def _run_dirs(paths: list[Path]) -> list[Path]:
    """Runs given directly, or every finished run of experiments; their configs
    must match, as `whence run` draws a new seed but nothing else.
    """
    dirs = []
    for path in paths:
        found = (
            [path]
            if (path / "results.csv").exists()
            else sorted(p.parent for p in path.glob("runs/*/results.csv"))
        )
        if not found:
            msg = f"{path}: no results.csv, nor in runs/*/"
            raise ValueError(msg)
        dirs += found
    configs = [tomllib.loads((d / "config.toml").read_text()) for d in dirs]
    if differ := [
        str(d) for d, c in zip(dirs, configs, strict=True) if c != configs[0]
    ]:
        msg = f"configs differ from {dirs[0]}'s: {', '.join(differ)}"
        raise ValueError(msg)
    return dirs


def _read_results(run_dir: Path) -> list[dict]:
    with (run_dir / "results.csv").open(newline="") as f:
        return [
            {
                k: v if k in ("arm", "role") else float(v)
                for k, v in row.items()
                if v or k == "role"
            }
            for row in csv.DictReader(f)
        ]


def _label(key: str) -> str:
    period, metric = key.split(".")
    return f"{period} {TABLE[metric]}"


def _stat(values: list[float]) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.4f} (1)"
    return f"{mean:.4f} ± {statistics.stdev(values):.4f} ({len(values)})"


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

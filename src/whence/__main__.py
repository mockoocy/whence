import argparse
import logging
import sys
import traceback
from pathlib import Path

from whence.config import load_config
from whence.runs import LOG_FORMAT, new_experiment, run_experiment, summarise


def main() -> None:
    parser = argparse.ArgumentParser(prog="whence")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="create an experiment from a template config")
    new.add_argument("experiment", type=Path)

    run = commands.add_parser(
        "run",
        help="run every arm of each experiment, one after another, into "
        "<experiment>/runs/<timestamp>",
    )
    run.add_argument("experiments", type=Path, nargs="+")
    run.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        help="what relative data paths in the config are relative to (default: cwd)",
    )
    run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="arms run at once, each in its own process (default: 1)",
    )
    run.add_argument(
        "--seed",
        type=int,
        help="seeds the initial model and every arm (default: drawn per run, and "
        "recorded in meta.json)",
    )

    summary = commands.add_parser(
        "summary",
        help="mean ± sd of each arm's dev and test metrics over runs of one experiment",
    )
    summary.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="run directories, or experiments for all their finished runs",
    )
    summary.add_argument(
        "--reference",
        help="arm whose test metrics are subtracted from the others', run by run",
    )

    args = parser.parse_args()
    if args.command == "new":
        print(f"wrote {new_experiment(args.experiment)}")
        return
    if args.command == "summary":
        try:
            print(summarise(args.runs, args.reference), end="")
        except ValueError as e:
            sys.exit(str(e))
        return
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    # every config is checked up front, so a typo in the last one fails now, not
    # after hours of the others
    for experiment in args.experiments:
        load_config(experiment / "config.toml", args.data_root)
    failed = []
    for experiment in args.experiments:
        try:
            run_dir = run_experiment(experiment, args.data_root, args.jobs, args.seed)
            print(f"finished {run_dir}")
        except Exception:  # noqa: BLE001 -- in its meta.json; the others still run
            traceback.print_exc()
            failed.append(experiment)
    if failed:
        sys.exit(f"failed: {' '.join(map(str, failed))}")


if __name__ == "__main__":
    main()

import argparse
import sys
import traceback
from pathlib import Path

from whence.config import load_config
from whence.runs import new_experiment, run_experiment


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

    args = parser.parse_args()
    if args.command == "new":
        print(f"wrote {new_experiment(args.experiment)}")
        return
    # every config is checked up front, so a typo in the last one fails now, not
    # after hours of the others
    for experiment in args.experiments:
        load_config(experiment / "config.toml", args.data_root)
    failed = []
    for experiment in args.experiments:
        try:
            print(f"finished {run_experiment(experiment, args.data_root)}")
        except Exception:  # noqa: BLE001 -- in its meta.json; the others still run
            traceback.print_exc()
            failed.append(experiment)
    if failed:
        sys.exit(f"failed: {' '.join(map(str, failed))}")


if __name__ == "__main__":
    main()

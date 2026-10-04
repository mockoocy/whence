import argparse
from pathlib import Path

from whence.runs import new_experiment, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="whence")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="create an experiment from a template config")
    new.add_argument("experiment", type=Path)

    run = commands.add_parser(
        "run",
        help="run every arm into <experiment>/runs/<timestamp>",
    )
    run.add_argument("experiment", type=Path)
    run.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        help="what relative data paths in the config are relative to (default: cwd)",
    )

    args = parser.parse_args()
    if args.command == "new":
        print(f"wrote {new_experiment(args.experiment)}")
    else:
        print(f"finished {run_experiment(args.experiment, args.data_root)}")


if __name__ == "__main__":
    main()

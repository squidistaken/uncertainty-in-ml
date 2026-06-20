import argparse
import importlib
import sys

COMMANDS = {
    "download": (
        "src.data.download_data",
        "Download the raw dataset from Kaggle.",
    ),
    "preprocess": (
        "src.features.preprocess_data",
        "Preprocess the data and engineer features.",
    ),
    "train": (
        "src.training.train",
        "Train a model.",
    ),
    "evaluate": (
        "src.training.evaluate",
        "Evaluate a model.",
    ),
    "tune": (
        "src.training.tune_hyperparameters",
        "Tune model hyperparameters via cross-validation.",
    ),
}


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        prog="uml",
        description="S&P 500 forecasting and uncertainty quantification "
        "pipeline.",
        epilog="Run 'uv run main.py <command> --help' for command-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        metavar="command",
        help="The pipeline stage to run: "
        + ", ".join(f"{name} ({desc})" for name, (_, desc) in COMMANDS.items()),
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected command.",
    )

    args = parser.parse_args()

    module_path, _ = COMMANDS[args.command]
    module = importlib.import_module(module_path)

    sys.argv = [f"uml {args.command}", *args.args]
    module.main()


if __name__ == "__main__":
    main()

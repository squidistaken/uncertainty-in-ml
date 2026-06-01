from src.features.spy_preprocessor import SPYPreprocessor
from src.constants import LOGGER
import argparse


def main():
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="SPY Data Preprocessing Pipeline"
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=21,
        help="The size of the sliding window for sequence generation. Defaults to 21.",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2016,
        help="The year to split the data for training. Defaults to 2016.",
    )
    parser.add_argument(
        "--val-end-year",
        type=int,
        default=2020,
        help="The year to split the data for validation. Defaults to 2020.",
    )
    args = parser.parse_args()

    LOGGER.info("Starting data preprocessing pipeline...")

    preprocessor = SPYPreprocessor(
        window_size=args.window_size,
        train_end_year=args.train_end_year,
        val_end_year=args.val_end_year,
    )
    preprocessor.run()

    LOGGER.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()

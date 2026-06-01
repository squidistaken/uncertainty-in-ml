import argparse
from pathlib import Path
from src.constants import DATA_DIR, LOGGER, DEBUG
import kaggle


class DataDownloader:
    """Class to handle downloading and organizing the dataset using the official Kaggle API."""

    def __init__(self, raw_data_path: Path) -> None:
        """Initialise the class.

        Args:
            raw_data_path (Path): The path to the raw data directory.
        """
        self.raw_data_path = raw_data_path

    def _download_from_kaggle(self) -> None:
        """Download and extract the dataset from Kaggle using the official API."""
        LOGGER.info("Downloading data from Kaggle...")

        try:
            kaggle.api.authenticate()
        except Exception as e:
            LOGGER.error(f"Kaggle authentication failed: {e}")
            raise e

        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(
            "yousefeddin/s-and-p-500-stock-price-end-of-2024",
            path=str(self.raw_data_path),
            unzip=True,
        )
        LOGGER.info("Data downloaded and extracted successfully.")

    def run(self, force_download: bool = False) -> None:
        """Run the download and organization pipeline, saving files as raw data.

        Args:
            force_download (bool): Whether to force download the data even if
                                   it already exists. Defaults to False.
        """
        if DEBUG:
            LOGGER.debug(f"Force Download: {force_download}")

        spy_file = self.raw_data_path / "SPY.csv"
        if not force_download and spy_file.exists():
            LOGGER.info("Raw data already exists. Skipping download.")
            return

        self._download_from_kaggle()
        LOGGER.info("Data prepared as raw files successfully.")


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Download and copy Kaggle dataset to raw directory."
    )

    parser.add_argument(
        "--force",
        "--force-download",
        dest="force_download",
        action="store_true",
        default=False,
        help="Force download the data even if it exists",
    )

    args = parser.parse_args()
    raw_path = DATA_DIR / "raw"
    downloader = DataDownloader(raw_path)

    downloader.run(force_download=args.force_download)


if __name__ == "__main__":
    main()

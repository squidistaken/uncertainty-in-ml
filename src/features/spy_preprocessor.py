from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from src.constants import LOGGER, DATA_DIR, DEBUG
from typing import cast


# TODO: If it is really needed, we can explore more intense feature
#       engineering. We will need to evaluate as we go.
class SPYPreprocessor:
    """SPY Processor class for preparing data for training."""

    def __init__(
        self, window_size=21, train_end_year=2016, val_end_year=2020
    ) -> None:
        """Initialise the class.

        Args:
            window_size (int, optional): The size of the sliding window for
                                         sequence generation. Defaults to 21.
            train_end_year (int, optional): The year to split the data for
                                            training. Defaults to 2016.
            val_end_year (int, optional): The year to split the data for
                                          validation. Defaults to 2020.
        """

        self.window_size = window_size
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year
        self.scaler = StandardScaler()

    def _calculate_log_returns(self) -> pd.DataFrame:
        """Calculate log returns for the data.

        Returns:
            pd.DataFrame: A DataFrame with the log returns.
        """
        df = pd.read_csv(DATA_DIR / "raw" / "SPY.csv")
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)

        LOGGER.info("Calculating log returns...")

        # Log returns are calculated as: ln(P_t / P_{t-1})
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna(subset=["Log_Return"]).copy()

        return df

    def _split_data_by_year(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the data for training, validation, and testing datasets.

        Args:
            df (pd.DataFrame): The dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
                                                             the training,
                                                             validation, and
                                                             test datasets.
        """
        LOGGER.info("Splitting data...")
        LOGGER.info(" - Training Set: 1993 - 2016.")
        LOGGER.info(" - Validation Set: 2017 - 2020.")
        LOGGER.info(" - Testing Set: 2021 - 2024.")

        df["Year"] = df["Date"].dt.year

        train_df = df[df["Year"] <= self.train_end_year].copy()
        val_df = df[
            (df["Year"] > self.train_end_year)
            & (df["Year"] <= self.val_end_year)
        ].copy()
        test_df = df[df["Year"] > self.val_end_year].copy()

        if DEBUG:
            LOGGER.debug(f"Training Dataset Size: {len(train_df)} days.")
            LOGGER.debug(f"Validation Dataset Size: {len(val_df)} days.")
            LOGGER.debug(f"Testing Dataset Size: {len(test_df)} days.")

        return train_df, val_df, test_df

    def create_sequences(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create the sequences based on `window_size` to predict the
        subsequent step.

        Args:
            data (np.ndarray): The input data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the sequences of
                                           X and y.
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : (i + self.window_size)])
            y.append(data[i + self.window_size])

        if DEBUG:
            LOGGER.debug(f"X shape: {np.array(X).shape}")
            LOGGER.debug(f"y shape: {np.array(y).shape}")

        return np.array(X), np.array(y)

    def run(self) -> None:
        """RUn the pipeline."""
        output_dir = DATA_DIR / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self._calculate_log_returns()
        train_df, val_df, test_df = self._split_data_by_year(df)

        # We standardise the data for a better Neural Net and GP convergence.
        # We fit only on the training dataset to prevent data leakage.
        LOGGER.info("Extracting log returns...")

        train_returns = train_df["Log_Return"].values.reshape(-1, 1)
        val_returns = val_df["Log_Return"].values.reshape(-1, 1)
        test_returns = test_df["Log_Return"].values.reshape(-1, 1)

        LOGGER.info("Scaling data...")

        train_scaled = self.scaler.fit_transform(train_returns)
        val_scaled = self.scaler.transform(val_returns)
        test_scaled = self.scaler.transform(test_returns)

        LOGGER.info(
            f"Creating sequences with window size {self.window_size}..."
        )

        X_train, y_train = self.create_sequences(train_scaled)
        X_val, y_val = self.create_sequences(val_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        LOGGER.info("Saving arrays...")
        np.savez_compressed(
            os.path.join(output_dir, "train.npz"), X=X_train, y=y_train
        )
        np.savez_compressed(
            os.path.join(output_dir, "val.npz"), X=X_val, y=y_val
        )
        np.savez_compressed(
            os.path.join(output_dir, "test.npz"), X=X_test, y=y_test
        )

        # We save the scaler parameters so we can reverse the scaling later
        # during evaluation.
        LOGGER.info("Saving scaler parameters...")
        np.save(
            os.path.join(output_dir, "scaler_mean.npy"),
            cast(ArrayLike, self.scaler.mean_),
        )
        np.save(
            os.path.join(output_dir, "scaler_scale.npy"),
            cast(ArrayLike, self.scaler.scale_),
        )

        LOGGER.info("Preprocessing complete.")


def main():
    """Run the script."""
    LOGGER.info("Starting data preprocessing pipeline...")

    processor = SPYPreprocessor()
    processor.run()

    LOGGER.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()

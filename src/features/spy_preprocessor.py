from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from src.constants import LOGGER, DATA_DIR, DEBUG
from typing import cast


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

        LOGGER.info("Sorting data by date...")

        df = df.sort_values("Date", ascending=True).reset_index(drop=True)

        LOGGER.info("Handling dividends...")

        if "Dividends" in df.columns:
            df["Log_Return"] = np.log(
                (df["Close"] + df["Dividends"]) / df["Close"].shift(1)
            )
        else:
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        LOGGER.info("Calculating log returns...")

        df = df.dropna().reset_index(drop=True)

        LOGGER.info("Adding time index...")

        df["Time_Index"] = np.arange(len(df))

        return df

    def _create_sequences(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the sequences based on `window_size` to predict the
        subsequent step.

        Args:
            data (np.ndarray): The input data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the sequences of
                                           X and y.
        """
        X, y, dates, time_indices = [], [], [], []
        values = df["Scaled_Log_Return"].values
        date_vals = df["Date"].values
        t_idx_vals = df["Time_Index"].values

        for i in range(len(df) - self.window_size):
            X.append(values[i : i + self.window_size])
            y.append(values[i + self.window_size])
            dates.append(date_vals[i + self.window_size])
            time_indices.append(t_idx_vals[i + self.window_size])

        # LSTMs require a 3D tensor, while GPs only need a 2D tensor.
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)

        if DEBUG:
            LOGGER.debug(f"X shape: {np.array(X).shape}")
            LOGGER.debug(f"y shape: {np.array(y).shape}")

        return X, y, np.array(dates), np.array(time_indices)

    def run(self) -> None:
        """RUn the pipeline."""
        output_dir = DATA_DIR / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self._calculate_log_returns()
        train_mask = df["Date"].dt.year <= self.train_end_year
        train_returns = df.loc[train_mask, ["Log_Return"]]

        LOGGER.info("Fitting scaler on training data...")

        self.scaler.fit(train_returns)

        df["Scaled_Log_Return"] = self.scaler.transform(df[["Log_Return"]])

        LOGGER.info(
            f"Creating sequences with window size {self.window_size}..."
        )

        X_all, y_all, dates_all, t_idx_all = self._create_sequences(df)

        LOGGER.info("Splitting sequences chronologically by target dates...")

        years_all = pd.Series(dates_all).dt.year.values

        train_idx = years_all <= self.train_end_year
        val_idx = (years_all > self.train_end_year) & (
            years_all <= self.val_end_year
        )
        test_idx = years_all > self.val_end_year

        X_train, y_train, t_train = (
            X_all[train_idx],
            y_all[train_idx],
            t_idx_all[train_idx],
        )
        X_val, y_val, t_val = X_all[val_idx], y_all[val_idx], t_idx_all[val_idx]
        X_test, y_test, t_test = (
            X_all[test_idx],
            y_all[test_idx],
            t_idx_all[test_idx],
        )

        LOGGER.info("Saving arrays...")

        np.savez_compressed(
            os.path.join(output_dir, "train.npz"),
            X=X_train,
            y=y_train,
            time_idx=t_train,
        )
        np.savez_compressed(
            os.path.join(output_dir, "val.npz"),
            X=X_val,
            y=y_val,
            time_idx=t_val,
        )
        np.savez_compressed(
            os.path.join(output_dir, "test.npz"),
            X=X_test,
            y=y_test,
            time_idx=t_test,
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

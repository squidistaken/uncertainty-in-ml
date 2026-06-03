from torch.utils.data import Dataset
import numpy as np
from typing import Literal, Union
from src.constants import DATA_DIR, LOGGER, DEBUG
import torch


class SPYDataset(Dataset):
    "PyTorch Dataset class for the S&P500 data."

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialise the class.

        Args:
            split (Literal["train", "val", "test], optional): The data split to
                                                              load. Defaults to
                                                              "train".
            device (Union[str, torch.device], optional): The device to load the
                                                         tensors. Defaults to "cpu".
        """
        if DEBUG:
            LOGGER.debug(f"Split: {split}")
            LOGGER.debug(f"Device: {device}")

        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")

        self.split = split
        self.device = device

        file_path = DATA_DIR / "processed" / f"{split}.npz"

        if not file_path.exists():
            LOGGER.error(
                f"Data file not found: {file_path}. Run the preprocessor first."
            )
            raise FileNotFoundError(f"Missing {file_path}")

        LOGGER.info(f"Loading {split} dataset from {file_path}...")

        data = np.load(file_path)

        # We use float32 as it is the standard for PyTorch neural network weights.
        self.X = torch.tensor(
            data["X"], dtype=torch.float32, device=self.device
        )
        self.y = torch.tensor(
            data["y"], dtype=torch.float32, device=self.device
        )

        # Time indices are useful for plotting or specific GP kernels.
        self.time_idx = torch.tensor(
            data["time_idx"], dtype=torch.float32, device=self.device
        )

        LOGGER.info(
            f"{split.capitalize()} dataset loaded. X shape: {self.X.shape}, y shape: {self.y.shape}"
        )

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: A total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence sample and its target.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                - the input sequence;
                - the target value;
                - the time index for the target value.
        """
        return self.X[index], self.y[index], self.time_idx[index]

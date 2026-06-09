import numpy as np
import torch
from typing import Union
from src.constants import DATA_DIR


class SPYScaler:
    """
    Utility class to load saved Standard Scaler parameters and handle forward
    and inverse transforms for predictions, standard deviations, and variances.
    """

    def __init__(self) -> None:
        """Initialise the class."""
        mean_path = DATA_DIR / "processed" / "scaler_mean.npy"
        scale_path = DATA_DIR / "processed" / "scaler_scale.npy"

        self.mean = float(np.load(mean_path).squeeze())
        self.scale = float(np.load(scale_path).squeeze())

    def inverse_transform_predictions(
        self, y_scaled: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert scaled model predictions back to raw percentage returns.

        Args:
            y_scaled (Union[np.ndarray, torch.Tensor]): The scaled predictions.

        Returns:
            Union[np.ndarray, torch.Tensor]: An unscaled version of the input.
        """
        return (y_scaled * self.scale) + self.mean

    def inverse_transform_std(
        self, std_scaled: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert scaled standard deviations (uncertainties) back to original scales.

        Args:
            std_scaled (Union[np.ndarray, torch.Tensor]): The scaled standard
                                                          deviations.

        Returns:
            Union[np.ndarray, torch.Tensor]: An unscaled version of the input.
        """
        return std_scaled * self.scale

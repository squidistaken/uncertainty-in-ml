from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from src.constants import MODELS_DIR, LOGGER


class BaseModel(ABC):
    """Abstract Base Class for the machine learning models of the project.

    It is intended for two models:
    1. Baseline Model: A deterministic Long Short-Term Memory (LSTM) network.
    2. Proposed Model: An approximate Variational Gaussian Process (GP).
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the class.

        Args:
            **kwargs: The configuration.
        """
        self.config = kwargs

    @abstractmethod
    def forward_pass(self, x: Any) -> Any:
        """
        Predict the target for the input sequence.

        Args:
            x (Any): The input data sequence.

        Returns:
            Any: A prediction of the target.
        """
        ...

    @abstractmethod
    def backward_pass(self, x_train: Any, y_train: Any, **kwargs: Any) -> Any:
        """
        Train the model on the provided data.

        Args:
            x_train (Any): The training data sequences.
            y_train (Any): The training targets.
            **kwargs: The configuration.

        Returns:
            Any: A trained model state.
        """
        ...

    @abstractmethod
    def evaluate(self, x_test: Any, y_test: Any, fit_noise: bool = True) -> Any:
        """
        Test the performance of the model.

        Args:
            x_test (Any): The testing data features.
            y_test (Any): The testing data true targets.
            fit_noise (bool): Whether to fit noise. Defaults to True.

        Returns:
            Any: A metric(s) indicating the performance.
        """
        ...

    def save_model(self, filename: Optional[str] = None) -> None:
        """
        Save a model and its weights.

        Args:
            filename (str, optional): The filename to save the model to.
                                      Defaults to None.
        """
        if filename is None:
            filename = f"{self.__class__.__name__}.pt"

        path = MODELS_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Saving {self.__class__.__name__} to {path}...")
        self._save_weights(path)
        LOGGER.info("Model saved successfully.")

    @abstractmethod
    def _save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path (Path): The path to save the model weights to.
        """
        ...

    def load(self, filename: Optional[str] = None) -> None:
        """
        Load a trained model.

        Args:
            filename (str, optional): The filename to load the model from.
                                      Defaults to None.
        """
        if filename is None:
            filename = f"{self.__class__.__name__}.pt"

        path = MODELS_DIR / filename

        if not path.exists():
            LOGGER.error(f"Model file not found: {path}")
            raise FileNotFoundError(
                f"Cannot load model. File does not exist: {path}"
            )

        LOGGER.info(f"Loading {self.__class__.__name__} from {path}...")
        self._load_weights(path)
        LOGGER.info("Model loaded successfully.")

    @abstractmethod
    def _load_weights(self, path: Path) -> None:
        """
        Load model weights.

        Args:
            path (Path): The path to load the model weights from.
        """
        ...

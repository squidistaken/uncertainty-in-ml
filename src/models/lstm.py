from typing import Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.base_model import BaseModel
from src.constants import DEVICE
from src.utils.scaling import SPYScaler


class LSTM(nn.Module, BaseModel):
    """Deterministic Long Short-Term Memory (LSTM) model class."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """Initialise the model.

        Args:
            input_size (int): The number of input features per time step.
            hidden_size (int): The size of the LSTM hidden state. Defaults
                               to 64.
            num_layers (int): The number of stacked LSTM layers. Defaults to 2.
            dropout (float): The dropout probability between stacked layers.
                             Defaults to 0.2.
            **kwargs: The configuration.
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self, **kwargs)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

        self.to(DEVICE)

        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scaler = SPYScaler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LSTM prior forward pass.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: A flat tensor of shape with the predicted next-step
                          returns.
        """
        _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]
        prediction = self.fc(last_hidden)

        return prediction.flatten()

    def forward_pass(self, x: torch.Tensor) -> torch.distributions.Normal:
        """
        Predict the target for the input sequence.

        Args:
            x (torch.Tensor): The input data sequence.

        Returns:
            torch.distributions.Normal: A prediction of the target.
        """
        self.eval()

        with torch.no_grad():
            x = x.to(DEVICE)
            mean = self(x)
            std = torch.full_like(mean, 1e-3)

            return torch.distributions.Normal(mean, std)

    def backward_pass(
        self,
        x_train: DataLoader,
        y_train: Optional[Any] = None,
        **kwargs,
    ) -> float:
        """Train the model for one epoch over the provided data.

        Args:
            x_train (DataLoader): The training data sequences.
            y_train (Optional[Any], optional): The testing data true targets.
            **kwargs: The configuration.

        Returns:
            float: A metric(s) indicating the performance.
        """
        lr = kwargs.get("lr", 0.001)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()

        epoch_loss = 0.0

        for batch_x, batch_y, _ in x_train:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            self.optimizer.zero_grad()

            predictions = self(batch_x)
            loss = self.criterion(predictions, batch_y)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(x_train)

    def evaluate(
        self,
        x_test: DataLoader,
        y_test: torch.Tensor,
    ) -> dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (DataLoader): The testing data features.
            y_test (torch.Tensor): The testing data true targets.

        Returns:
            dict[str, float]: A metric(s) indicating the performance.
        """
        self.eval()

        predictions = []

        with torch.no_grad():
            for batch_x, _, _ in x_test:
                batch_x = batch_x.to(DEVICE)
                predictions.append(self(batch_x))

        predictions = torch.cat(predictions)
        pred_np = predictions.cpu().numpy()
        target_np = y_test.cpu().numpy()
        raw_preds = self.scaler.inverse_transform_predictions(pred_np)
        raw_targets = self.scaler.inverse_transform_predictions(target_np)

        # Financial & Point Metrics:
        # 1. Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((raw_preds - raw_targets) ** 2))

        # 2. Hit Rate/Directional Accuracy
        hit_rate = np.mean(np.sign(raw_preds) == np.sign(raw_targets)) * 100

        # Probabilistic Metrics
        # 1. Negative Log-Likelihood (NLL)
        residual_var = np.var(target_np - pred_np) + 1e-6
        std = float(np.sqrt(residual_var))
        normal_dist = torch.distributions.Normal(
            torch.tensor(pred_np),
            std,
        )
        nll = -normal_dist.log_prob(torch.tensor(target_np)).mean().item()

        # 2. Expected Calibration Error (ECE)
        #    NOTE: As a deterministic model, no per-prediction calibration
        #          estimate is available.
        ece = 0.0

        return {
            "RMSE": float(rmse),
            "Hit_Rate": float(hit_rate),
            "NLL": float(nll),
            "ECE": float(ece),
        }

    def _save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path (Path): The path to save the model weights to.
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )

    def _load_weights(self, path: Path) -> None:
        """
        Load model weights.

        Args:
            path (Path): The path to load the model weights from.
        """
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )

        self.load_state_dict(checkpoint["model_state_dict"])

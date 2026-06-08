from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel
from src.constants import DEVICE
from src.utils.scaling import SPYScaler


class LSTMNetwork(nn.Module):
    """
    Core LSTM architecture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]

        prediction = self.fc(last_hidden)

        return prediction.flatten()


class LSTMModel(BaseModel):
    """
    BaseModel-compatible wrapper around the LSTM network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)

        self.optimizer = None
        self.criterion = nn.MSELoss()

        self.scaler = SPYScaler()

    def forward_pass(self, x: torch.Tensor):

        self.model.eval()

        with torch.no_grad():

            x = x.to(DEVICE)

            mean = self.model(x)

            #
            # Trainer expects:
            #
            # predictive_dist.mean
            # predictive_dist.variance
            #
            variance = torch.ones_like(mean) * 1e-6

            return SimpleNamespace(
                mean=mean,
                variance=variance,
            )

    def backward_pass(
        self,
        x_train: DataLoader,
        y_train=None,
        **kwargs,
    ) -> float:

        lr = kwargs.get("lr", 0.001)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
            )

        self.model.train()

        epoch_loss = 0.0

        for batch_x, batch_y, _ in x_train:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            self.optimizer.zero_grad()

            predictions = self.model(batch_x)

            loss = self.criterion(
                predictions,
                batch_y,
            )

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(x_train)

    def evaluate(
        self,
        x_test: DataLoader,
        y_test: torch.Tensor,
    ) -> dict[str, float]:

        self.model.eval()

        predictions = []

        with torch.no_grad():

            for batch_x, _, _ in x_test:

                batch_x = batch_x.to(DEVICE)

                predictions.append(
                    self.model(batch_x)
                )

        predictions = torch.cat(predictions)

        pred_np = predictions.cpu().numpy()
        target_np = y_test.cpu().numpy()

        raw_preds = self.scaler.inverse_transform_predictions(
            pred_np
        )

        raw_targets = self.scaler.inverse_transform_predictions(
            target_np
        )

        rmse = np.sqrt(
            np.mean((raw_preds - raw_targets) ** 2)
        )

        hit_rate = (
            np.mean(
                np.sign(raw_preds)
                == np.sign(raw_targets)
            )
            * 100
        )

        residual_var = np.var(target_np - pred_np) + 1e-6

        std = np.sqrt(residual_var)

        normal_dist = torch.distributions.Normal(
            torch.tensor(pred_np),
            std,
        )

        nll = (
            -normal_dist.log_prob(
                torch.tensor(target_np)
            )
            .mean()
            .item()
        )

        #
        # Deterministic model.
        # No calibration estimate available.
        #
        ece = 0.0

        return {
            "RMSE": float(rmse),
            "Hit_Rate": float(hit_rate),
            "NLL": float(nll),
            "ECE": float(ece),
        }

    def _save_weights(self, path: Path) -> None:

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )

    def _load_weights(self, path: Path) -> None:

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
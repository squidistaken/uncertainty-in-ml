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

        # Homoscedastic aleatoric-noise std (in scaled space), fitted from the
        # residuals during evaluation and shared across all predictions.
        self.observation_noise: Optional[float] = None

        # Number of stochastic forward passes for MC-Dropout uncertainty at
        # inference.
        self.mc_samples: int = int(kwargs.get("mc_samples", 1))

        # Cache of the most recent per-point epistemic variance (MC-Dropout
        # sample variance), exposed for the error-vs-uncertainty diagnostic.
        self.last_epistemic_variance: Optional[torch.Tensor] = None

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
        mean, epistemic_var = self._predict(x)

        aleatoric_var = (
            self.observation_noise**2
            if self.observation_noise is not None
            else 1.0
        )
        std = torch.sqrt(epistemic_var + aleatoric_var)

        self.last_epistemic_variance = epistemic_var

        return torch.distributions.Normal(mean, std)

    def _predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the per-point mean and epistemic variance for a batch.

        Args:
            x (torch.Tensor): The input data sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The per-point mean and epistemic
                                               variance.
        """
        self.eval()
        x = x.to(DEVICE)

        if self.mc_samples > 1:
            self.lstm.train()

            with torch.no_grad():
                samples = torch.stack([self(x) for _ in range(self.mc_samples)])

            self.lstm.eval()

            return samples.mean(dim=0), samples.var(dim=0)

        with torch.no_grad():
            mean = self(x)

        return mean, torch.zeros_like(mean)

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
        fit_noise: bool = True,
    ) -> dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (DataLoader): The testing data features.
            y_test (torch.Tensor): The testing data true targets.
            fit_noise (bool): Whether to (re)fit the homoscedastic aleatoric
                              noise from this set's residuals. Set to False to
                              reuse a previously fitted noise (e.g. fitted on
                              the validation set) so the probabilistic metrics
                              on a held-out set are not computed with a variance
                              tuned to that same set. Defaults to True.

        Returns:
            dict[str, float]: A metric(s) indicating the performance.
        """
        self.eval()

        means = []
        epistemic_vars = []

        for batch_x, _, _ in x_test:
            batch_mean, batch_var = self._predict(batch_x)
            means.append(batch_mean)
            epistemic_vars.append(batch_var)

        predictions = torch.cat(means)
        epistemic_var = torch.cat(epistemic_vars)
        pred_np = predictions.cpu().numpy()
        epistemic_var_np = epistemic_var.cpu().numpy()
        target_np = y_test.cpu().numpy()
        raw_preds = self.scaler.inverse_transform_predictions(pred_np)
        raw_targets = self.scaler.inverse_transform_predictions(target_np)

        # Financial & Point Metrics:
        # 1. Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((raw_preds - raw_targets) ** 2))

        # 2. Hit Rate/Directional Accuracy
        hit_rate = np.mean(np.sign(raw_preds) == np.sign(raw_targets)) * 100

        # Probabilistic Metrics:
        # The predictive variance decomposes into a per-point epistemic term
        # and a homoscedastic aleatoric term fitted from the residuals.
        # The aleatoric variance is the residual variance with the mean
        # epistemic variance removed, so the two terms sum to the observed
        # error variance without double counting.
        # With MC disabled the epistemic term is zero and this recovers the
        # deterministic homoscedastic baseline.
        if fit_noise or self.observation_noise is None:
            residual_var = float(np.var(target_np - pred_np)) + 1e-6
            aleatoric_var = max(
                residual_var - float(epistemic_var_np.mean()), 1e-6
            )
            self.observation_noise = float(np.sqrt(aleatoric_var))

        aleatoric_var = self.observation_noise**2
        std_np = np.sqrt(aleatoric_var + epistemic_var_np)

        means_t = torch.tensor(pred_np)
        targets_t = torch.tensor(target_np)
        std_t = torch.tensor(std_np, dtype=means_t.dtype)

        # 1. Negative Log-Likelihood (NLL)
        #    Computed in scaled space, then shifted by log(scale) to express it
        #    in raw-return units.
        normal_dist = torch.distributions.Normal(means_t, std_t)
        nll_scaled = -normal_dist.log_prob(targets_t).mean().item()
        nll = nll_scaled + np.log(self.scaler.scale)

        # 2. Expected Calibration Error (ECE)
        #    We measure how well the predictive interval widths match the
        #    observed coverage across confidence levels.
        p_levels = torch.linspace(0.01, 0.99, 99)
        unit_normal = torch.distributions.Normal(0.0, 1.0)

        abs_errs = []

        for p in p_levels:
            z = unit_normal.icdf((1.0 + p) / 2.0)

            lower_bound = means_t - z * std_t
            upper_bound = means_t + z * std_t

            observed_coverage = (
                ((targets_t >= lower_bound) & (targets_t <= upper_bound))
                .float()
                .mean()
            )
            abs_errs.append(torch.abs(observed_coverage - p))

        ece = torch.tensor(abs_errs).mean().item()

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

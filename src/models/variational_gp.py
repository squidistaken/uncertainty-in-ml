from pathlib import Path
import torch
import gpytorch
import numpy as np
from torch.utils.data import DataLoader
from src.models.base_model import BaseModel
from src.constants import DEVICE
from typing import Any, Optional, cast, Sized
from src.utils.scaling import SPYScaler


class VariationalGP(gpytorch.models.ApproximateGP, BaseModel):
    """Variational Gaussian Process model class."""

    def __init__(self, inducing_points: torch.Tensor, **kwargs: Any) -> None:
        """Initialise the model.

        Args:
            inducing_points (torch.Tensor): The initial locations for the
                                            inducing points.
            **kwargs: The configuration.
        """
        if inducing_points.dim() > 2:
            # If the inducing points retain an explicit feature dimension, it
            # is flattened to match the flattened model inputs.
            inducing_points = inducing_points.view(inducing_points.size(0), -1)

        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            # This allows the GP to optimise the inducing point placement.
            learn_inducing_locations=True,
        )

        gpytorch.models.ApproximateGP.__init__(self, variational_strategy)
        BaseModel.__init__(self, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.to(DEVICE)
        self.optimizer = None
        self.scaler = SPYScaler()

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Compute the GP prior forward pass.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: A multivariate normal
                                                       distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        """Format sequences for the GP.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: A flattened tensor.
        """
        if x.dim() > 2:
            return x.view(x.size(0), -1)

        return x

    def forward_pass(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Predict the target for the input sequence.

        Args:
            x (torch.Tensor): The input data sequence.

        Returns:
            gpytorch.distributions.MultivariateNormal: A predictive
                                                       distribution containing
                                                       mean and variance.
        """
        self.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x = self._flatten_input(x).to(DEVICE)

            return self.likelihood(self(x))

    def backward_pass(
        self, x_train: DataLoader, y_train: Optional[Any] = None, **kwargs: Any
    ) -> float:
        """Train the model for one epoch over the provided data.

        Args:
            x_train (DataLoader): The training data sequences.
            y_train (Optional[Any], optional): The testing data true targets.
            **kwargs: The configuration.

        Returns:
            float: A metric(s) indicating the performance.
        """
        lr = kwargs.get("lr", 0.01)

        self.train()

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # The ELBO is built locally rather than cached as an attribute; it is
        # just a harmless (weird) workaround so we do not need to make multiple
        # classes.
        num_data = cast(Sized, x_train.dataset)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self, num_data=len(num_data)
        )

        epoch_loss = 0.0

        for batch_x, batch_y, _ in x_train:
            batch_x = self._flatten_input(batch_x).to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            self.optimizer.zero_grad()
            output = self(batch_x)

            loss_tensor = cast(torch.Tensor, mll(output, batch_y))
            loss = -loss_tensor
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(x_train)

    def evaluate(
        self, x_test: DataLoader, y_test: torch.Tensor, fit_noise: bool = True
    ) -> dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (DataLoader): The testing data features.
            y_test (torch.Tensor): The testing data true targets.
            fit_noise (bool): Whether to fit noise. Defaults to True.

        Returns:
            dict[str, float]: A metric(s) indicating the performance.
        """
        self.eval()

        all_means = []
        all_vars = []
        all_targets = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for batch_x, batch_y, _ in x_test:
                batch_x = self._flatten_input(batch_x).to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                predictive_dist = self.likelihood(self(batch_x))
                all_means.append(predictive_dist.mean)
                all_vars.append(predictive_dist.variance)
                all_targets.append(batch_y)

        means = torch.cat(all_means, dim=0)
        variances = torch.cat(all_vars, dim=0)
        stds = torch.sqrt(variances)
        targets = torch.cat(all_targets, dim=0)
        means_np = means.cpu().numpy()
        targets_np = targets.cpu().numpy()

        raw_means = self.scaler.inverse_transform_predictions(means_np)
        raw_targets = self.scaler.inverse_transform_predictions(targets_np)

        # Financial & Point Metrics:
        # 1. Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((raw_means - raw_targets) ** 2))

        # 2. Hit Rate/Directional Accuracy
        hit_rate = np.mean(np.sign(raw_means) == np.sign(raw_targets)) * 100

        # Probabilistic Metrics:
        # 1. Negative Log-Likelihood (NLL)
        scaled_normal_dist = torch.distributions.Normal(means, stds)
        nll_scaled = -scaled_normal_dist.log_prob(targets).mean().item()
        nll = nll_scaled + np.log(self.scaler.scale)

        # 2. Expected Calibration Error (ECE)
        p_levels = torch.linspace(0.01, 0.99, 99, device=DEVICE)

        unit_normal = torch.distributions.Normal(
            torch.tensor(0.0, device=DEVICE),
            torch.tensor(1.0, device=DEVICE),
        )

        abs_errs = []

        for p in p_levels:
            z = unit_normal.icdf((1.0 + p) / 2.0)

            lower_bound = means - z * stds
            upper_bound = means + z * stds

            observed_coverage = (
                ((targets >= lower_bound) & (targets <= upper_bound))
                .float()
                .mean()
            )
            abs_errs.append(torch.abs(observed_coverage - p))

        ece = torch.tensor(abs_errs).mean().item()

        metrics = {
            "RMSE": float(rmse),
            "Hit_Rate": float(hit_rate),
            "NLL": float(nll),
            "ECE": float(ece),
        }

        return metrics

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
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])

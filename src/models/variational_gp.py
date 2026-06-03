from pathlib import Path
import torch
import gpytorch
from torch.utils.data import DataLoader
from src.models.base_model import BaseModel
from src.constants import DEVICE
from typing import Any, Optional, cast, Sized


class SPYVariationalGP(gpytorch.models.ApproximateGP):
    """GPyTorch Variational GP architecture class for the S&P 500 dataset.

    This class is the mathematical core of the Guassian Process, defining the
    variational strategy, mean function, and covariance (kernel) function.
    """

    def __init__(self, inducing_points: torch.Tensor) -> None:
        """Initialise the class

        Args:
            inducing_points (torch.Tensor): The initial locations for the
                                            inducing points.
        """
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

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: A multivariate normal
                                                       distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGP(BaseModel):
    """
    Wrapper class for the Variational Gaussian Process model.
    """

    def __init__(self, inducing_points: torch.Tensor, **kwargs: Any) -> None:
        """Initialise the class.

        Args:
            inducing_points (torch.Tensor): The initial locations for the
                                            inducing points.
            **kwargs: The configuration.
        """
        super().__init__(**kwargs)

        if inducing_points.dim() > 2:
            # If the inducing points retain an explicit 1D feature dimension,
            # it is flattened.
            inducing_points = inducing_points.view(inducing_points.size(0), -1)

        self.model = SPYVariationalGP(inducing_points).to(DEVICE)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        """Format sequences for the GP."""
        if x.dim() > 2:
            return x.view(x.size(0), -1)

        return x

    def forward_pass(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Predict the target for the input sequence.

        Args:
            x (torch.Tensor): The input data sequence.

        Returns:
            gpytorch.distributions.MultivariateNormal: A predictive
                                                       distribution containing
                                                       mean and variance.
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x = self._flatten_input(x).to(DEVICE)

            predictions = self.likelihood(self.model(x))

        return predictions

    def backward_pass(
        self, x_train: DataLoader, y_train: Optional[Any] = None, **kwargs: Any
    ) -> list[float]:
        """
        Train the model on the provided data.

        Args:
            x_train (DataLoader): The training data sequences.
            y_train (Optional[Any], optional): The training targets. Defaults to None.
            **kwargs: The configuration.
        """
        epochs = kwargs.get("epochs", 50)
        lr = kwargs.get("lr", 0.01)

        self.model.train()
        self.likelihood.train()

        optimiser = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lr,
        )

        # Loss function: Negative Variational ELBO.
        num_data = cast(Sized, x_train.dataset)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=len(num_data)
        )

        loss_history = []

        for _ in range(epochs):
            epoch_loss = 0.0

            for batch_x, batch_y, _ in x_train:
                batch_x = self._flatten_input(batch_x).to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimiser.zero_grad()
                output = self.model(batch_x)

                loss_tensor = cast(torch.Tensor, mll(output, batch_y))
                loss = -loss_tensor
                loss.backward()

                optimiser.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(x_train)
            loss_history.append(avg_loss)

        return loss_history

    def evaluate(
        self, x_test: DataLoader, y_test: torch.Tensor
    ) -> dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (DataLoader): The testing data features.
            y_test (torch.Tensor): The testing data true targets.

        Returns:
            Any: A metric(s) indicating the performance.
        """
        self.model.eval()
        self.likelihood.eval()

        all_means = []
        all_vars = []
        all_targets = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for batch_x, batch_y, _ in x_test:
                batch_x = self._flatten_input(batch_x).to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                predictive_dist = self.likelihood(self.model(batch_x))
                all_means.append(predictive_dist.mean)
                all_vars.append(predictive_dist.variance)
                all_targets.append(batch_y)

        # We concatenate arrays for globally accurate and unbiased metric
        # calculations.
        means = torch.cat(all_means, dim=0)
        variances = torch.cat(all_vars, dim=0)
        stds = torch.sqrt(variances)
        targets = torch.cat(all_targets, dim=0)

        # Point Metrics:
        # 1. Mean Squared Error (MSE).
        # 2. Mean Absolute Percentage Error (MAPE).
        mse = torch.nn.functional.mse_loss(means, targets).item()
        mape = (
            torch.mean(torch.abs((means - targets) / (targets + 1e-8))).item()
            * 100
        )

        # Probabilistic Metrics:
        # 1. Negative Log-Likelihood (NLL).
        # 2. Expected Calibration Error (ECE).
        normal_dist = torch.distributions.Normal(means, stds)
        nll = -normal_dist.log_prob(targets).mean().item()
        p_levels = torch.linspace(0.01, 0.99, 99, device=DEVICE)

        # We setup a Guassian Unit to derive symmetric quantities.
        unit_normal = torch.distributions.Normal(
            torch.tensor([0.0], device=DEVICE),
            torch.tensor([1.0], device=DEVICE),
        )

        abs_errs = []

        for p in p_levels:
            z = unit_normal.icdf((1.0 + p) / 2.0)

            lower_bound = means - z * stds
            upper_bound = means + z * stds

            # This is the true fraction of targets falling within the prediction
            # interval.
            observed_coverage = (
                ((targets >= lower_bound) & (targets <= upper_bound))
                .float()
                .mean()
            )
            abs_errs.append(torch.abs(observed_coverage - p))

        ece = torch.tensor(abs_errs).mean().item()

        metrics = {"MSE": mse, "MAPE": mape, "NLL": nll, "ECE": ece}

        return metrics

    def _save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path (Path): The path to save the model weights to.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "likelihood_state_dict": self.likelihood.state_dict(),
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
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

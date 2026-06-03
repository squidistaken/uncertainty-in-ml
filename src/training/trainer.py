import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.constants import DEVICE, LOGGER, LOGS_DIR, RESULTS_DIR
from src.models.base_model import BaseModel
from src.utils.scaling import SPYScaler


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = DEVICE,
    ) -> None:
        """Initialise the Trainer.

        Args:
            model (BaseModel): The model.
            train_loader (DataLoader): The training set.
            val_loader (DataLoader): The validation set.
            test_loader (DataLoader): The test set.
            device (str): The device to run the training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Initialize the global unscaling utility
        self.scaler = SPYScaler()

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_metrics": [],
            "test_metrics": [],
        }

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = self.model.__class__.__name__
        self.tb_dir = (
            Path(LOGS_DIR) / "tensorboard" / f"{model_name}_{timestamp}"
        )
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def train(
        self,
        epochs: int = 100,
        lr: float = 0.001,
        patience: int = 10,
        monitor: Literal["MSE", "NLL", "ECE", "MAPE"] = "NLL",
    ) -> list[float]:
        """Execute the model training.

        Args:
            epochs (int): The number of epochs to train. Defaults to 100.
            lr (float): The learning rate. Defaults to 0.01.
            patience (Optional[int]): The number of epochs to wait for
                                      early stopping. Defaults to 10.
            monitor (Literal["MSE", "NLL", "ECE", "MAPE"]): The validation
                                                            metric to track for
                                                            early stopping.
                                                            Defaults to 'NLL'.

        Returns:
            list[float]: A history of training loss per epoch.
        """
        LOGGER.info(f"Starting training loop for {epochs} epochs...")
        if patience is not None:
            LOGGER.info(
                f"Early stopping enabled. Monitoring '{monitor}' with a patience of {patience} epochs."
            )

        loss_history = []
        best_score = float("inf")
        patience_counter = 0
        best_model_state = None
        best_likelihood_state = None

        for epoch in range(epochs):
            epoch_loss = self.model.backward_pass(
                x_train=self.train_loader, y_train=None, lr=lr
            )
            loss_history.append(epoch_loss)

            self.writer.add_scalar("Loss/Train", epoch_loss, epoch + 1)

            val_metrics = self.evaluate(use_test=False, step=epoch + 1)

            LOGGER.info(
                f"Epoch {epoch + 1:02d}/{epochs:02d} | "
                f"Train Loss: {epoch_loss:.6f} | "
                f"Val MSE: {val_metrics['MSE']:.6f} | "
                f"Val ECE: {val_metrics['ECE']:.6f}"
            )

            if patience is not None:
                current_score = val_metrics.get(monitor, float("inf"))
                if current_score < best_score:
                    best_score = current_score
                    patience_counter = 0

                    model_attr = getattr(self.model, "model", None)
                    likelihood_attr = getattr(self.model, "likelihood", None)

                    if isinstance(model_attr, torch.nn.Module) and isinstance(
                        likelihood_attr, torch.nn.Module
                    ):
                        best_model_state = copy.deepcopy(
                            model_attr.state_dict()
                        )
                        best_likelihood_state = copy.deepcopy(
                            likelihood_attr.state_dict()
                        )
                    elif isinstance(self.model, torch.nn.Module):
                        best_model_state = copy.deepcopy(
                            self.model.state_dict()
                        )
                    elif hasattr(self.model, "state_dict"):
                        state_dict_fn = getattr(self.model, "state_dict")
                        best_model_state = copy.deepcopy(state_dict_fn())
                else:
                    patience_counter += 1
                    LOGGER.info(
                        f"--> Validation {monitor} did not improve. "
                        f"Early stopping counter: {patience_counter}/{patience}"
                    )
                    if patience_counter >= patience:
                        LOGGER.info(
                            f"Early stopping triggered at epoch {epoch + 1}!"
                        )
                        break

        self.history["train_loss"] = loss_history

        if patience is not None and best_model_state is not None:
            model_attr = getattr(self.model, "model", None)
            likelihood_attr = getattr(self.model, "likelihood", None)

            if (
                isinstance(model_attr, torch.nn.Module)
                and isinstance(likelihood_attr, torch.nn.Module)
                and best_likelihood_state is not None
            ):
                model_attr.load_state_dict(best_model_state)
                likelihood_attr.load_state_dict(best_likelihood_state)
            elif isinstance(self.model, torch.nn.Module):
                self.model.load_state_dict(best_model_state)
            elif hasattr(self.model, "load_state_dict"):
                load_state_dict_fn = getattr(self.model, "load_state_dict")
                load_state_dict_fn(best_model_state)

            LOGGER.info(
                f"Restored best weights from validation epoch with lowest {monitor}: {best_score:.6f}"
            )

        LOGGER.info("Training complete.")

        return loss_history

    def evaluate(
        self, use_test: bool = False, step: Optional[int] = None
    ) -> dict[str, float]:
        """Evaluate the performance of the model.

        Args:
            use_test (bool): Whether to evaluate on the test set. Defaults to
                             False.
            step (Optional[int]): The current training step (epoch) for
                                  TensorBoard logging. Defaults to None.

        Returns:
            dict[str, float]: A dictionary containing computed point and
                              probabilistic metrics.
        """
        loader = self.test_loader if use_test else self.val_loader

        all_targets = []
        for _, batch_y, _ in loader:
            all_targets.append(batch_y)
        targets = torch.cat(all_targets, dim=0).to(self.device)

        metrics = self.model.evaluate(loader, targets)

        if step is None:
            step = len(self.history["train_loss"])

        if not use_test:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(
                    f"Metrics/Validation_{metric_name}", value, step
                )

        return metrics

    def get_predictions(self, loader: DataLoader) -> dict[str, np.ndarray]:
        """Get predictions, targets, uncertainties, and times from a
        DataLoader.

        Args:
            loader (DataLoader): The DataLoader to generate predictions for.

        Returns:
            dict[str, np.ndarray]: A predictions dictionary containing means,
                                   variances, targets, and time indices.
        """
        all_means = []
        all_vars = []
        all_targets = []
        all_times = []

        for batch_x, batch_y, batch_time in loader:
            predictive_dist = self.model.forward_pass(batch_x)

            all_means.append(predictive_dist.mean.cpu())
            all_vars.append(predictive_dist.variance.cpu())
            all_targets.append(batch_y.cpu())
            all_times.append(batch_time.cpu())

        return {
            "means": torch.cat(all_means, dim=0).numpy(),
            "variances": torch.cat(all_vars, dim=0).numpy(),
            "targets": torch.cat(all_targets, dim=0).numpy(),
            "time_idx": torch.cat(all_times, dim=0).numpy(),
        }

    def plot_history(self) -> None:
        """Plot training loss history with premium aesthetics and log to TensorBoard."""
        losses = self.history.get("train_loss", [])

        if not losses:
            LOGGER.warning("No training history to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        epochs = np.arange(1, len(losses) + 1)

        ax.plot(
            epochs,
            losses,
            marker="o",
            color="#4F70E5",
            label="Train Loss",
            linewidth=1.8,
            markersize=6,
        )

        ax.set_title("Variational GP Training Loss Convergence", fontsize=14)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Negative ELBO Loss", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=11, loc="upper right")
        fig.tight_layout()

        save_path = RESULTS_DIR / "loss_history.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

        self.writer.add_figure(
            "Plots/Training Loss", fig, global_step=len(losses)
        )

        LOGGER.info(f"Saved loss history plot to: {save_path}")
        plt.close(fig)

    def plot_predictions(
        self, results: dict[str, np.ndarray], max_points: int = 150
    ) -> None:
        """Plot observed returns against predicted returns with a shaded
        95% predictive confidence interval using premium aesthetics and
        unscaling utilities.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            max_points (int): The maximum chronologically last points to plot
                              for clarity.
        """
        means = results["means"][-max_points:]
        variances = results["variances"][-max_points:]
        targets = results["targets"][-max_points:]
        time_idx = results["time_idx"][-max_points:]

        stds = np.sqrt(variances)

        # Unscale variables to true financial return percentages
        raw_targets = self.scaler.inverse_transform_predictions(targets)
        raw_means = self.scaler.inverse_transform_predictions(means)
        raw_stds = self.scaler.inverse_transform_std(stds)

        lower_bound = raw_means - 1.96 * raw_stds
        upper_bound = raw_means + 1.96 * raw_stds

        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

        ax.plot(
            time_idx,
            raw_targets,
            color="#3D3D3D",
            label="True Log Returns",
            linewidth=1.2,
            alpha=0.9,
        )

        ax.plot(
            time_idx,
            raw_means,
            color="#DC143C",
            label="GP Mean Prediction",
            linewidth=1.5,
        )

        ax.fill_between(
            time_idx,
            lower_bound,
            upper_bound,
            color="#F7D6DB",
            alpha=0.6,
            label="95% Confidence Interval",
        )

        ax.set_xlabel("Time Index", fontsize=12)
        ax.set_ylabel("Raw Log Returns", fontsize=12)
        ax.set_title(
            "S&P 500 Out-of-Sample Predictions with Uncertainty Bounds (Unscaled)",
            fontsize=14,
        )
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = RESULTS_DIR / "predictions_with_uncertainty.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

        self.writer.add_figure(
            "Plots/Out-of-Sample Predictions",
            fig,
            global_step=len(self.history["train_loss"]),
        )

        LOGGER.info(f"Saved predictions visualisation to: {save_path}")
        plt.close(fig)

    def plot_calibration(self, results: dict[str, np.ndarray]) -> None:
        """Plot the regression calibration curve comparing observed coverage
        levels against expected coverage levels with premium aesthetics.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
        """
        means = results["means"]
        variances = results["variances"]
        targets = results["targets"]

        means_t = torch.tensor(means)
        stds_t = torch.tensor(np.sqrt(variances))
        targets_t = torch.tensor(targets)

        # We set up standard unit normal distribution to compute intervals.
        unit_normal = torch.distributions.Normal(0.0, 1.0)
        p_levels = torch.linspace(0.01, 0.99, 99)
        observed_coverage = []

        for p in p_levels:
            # Find symmetric boundaries corresponding to probability level p.
            z = unit_normal.icdf((1.0 + p) / 2.0)
            lower = means_t - z * stds_t
            upper = means_t + z * stds_t

            coverage = (
                ((targets_t >= lower) & (targets_t <= upper))
                .float()
                .mean()
                .item()
            )
            observed_coverage.append(coverage)

        ece = np.mean(np.abs(np.array(observed_coverage) - p_levels.numpy()))
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#2D2D2D",
            label="Perfect Calibration",
            linewidth=1.5,
        )

        ax.plot(
            p_levels.numpy(),
            observed_coverage,
            color="#4263EB",
            label="Model Calibration",
            linewidth=2.5,
        )

        ax.fill_between(
            p_levels.numpy(),
            p_levels.numpy(),
            observed_coverage,
            color="#E8EEFF",
            alpha=0.5,
        )

        ax.set_xlabel("Expected Probability Coverage", fontsize=12)
        ax.set_ylabel("Observed Coverage", fontsize=12)
        ax.set_title(
            f"Uncertainty Calibration Curve (ECE: {ece:.4f})", fontsize=14
        )
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.set_aspect("equal", "box")
        fig.tight_layout()

        save_path = RESULTS_DIR / "uncertainty_calibration.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

        self.writer.add_figure(
            "Plots/Uncertainty Calibration",
            fig,
            global_step=len(self.history["train_loss"]),
        )

        LOGGER.info(f"Saved uncertainty calibration curve to: {save_path}")
        plt.close(fig)

    def close(self) -> None:
        """Clean up resources."""
        self.writer.close()

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
        self.model_name = self.model.__class__.__name__
        self.tb_dir = (
            Path(LOGS_DIR) / "tensorboard" / f"{self.model_name}_{timestamp}"
        )
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def train(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        patience: int = 10,
        monitor: Literal["RMSE", "NLL", "ECE", "Hit_Rate"] = "NLL",
    ) -> list[float]:
        """Execute the model training.

        Args:
            epochs (int): The number of epochs to train. Defaults to 100.
            lr (float): The learning rate. Defaults to 0.01.
            patience (Optional[int]): The number of epochs to wait for
                                      early stopping. Defaults to 10.
            monitor (Literal["RMSE", "NLL", "ECE", "Hit_Rate"]): The validation
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

        # Adjust best starting score based on whether higher is better or lower
        # is better.
        higher_is_better = monitor in ["Hit_Rate"]
        best_score = float("-inf") if higher_is_better else float("inf")

        patience_counter = 0
        best_state = None

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
                f"Val NLL: {val_metrics['NLL']:.6f} | "
                f"Val ECE: {val_metrics['ECE']:.6f} | "
                f"Val RMSE: {val_metrics['RMSE']:.6f} | "
                f"Val Hit Rate: {val_metrics['Hit_Rate']:.2f}%"
            )

            if patience is not None:
                current_score = val_metrics.get(
                    monitor, float("-inf") if higher_is_better else float("inf")
                )

                # Check for improvement dynamically based on metric
                # directionality.
                is_improvement = (
                    (current_score > best_score)
                    if higher_is_better
                    else (current_score < best_score)
                )

                if is_improvement:
                    best_score = current_score
                    patience_counter = 0
                    best_state = self._capture_model_state()
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

        if patience is not None and best_state is not None:
            self._restore_model_state(best_state)

            LOGGER.info(
                f"Restored best weights from validation epoch with best {monitor}: {best_score:.6f}"
            )

        LOGGER.info("Training complete.")

        return loss_history

    def _capture_model_state(self) -> Optional[dict]:
        """Snapshot the trainable weights of the model.

        Returns:
            Optional[dict]: A copy of the model's weights
        """
        if isinstance(self.model, torch.nn.Module):
            return copy.deepcopy(self.model.state_dict())

        return None

    def _restore_model_state(self, state: dict) -> None:
        """Restore captured model weights.

        Args:
            state (dict): The captured model state dict.
        """
        if isinstance(self.model, torch.nn.Module):
            self.model.load_state_dict(state)

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

        ax.set_title(
            f"{self.model_name} Training Loss Convergence", fontsize=14
        )
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=11, loc="upper right")
        fig.tight_layout()

        save_path = RESULTS_DIR / f"{self.model_name}_loss_history.png"
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
        """Plot observed returns against predicted returns with 95% confidence
        intervals.

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
            label=f"{self.model_name} Mean Prediction",
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

        save_path = (
            RESULTS_DIR / f"{self.model_name}_predictions_with_uncertainty.png"
        )
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

        save_path = (
            RESULTS_DIR / f"{self.model_name}_uncertainty_calibration.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

        self.writer.add_figure(
            "Plots/Uncertainty Calibration",
            fig,
            global_step=len(self.history["train_loss"]),
        )

        LOGGER.info(f"Saved uncertainty calibration curve to: {save_path}")
        plt.close(fig)

    def plot_pit_histogram(
        self, results: dict[str, np.ndarray], bins: int = 20
    ) -> None:
        """Plot the Probability Integral Transform (PIT) histogram.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            bins (int): The number of histogram bins. Defaults to 20.
        """
        means = torch.tensor(results["means"])
        stds = torch.tensor(np.sqrt(results["variances"]))
        targets = torch.tensor(results["targets"])

        # Guard against degenerate (near-zero) predictive variances.
        stds = torch.clamp(stds, min=1e-12)

        predictive_dist = torch.distributions.Normal(means, stds)
        pit_values = predictive_dist.cdf(targets).numpy()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        ax.hist(
            pit_values,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            color="#4263EB",
            edgecolor="#2D2D2D",
            alpha=0.75,
            label="PIT Values",
        )

        ax.axhline(
            1.0,
            linestyle="--",
            color="#DC143C",
            linewidth=1.8,
            label="Ideal (Uniform)",
        )

        ax.set_xlabel("PIT Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"{self.model_name} PIT Histogram (Calibration Diagnostic)",
            fontsize=14,
        )
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="upper center", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = RESULTS_DIR / f"{self.model_name}_pit_histogram.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

        self.writer.add_figure(
            "Plots/PIT Histogram",
            fig,
            global_step=len(self.history["train_loss"]),
        )

        LOGGER.info(f"Saved PIT histogram to: {save_path}")
        plt.close(fig)

    def plot_error_vs_uncertainty(
        self, results: dict[str, np.ndarray], num_bins: int = 10
    ) -> None:
        """Plot absolute prediction error against predicted uncertainty.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            num_bins (int): The number of uncertainty bins for the trend line.
                            Defaults to 10.
        """
        stds = np.sqrt(results["variances"])

        raw_means = self.scaler.inverse_transform_predictions(results["means"])
        raw_targets = self.scaler.inverse_transform_predictions(
            results["targets"]
        )
        raw_stds = self.scaler.inverse_transform_std(stds)

        abs_errors = np.abs(raw_targets - raw_means)

        corr = (
            float(np.corrcoef(raw_stds, abs_errors)[0, 1])
            if raw_stds.std() > 0
            else float("nan")
        )

        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

        ax.scatter(
            raw_stds,
            abs_errors,
            color="#4263EB",
            alpha=0.35,
            s=18,
            edgecolor="none",
            label="Test Predictions",
        )

        # We overlay a binned-mean trend to summarise the relationship.
        if raw_stds.std() > 0:
            bin_edges = np.linspace(
                raw_stds.min(), raw_stds.max(), num_bins + 1
            )
            bin_idx = np.digitize(raw_stds, bin_edges[1:-1])
            bin_centers = []
            bin_means = []
            for b in range(num_bins):
                mask = bin_idx == b
                if np.any(mask):
                    bin_centers.append(raw_stds[mask].mean())
                    bin_means.append(abs_errors[mask].mean())

            ax.plot(
                bin_centers,
                bin_means,
                marker="o",
                color="#DC143C",
                linewidth=2.0,
                markersize=6,
                label="Binned Mean |Error|",
            )

        ax.set_xlabel("Predicted Std (Raw Log Returns)", fontsize=12)
        ax.set_ylabel("Absolute Error (Raw Log Returns)", fontsize=12)
        ax.set_title(
            f"{self.model_name} Error vs. Uncertainty (corr = {corr:.3f})",
            fontsize=14,
        )
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        save_path = RESULTS_DIR / f"{self.model_name}_error_vs_uncertainty.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

        self.writer.add_figure(
            "Plots/Error vs Uncertainty",
            fig,
            global_step=len(self.history["train_loss"]),
        )

        LOGGER.info(f"Saved error-vs-uncertainty plot to: {save_path}")
        plt.close(fig)

    def close(self) -> None:
        """Clean up resources."""
        self.writer.close()

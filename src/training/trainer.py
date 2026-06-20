import copy
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.constants import DEVICE, LOGGER, LOGS_DIR
from src.models.base_model import BaseModel
from src.utils.scaling import SPYScaler
from src.utils import plotting


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

        # train_loss holds per-epoch floats; val_metrics/test_metrics hold
        # per-epoch metric dictionaries, so the value type is a heterogeneous
        # list.
        self.history: dict[str, list] = {
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
        val_history = []
        higher_is_better = monitor in ["Hit_Rate"]
        best_score = float("-inf") if higher_is_better else float("inf")

        patience_counter = 0
        best_state = None
        best_epoch = None

        for epoch in range(epochs):
            epoch_loss = self.model.backward_pass(
                x_train=self.train_loader, y_train=None, lr=lr
            )
            loss_history.append(epoch_loss)

            self.writer.add_scalar("Loss/Train", epoch_loss, epoch + 1)

            val_metrics = self.evaluate(use_test=False, step=epoch + 1)
            val_history.append(val_metrics)

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
                    best_epoch = epoch + 1
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
        self.history["val_metrics"] = val_history
        self.monitor = monitor
        self.best_epoch = best_epoch

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
        self,
        use_test: bool = False,
        step: Optional[int] = None,
        fit_noise: bool = True,
    ) -> dict[str, float]:
        """Evaluate the performance of the model.

        Args:
            use_test (bool): Whether to evaluate on the test set. Defaults to
                             False.
            step (Optional[int]): The current training step (epoch) for
                                  TensorBoard logging. Defaults to None.
            fit_noise (bool): Whether to fit noise. Defaults to True.

        Returns:
            dict[str, float]: A dictionary containing computed point and
                              probabilistic metrics.
        """
        loader = self.test_loader if use_test else self.val_loader

        all_targets = []
        for _, batch_y, _ in loader:
            all_targets.append(batch_y)
        targets = torch.cat(all_targets, dim=0).to(self.device)

        metrics = self.model.evaluate(loader, targets, fit_noise=fit_noise)

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
        all_epistemic = []
        all_targets = []
        all_times = []

        for batch_x, batch_y, batch_time in loader:
            predictive_dist = self.model.forward_pass(batch_x)

            all_means.append(predictive_dist.mean.cpu())
            all_vars.append(predictive_dist.variance.cpu())
            epistemic = getattr(self.model, "last_epistemic_variance", None)
            if epistemic is None:
                epistemic = predictive_dist.variance
            all_epistemic.append(epistemic.cpu())

            all_targets.append(batch_y.cpu())
            all_times.append(batch_time.cpu())

        return {
            "means": torch.cat(all_means, dim=0).numpy(),
            "variances": torch.cat(all_vars, dim=0).numpy(),
            "epistemic_variances": torch.cat(all_epistemic, dim=0).numpy(),
            "targets": torch.cat(all_targets, dim=0).numpy(),
            "time_idx": torch.cat(all_times, dim=0).numpy(),
        }

    def _step(self) -> int:
        """The current global step (number of completed epochs) for logging."""
        return len(self.history["train_loss"])

    def plot_history(self) -> None:
        """Plot training loss history."""
        plotting.plot_history(self.model_name, self.history, self.writer)

    def plot_learning_curves(self) -> None:
        """Plot the training loss and validation metric per epoch."""
        plotting.plot_learning_curves(
            self.model_name,
            self.history,
            getattr(self, "monitor", "NLL"),
            getattr(self, "best_epoch", None),
            self.writer,
        )

    def plot_validation_metrics(self) -> None:
        """Plot every tracked validation metric over epochs."""
        plotting.plot_validation_metrics(
            self.model_name,
            self.history,
            getattr(self, "monitor", "NLL"),
            getattr(self, "best_epoch", None),
            self.writer,
        )

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
        plotting.plot_predictions(
            self.model_name,
            results,
            self.scaler,
            self.writer,
            self._step(),
            max_points,
        )

    def plot_price_predictions(
        self,
        results: dict[str, np.ndarray],
        prev_close: np.ndarray,
        actual_open: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
        max_points: int = 150,
    ) -> None:
        """Plot one-step-ahead price-level predictions with uncertainty.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            prev_close (np.ndarray): The actual close preceding each prediction,
                                     aligned element-wise with ``results``.
            actual_open (Optional[np.ndarray]): The actual open on each target
                                                day, overlaid for context.
            dates (Optional[np.ndarray]): The x-axis dates aligned with the
                                          predictions. Defaults to time indices.
            max_points (int): The maximum chronologically last points to plot.
        """
        plotting.plot_price_predictions(
            self.model_name,
            results,
            self.scaler,
            prev_close,
            actual_open=actual_open,
            dates=dates,
            writer=self.writer,
            step=self._step(),
            max_points=max_points,
        )

    def plot_calibration(self, results: dict[str, np.ndarray]) -> None:
        """Plot the regression calibration curve.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
        """
        plotting.plot_calibration(
            self.model_name, results, self.writer, self._step()
        )

    def plot_pit_histogram(
        self, results: dict[str, np.ndarray], bins: int = 20
    ) -> None:
        """Plot the Probability Integral Transform (PIT) histogram.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            bins (int): The number of histogram bins. Defaults to 20.
        """
        plotting.plot_pit_histogram(
            self.model_name, results, self.writer, self._step(), bins
        )

    def plot_error_vs_uncertainty(
        self, results: dict[str, np.ndarray], num_bins: int = 10
    ) -> None:
        """Plot absolute prediction error against predicted uncertainty.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            num_bins (int): The number of uncertainty bins for the trend line.
                            Defaults to 10.
        """
        plotting.plot_error_vs_uncertainty(
            self.model_name,
            results,
            self.scaler,
            self.writer,
            self._step(),
            num_bins,
        )

    def plot_disentangled_uncertainty(
        self, results: dict[str, np.ndarray], max_points: int = 150
    ) -> None:
        """Plot predictive, aleatoric, and epistemic uncertainty.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            max_points (int): The maximum chronologically last points to plot.
                              Defaults to 150.
        """
        plotting.plot_disentangled_uncertainty(
            self.model_name,
            results,
            self.scaler,
            self.writer,
            self._step(),
            max_points,
        )

    def plot_error_confidence_threshold(
        self, results: dict[str, np.ndarray], num_thresholds: int = 50
    ) -> None:
        """Plot RMSE vs confidence threshold.

        Args:
            results (dict[str, np.ndarray]): The prediction outputs.
            num_thresholds (int): Number of sweep points. Defaults to 50.
        """
        plotting.plot_error_confidence_threshold(
            self.model_name,
            results,
            self.scaler,
            self.writer,
            self._step(),
            num_thresholds,
        )

    def close(self) -> None:
        """Clean up resources."""
        self.writer.close()

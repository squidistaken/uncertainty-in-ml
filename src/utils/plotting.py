from pathlib import Path
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.constants import LOGGER, RESULTS_DIR
from src.utils.scaling import SPYScaler


def _finalize(
    fig: plt.Figure,
    save_path: Path,
    writer: Optional[SummaryWriter],
    tag: str,
    step: int,
    description: str,
) -> None:
    """Save a figure to disk, log it to TensorBoard, and close it.

    Args:
        fig (plt.Figure): The figure to persist.
        save_path (Path): The destination path for the PNG.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
        tag (str): The TensorBoard tag for the figure.
        step (int): The global step for the TensorBoard entry.
        description (str): A short description used in the log message.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)

    if writer is not None:
        writer.add_figure(tag, fig, global_step=step)

    LOGGER.info(f"Saved {description} to: {save_path}")
    plt.close(fig)


def plot_history(
    model_name: str,
    history: dict[str, list],
    writer: Optional[SummaryWriter] = None,
) -> None:
    """Plot training loss history.

    Args:
        model_name (str): The model name, used in titles and filenames.
        history (dict[str, list]): The training history.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
    """
    losses = history.get("train_loss", [])

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

    ax.set_title(f"{model_name} Training Loss Convergence", fontsize=14)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_loss_history.png",
        writer,
        "Plots/Training Loss",
        len(losses),
        "loss history plot",
    )


def plot_learning_curves(
    model_name: str,
    history: dict[str, list],
    monitor: str = "NLL",
    best_epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """Plot the training loss and validation metric per epoch.

    Args:
        model_name (str): The model name, used in titles and filenames.
        history (dict[str, list]): The training history.
        monitor (str): The monitored validation metric. Defaults to 'NLL'.
        best_epoch (Optional[int]): The restored early-stopping epoch, if any.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
    """
    losses = history.get("train_loss", [])
    val_metrics = history.get("val_metrics", [])

    if not losses or not val_metrics:
        LOGGER.warning("No training/validation history to plot.")
        return

    epochs = np.arange(1, len(losses) + 1)
    val_scores = [m[monitor] for m in val_metrics]

    fig, ax_train = plt.subplots(figsize=(10, 5), dpi=150)

    train_color = "#4F70E5"
    ax_train.plot(
        epochs,
        losses,
        marker="o",
        color=train_color,
        label="Train Loss",
        linewidth=1.8,
        markersize=5,
    )
    ax_train.set_xlabel("Epochs", fontsize=12)
    ax_train.set_ylabel("Training Loss", color=train_color, fontsize=12)
    ax_train.tick_params(axis="y", labelcolor=train_color)
    ax_train.grid(True, linestyle="--", alpha=0.5)

    val_color = "#DC143C"
    ax_val = ax_train.twinx()
    ax_val.plot(
        epochs,
        val_scores,
        marker="s",
        color=val_color,
        label=f"Validation {monitor}",
        linewidth=1.8,
        markersize=5,
    )
    ax_val.set_ylabel(f"Validation {monitor}", color=val_color, fontsize=12)
    ax_val.tick_params(axis="y", labelcolor=val_color)

    if best_epoch is not None:
        ax_train.axvline(
            best_epoch,
            color="#555555",
            linestyle=":",
            linewidth=1.5,
            label=f"Best epoch ({best_epoch})",
        )

    # Merge the legends from both axes into a single box.
    lines, labels = ax_train.get_legend_handles_labels()
    v_lines, v_labels = ax_val.get_legend_handles_labels()
    ax_train.legend(
        lines + v_lines, labels + v_labels, loc="upper right", fontsize=10
    )

    ax_train.set_title(f"{model_name} Learning Curves", fontsize=14)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_learning_curves.png",
        writer,
        "Plots/Learning Curves",
        len(losses),
        "learning-curves plot",
    )


def plot_validation_metrics(
    model_name: str,
    history: dict[str, list],
    monitor: str = "NLL",
    best_epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """Plot every tracked validation metric over epochs.

    Args:
        model_name (str): The model name, used in titles and filenames.
        history (dict[str, list]): The training history.
        monitor (str): The monitored validation metric. Defaults to 'NLL'.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
    """
    val_metrics = history.get("val_metrics", [])

    if not val_metrics:
        LOGGER.warning("No validation history to plot.")
        return

    epochs = np.arange(1, len(val_metrics) + 1)
    metric_names = ["RMSE", "NLL", "ECE", "Hit_Rate"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    for ax, name in zip(axes.flat, metric_names):
        scores = [m[name] for m in val_metrics]

        # Highlight the metric that drove early stopping.
        is_monitor = name == monitor
        ax.plot(
            epochs,
            scores,
            marker="o",
            color="#DC143C" if is_monitor else "#4F70E5",
            linewidth=1.8,
            markersize=5,
            label="Monitored" if is_monitor else None,
        )

        if best_epoch is not None:
            ax.axvline(
                best_epoch,
                color="#555555",
                linestyle=":",
                linewidth=1.5,
                label=f"Best epoch ({best_epoch})",
            )

        title = name.replace("_", " ")
        ax.set_title(
            f"Validation {title}" + (" (monitored)" if is_monitor else ""),
            fontsize=12,
        )
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        if best_epoch is not None:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"{model_name} Validation Metrics over Epochs", fontsize=14)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_validation_metrics.png",
        writer,
        "Plots/Validation Metrics",
        len(val_metrics),
        "validation-metrics plot",
    )


def plot_predictions(
    model_name: str,
    results: dict[str, np.ndarray],
    scaler: SPYScaler,
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    max_points: int = 150,
) -> None:
    """Plot observed returns against predicted returns with 95% confidence
    intervals.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs.
        scaler (SPYScaler): The unscaling utility.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
        max_points (int): The maximum chronologically last points to plot
                          for clarity. Defaults to 150.
    """
    means = results["means"][-max_points:]
    variances = results["variances"][-max_points:]
    targets = results["targets"][-max_points:]
    time_idx = results["time_idx"][-max_points:]

    stds = np.sqrt(variances)

    # Unscale variables to true financial return percentages.
    raw_targets = scaler.inverse_transform_predictions(targets)
    raw_means = scaler.inverse_transform_predictions(means)
    raw_stds = scaler.inverse_transform_std(stds)

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
        label=f"{model_name} Mean Prediction",
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

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_predictions_with_uncertainty.png",
        writer,
        "Plots/Out-of-Sample Predictions",
        step,
        "predictions visualisation",
    )


def plot_price_predictions(
    model_name: str,
    results: dict[str, np.ndarray],
    scaler: SPYScaler,
    prev_close: np.ndarray,
    actual_open: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    max_points: int = 150,
) -> None:
    """Plot one-step-ahead price-level predictions with 95% confidence bounds.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs (scaled means,
                                         variances, targets, and time indices).
        scaler (SPYScaler): The unscaling utility.
        prev_close (np.ndarray): The actual close of the trading day preceding
                                 each prediction, aligned element-wise with the
                                 ``results`` arrays.
        actual_open (Optional[np.ndarray]): The actual open price on each target
                                            day, overlaid for context, if given.
        dates (Optional[np.ndarray]): The x-axis dates aligned with the
                                      predictions. Defaults to the results' time
                                      indices.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
        max_points (int): The maximum chronologically last points to plot
                          for clarity. Defaults to 150.
    """
    means = results["means"][-max_points:]
    variances = results["variances"][-max_points:]
    targets = results["targets"][-max_points:]
    prev_close = np.asarray(prev_close)[-max_points:]

    x = (np.asarray(dates) if dates is not None else results["time_idx"])[
        -max_points:
    ]

    # Unscale the return forecast, then lift it into the price domain.
    raw_means = scaler.inverse_transform_predictions(means)
    raw_stds = scaler.inverse_transform_std(np.sqrt(variances))
    raw_targets = scaler.inverse_transform_predictions(targets)

    pred_close = prev_close * np.exp(raw_means)
    lower_bound = prev_close * np.exp(raw_means - 1.96 * raw_stds)
    upper_bound = prev_close * np.exp(raw_means + 1.96 * raw_stds)
    actual_close = prev_close * np.exp(raw_targets)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    ax.plot(
        x,
        actual_close,
        color="#3D3D3D",
        label="Actual Close",
        linewidth=1.2,
        alpha=0.9,
    )

    if actual_open is not None:
        ax.plot(
            x,
            np.asarray(actual_open)[-max_points:],
            color="#2E8B57",
            label="Actual Open",
            linewidth=1.0,
            alpha=0.6,
            linestyle="--",
        )

    ax.plot(
        x,
        pred_close,
        color="#DC143C",
        label=f"{model_name} Predicted Close",
        linewidth=1.5,
    )

    ax.fill_between(
        x,
        lower_bound,
        upper_bound,
        color="#F7D6DB",
        alpha=0.6,
        label="95% Confidence Interval",
    )

    ax.set_xlabel("Date" if dates is not None else "Time Index", fontsize=12)
    ax.set_ylabel("SPY Price ($)", fontsize=12)
    ax.set_title(
        f"{model_name} S&P 500 Next-Day Price Predictions "
        "with Uncertainty Bounds",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_price_predictions.png",
        writer,
        "Plots/Price Predictions",
        step,
        "price predictions visualisation",
    )


def plot_calibration(
    model_name: str,
    results: dict[str, np.ndarray],
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
) -> None:
    """Plot the regression calibration curve comparing observed coverage
    levels against expected coverage levels.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
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
            ((targets_t >= lower) & (targets_t <= upper)).float().mean().item()
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
    ax.set_title(f"Uncertainty Calibration Curve (ECE: {ece:.4f})", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_aspect("equal", "box")
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_uncertainty_calibration.png",
        writer,
        "Plots/Uncertainty Calibration",
        step,
        "uncertainty calibration curve",
    )


def plot_pit_histogram(
    model_name: str,
    results: dict[str, np.ndarray],
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    bins: int = 20,
) -> None:
    """Plot the Probability Integral Transform (PIT) histogram.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
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
        f"{model_name} PIT Histogram (Calibration Diagnostic)",
        fontsize=14,
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper center", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_pit_histogram.png",
        writer,
        "Plots/PIT Histogram",
        step,
        "PIT histogram",
    )


def plot_error_vs_uncertainty(
    model_name: str,
    results: dict[str, np.ndarray],
    scaler: SPYScaler,
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    num_bins: int = 10,
) -> None:
    """Plot absolute prediction error against predicted uncertainty.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs.
        scaler (SPYScaler): The unscaling utility.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
        num_bins (int): The number of uncertainty bins for the trend line.
                        Defaults to 10.
    """
    # Use the epistemic (model) uncertainty, which is what should track
    # error; for the GP this falls back to its total predictive variance.
    stds = np.sqrt(results["epistemic_variances"])

    raw_means = scaler.inverse_transform_predictions(results["means"])
    raw_targets = scaler.inverse_transform_predictions(results["targets"])
    raw_stds = scaler.inverse_transform_std(stds)

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
        bin_edges = np.linspace(raw_stds.min(), raw_stds.max(), num_bins + 1)
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
        f"{model_name} Error vs. Uncertainty (corr = {corr:.3f})",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_error_vs_uncertainty.png",
        writer,
        "Plots/Error vs Uncertainty",
        step,
        "error-vs-uncertainty plot",
    )


def plot_disentangled_uncertainty(
    model_name: str,
    results: dict[str, np.ndarray],
    scaler: SPYScaler,
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    max_points: int = 150,
) -> None:
    """Plot predictive, aleatoric, and epistemic uncertainty.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs.
        scaler (SPYScaler): The unscaling utility.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
                                          Defaults to None.
        step (int): The global step for the TensorBoard entry. Defaults to 0.
        max_points (int): The maximum chronologically last points to plot.
                          Defaults to 150.
    """
    means = results["means"][-max_points:]
    variances = results["variances"][-max_points:]
    epistemic_variances = results["epistemic_variances"][-max_points:]
    targets = results["targets"][-max_points:]
    time_idx = results["time_idx"][-max_points:]

    aleatoric_variances = np.maximum(variances - epistemic_variances, 0.0)

    raw_targets = scaler.inverse_transform_predictions(targets)
    raw_means = scaler.inverse_transform_predictions(means)
    total_stds = scaler.inverse_transform_std(np.sqrt(variances))
    aleatoric_stds = scaler.inverse_transform_std(np.sqrt(aleatoric_variances))
    epistemic_stds = scaler.inverse_transform_std(np.sqrt(epistemic_variances))

    panels = [
        (
            "(a) Predictive Uncertainty (Total)",
            total_stds,
            "#4F70E5",
            "#D0D9FA",
        ),
        (
            "(b) Aleatoric Uncertainty (Data Noise)",
            aleatoric_stds,
            "#2E8B57",
            "#C8EDD8",
        ),
        (
            "(c) Epistemic Uncertainty (Model)",
            epistemic_stds,
            "#DC143C",
            "#F7D6DB",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=150, sharex=True)

    for ax, (title, stds, line_color, fill_color) in zip(axes, panels):
        lower = raw_means - 1.96 * stds
        upper = raw_means + 1.96 * stds

        ax.plot(
            time_idx,
            raw_targets,
            color="#3D3D3D",
            linewidth=1.0,
            alpha=0.8,
            label="True Returns",
        )
        ax.plot(
            time_idx,
            raw_means,
            color=line_color,
            linewidth=1.5,
            label="Predicted Mean",
        )
        ax.fill_between(
            time_idx,
            lower,
            upper,
            color=fill_color,
            alpha=0.6,
            label="95% CI",
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Log Returns", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time Index", fontsize=12)
    fig.suptitle(f"{model_name} Uncertainty Disentanglement", fontsize=14)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_disentangled_uncertainty.png",
        writer,
        "Plots/Disentangled Uncertainty",
        step,
        "disentangled uncertainty plot",
    )


def plot_error_confidence_threshold(
    model_name: str,
    results: dict[str, np.ndarray],
    scaler: SPYScaler,
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    num_thresholds: int = 50,
) -> None:
    """Plot error vs confidence threshold.

    Args:
        model_name (str): The model name, used in titles and filenames.
        results (dict[str, np.ndarray]): The prediction outputs.
        scaler (SPYScaler): The unscaling utility.
        writer (Optional[SummaryWriter]): The TensorBoard writer, if any.
        step (int): The global step for the TensorBoard entry.
        num_thresholds (int): Number of sweep points. Defaults to 50.
    """
    stds = np.sqrt(results["epistemic_variances"])
    raw_means = scaler.inverse_transform_predictions(results["means"])
    raw_targets = scaler.inverse_transform_predictions(results["targets"])
    raw_stds = scaler.inverse_transform_std(stds)

    abs_errors = np.abs(raw_targets - raw_means)

    # Sort ascending by uncertainty → most confident predictions first.
    sorted_idx = np.argsort(raw_stds)
    sorted_errors = abs_errors[sorted_idx]

    fractions = np.linspace(0.1, 1.0, num_thresholds)
    rmse_values = []
    for frac in fractions:
        n_keep = max(1, int(len(sorted_errors) * frac))
        kept = sorted_errors[:n_keep]
        rmse_values.append(float(np.sqrt(np.mean(kept**2))))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(
        fractions * 100,
        rmse_values,
        color="#4263EB",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="RMSE on kept predictions",
    )

    ax.set_xlabel("% Predictions Kept (most confident first)", fontsize=12)
    ax.set_ylabel("RMSE (Raw Log Returns)", fontsize=12)
    ax.set_title(f"{model_name} Error vs. Confidence Threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    _finalize(
        fig,
        RESULTS_DIR / f"{model_name}_error_confidence_threshold.png",
        writer,
        "Plots/Error vs Confidence Threshold",
        step,
        "error-confidence threshold sweep",
    )

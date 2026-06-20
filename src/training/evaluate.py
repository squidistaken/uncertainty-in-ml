import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.constants import LOGGER, DEVICE, RESULTS_DIR, MODELS_DIR, SEED
from src.data.dataset import SPYDataset
from src.features.spy_preprocessor import load_prices_by_time_index
from src.models.variational_gp import VariationalGP
from src.models.lstm import LSTM
from src.training.trainer import Trainer
from src.utils.training import set_seeds


def _rebuild_lstm(path: Path, mc_samples: int) -> LSTM:
    """Reconstruct the LSTM network from the saved checkpoint.

    Args:
        path (Path): Path to the saved .pt checkpoint.
        mc_samples (int): The number of MC-Dropout forward passes at inference.

    Returns:
        LSTM: A model with weights loaded, ready for evaluation.
    """
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    sd = checkpoint["model_state_dict"]

    ih = sd["lstm.weight_ih_l0"]
    input_size = ih.shape[1]
    hidden_size = ih.shape[0] // 4
    num_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l"))

    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        mc_samples=mc_samples,
    )
    model._load_weights(path)
    return model


def _rebuild_vgp(path: Path) -> VariationalGP:
    """Reconstruct the VariationalGP model from a saved checkpoint.

    Args:
        path (Path): Path to the saved .pt checkpoint.

    Returns:
        VariationalGP: A model with weights loaded, ready for evaluation.
    """
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    sd = checkpoint["model_state_dict"]

    ip_shape = sd["variational_strategy.inducing_points"].shape
    dummy_ip = torch.zeros(ip_shape, device=DEVICE)

    model = VariationalGP(inducing_points=dummy_ip)
    model._load_weights(path)
    return model


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "vgp", "variational_gp"],
        help="The model to evaluate.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="[LSTM only] The number of MC-Dropout forward passes. Defaults to 50.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The batch size for evaluation. Defaults to 64.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"The deterministic seed. Defaults to {SEED}.",
    )

    args = parser.parse_args()
    set_seeds(args.seed)

    try:
        val_dataset = SPYDataset(split="val", device=DEVICE)
        test_dataset = SPYDataset(split="test", device=DEVICE)
    except FileNotFoundError as e:
        LOGGER.error(f"Failed to load dataset: {e}")
        raise

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_key = args.model if args.model != "variational_gp" else "vgp"

    if model_key == "vgp":
        checkpoint_path = MODELS_DIR / "VariationalGP.pt"
        LOGGER.info(f"Loading VariationalGP from {checkpoint_path}...")
        model = _rebuild_vgp(checkpoint_path)
        model_name = "VariationalGP"
    else:
        checkpoint_path = MODELS_DIR / "LSTM.pt"
        LOGGER.info(f"Loading LSTM from {checkpoint_path}...")
        model = _rebuild_lstm(checkpoint_path, mc_samples=args.mc_samples)
        model_name = "LSTM"

    # Trainer is used purely for its evaluation and plotting helpers and the
    # train_loader is unused so we pass the val_loader as a placeholder.
    trainer = Trainer(
        model=model,
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
    )

    LOGGER.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(use_test=False, fit_noise=True)

    LOGGER.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(use_test=True, fit_noise=False)

    LOGGER.info("Generating diagnostic plots...")
    test_results = trainer.get_predictions(test_loader)

    trainer.plot_predictions(test_results)
    trainer.plot_calibration(test_results)
    trainer.plot_pit_histogram(test_results)

    has_epistemic = model_key == "vgp" or (
        model_key == "lstm" and args.mc_samples > 1
    )
    if has_epistemic:
        trainer.plot_error_vs_uncertainty(test_results)
        trainer.plot_disentangled_uncertainty(test_results)
        trainer.plot_error_confidence_threshold(test_results)

    try:
        prices = load_prices_by_time_index()
        t_idx = test_results["time_idx"].astype(int)
        trainer.plot_price_predictions(
            test_results,
            prev_close=prices["Close"].reindex(t_idx - 1).to_numpy(),
            actual_open=prices["Open"].reindex(t_idx).to_numpy(),
            dates=prices["Date"].reindex(t_idx).to_numpy(),
        )
    except FileNotFoundError:
        LOGGER.warning("Raw price data not found; skipping price-level plot.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    report_data = {
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(report_data, f, indent=4)

    LOGGER.info(f"Saved evaluation report to: {metrics_path}")

    trainer.close()
    LOGGER.info("Evaluation complete.")


if __name__ == "__main__":
    main()

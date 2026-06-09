import argparse
import json
from torch.utils.data import DataLoader
from src.constants import LOGGER, DEVICE, RESULTS_DIR, MODELS_DIR, SEED, DEBUG
from src.data.dataset import SPYDataset
from src.features.spy_preprocessor import load_prices_by_time_index
from src.models.variational_gp import VariationalGP
from src.models.lstm import LSTM
from src.training.trainer import Trainer
from src.utils.training import set_seeds, select_inducing_points
from src.utils import ood, plotting


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "vgp", "variational_gp"],
        help="The model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to train. Defaults to 100.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The initial learning rate. Defaults to 0.01.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The training batch size. Defaults to 64.",
    )
    parser.add_argument(
        "--num-inducing",
        type=int,
        default=100,
        help="[Exclusive for Variational GP] The number of inducing points. Defaults to 100.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="[Exclusive for LSTM] The size of the LSTM hidden state. Defaults to 64.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="[Exclusive for LSTM] The number of stacked LSTM layers. Defaults to 2.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="[Exclusive for LSTM] The dropout probability between stacked LSTM layers. Defaults to 0.2.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="[Exclusive for LSTM] The number of MC-Dropout forward passes used "
        "to estimate predictive uncertainty at evaluation. Set to 1 for a "
        "deterministic (homoscedastic) model. Defaults to 50.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="The number of epochs to wait for early stopping. Defaults to 10.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="NLL",
        choices=["RMSE", "NLL", "ECE", "Hit_Rate"],
        help="The validation metric to track for early stopping. Defaults to 'NLL'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"The deterministic seed value to apply. Defaults to {SEED}",
    )

    args = parser.parse_args()

    if DEBUG:
        LOGGER.debug(f"Parsed arguments: {vars(args)}")

    set_seeds(args.seed)

    try:
        LOGGER.info("Loading S&P 500 Dataset partitions...")
        train_dataset = SPYDataset(split="train", device=DEVICE)
        val_dataset = SPYDataset(split="val", device=DEVICE)
        test_dataset = SPYDataset(split="test", device=DEVICE)
    except FileNotFoundError as e:
        LOGGER.error(f"Failed to load dataset partitions: {e}")
        LOGGER.info(
            "Please make sure you have downloaded and preprocessed the data first!"
        )

        raise FileNotFoundError("Dataset partitions not found.")

    # To stabilise the GP ELBO gradients, we use drop_last.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    LOGGER.info(f"Initialising model '{args.model}'...")

    if args.model in ["vgp", "variational_gp"]:
        inducing_points = select_inducing_points(
            train_dataset.X, args.num_inducing, seed=args.seed
        )
        model = VariationalGP(
            inducing_points=inducing_points,
            epochs=args.epochs,
            lr=args.lr,
            num_inducing=args.num_inducing,
        )
    elif args.model == "lstm":
        input_size = train_dataset.X.shape[-1]

        model = LSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
        )
    else:
        raise ValueError(
            f"Model architecture '{args.model}' is currently unsupported."
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
    )

    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        monitor=args.monitor,
    )

    LOGGER.info("Plotting training loss history...")
    trainer.plot_history()
    trainer.plot_learning_curves()
    trainer.plot_validation_metrics()

    if args.model == "lstm" and args.mc_samples > 1:
        LOGGER.info(
            f"Enabling MC-Dropout with {args.mc_samples} samples for "
            "evaluation."
        )
        model.mc_samples = args.mc_samples

    val_metrics = trainer.evaluate(use_test=False)
    test_metrics = trainer.evaluate(use_test=True, fit_noise=False)

    LOGGER.info(
        "Generating predictions and diagnostic plots on out-of-sample test data..."
    )
    test_results = trainer.get_predictions(test_loader)
    trainer.plot_predictions(test_results)
    trainer.plot_calibration(test_results)
    trainer.plot_pit_histogram(test_results)

    # Error-vs-uncertainty needs per-point uncertainty spread.
    if args.model in ["vgp", "variational_gp"] or (
        args.model == "lstm" and args.mc_samples > 1
    ):
        trainer.plot_error_vs_uncertainty(test_results)

    # Lift the return forecasts back into the price domain (P_t = P_{t-1} *
    # exp(r_t)), anchored on the actual previous close for each test point.
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
        LOGGER.warning(
            "Raw price data not found; skipping price-level prediction plot."
        )

    # Out-of-distribution diagnostics: does epistemic uncertainty grow on
    # inputs far from the training distribution?
    try:
        model_name = model.__class__.__name__
        scales = [1.0, 2.0, 3.0, 5.0, 8.0]
        sweep = ood.uncertainty_vs_severity(model, test_dataset.X, scales)
        report = ood.separability(model, test_dataset.X, ood_factor=scales[-1])
        plotting.plot_ood_severity(
            {model_name: sweep},
            scales,
            writer=trainer.writer,
            prefix=f"{model_name}_",
        )
        plotting.plot_ood_separability(
            {model_name: report},
            writer=trainer.writer,
            prefix=f"{model_name}_",
        )
    except Exception as e:
        LOGGER.warning(f"Skipping OOD diagnostics: {e}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        RESULTS_DIR / f"{model.__class__.__name__}_evaluation_report.json"
    )

    configuration = {
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(DEVICE),
    }

    if args.model in ["vgp", "variational_gp"]:
        configuration["num_inducing_points"] = args.num_inducing
    elif args.model == "lstm":
        configuration["hidden_size"] = args.hidden_size
        configuration["num_layers"] = args.num_layers
        configuration["dropout"] = args.dropout
        configuration["mc_samples"] = args.mc_samples

    report_data = {
        "configuration": configuration,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(report_data, f, indent=4)

    LOGGER.info(f"Saved complete evaluation report JSON to: {metrics_path}")

    trainer.close()

    LOGGER.info("Training pipeline successfully completed!")


if __name__ == "__main__":
    main()

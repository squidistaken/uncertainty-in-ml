import argparse
import json
from typing import Any, Callable
import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import GPSampler, TPESampler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset, TensorDataset
from src.constants import DEVICE, LOGGER, RESULTS_DIR, SEED, DEBUG
from src.data.dataset import SPYDataset
from src.models.base_model import BaseModel
from src.models.lstm import LSTM
from src.models.variational_gp import VariationalGP
from src.utils.training import select_inducing_points, set_seeds

# A larger hit rate value indicates a better model.
HIGHER_IS_BETTER = {"Hit_Rate"}
EVAL_BATCH_SIZE = 256


def build_combined_dataset() -> tuple[TensorDataset, torch.Tensor]:
    """
    Build the train and validation splits into one chronological set to
    yield a single time-ordered dataset for chronological cross-validation.

    Returns:
        tuple[TensorDataset, torch.Tensor]: A tuple containing the combined
                                            dataset and combined (unflattened)
                                            feature tensor.
    """
    train = SPYDataset(split="train", device=DEVICE)
    val = SPYDataset(split="val", device=DEVICE)

    combined_X = torch.cat([train.X, val.X], dim=0)
    combined_y = torch.cat([train.y, val.y], dim=0)
    combined_time = torch.cat([train.time_idx, val.time_idx], dim=0)

    LOGGER.info(
        f"Built combined CV pool of {combined_X.size(0)} samples "
        f"({train.X.size(0)} train + {val.X.size(0)} val)."
    )

    dataset = TensorDataset(combined_X, combined_y, combined_time)

    return dataset, combined_X


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Sample a hyperparameter configuration for the given model.

    Args:
        trial (optuna.Trial): The active Optuna trial.
        model_name (str): The model name.

    Returns:
        dict[str, Any]: A hyperparameter(s).
    """
    params: dict[str, Any] = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }

    if model_name == "lstm":
        params["hidden_size"] = trial.suggest_categorical(
            "hidden_size", [32, 64, 128]
        )
        params["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    elif model_name == "variational_gp":
        params["num_inducing"] = trial.suggest_int(
            "num_inducing", 50, 300, step=25
        )

    return params


def build_model(
    model_name: str,
    params: dict[str, Any],
    train_features: torch.Tensor,
    input_size: int,
    seed: int,
) -> BaseModel:
    """Instantiate a model for a single cross-validation fold.

    Args:
        model_name (str): The model name.
        params (dict[str, Any]): The hyperparameter(s).
        train_features (torch.Tensor): The fold's training feature tensor.
        input_size (int): The number of input features per time step.
        seed (int): The seed used for inducing point selection.

    Returns:
        BaseModel: An initialised model.
    """
    if model_name == "lstm":
        return LSTM(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            lr=params["lr"],
        )

    # The Variational GP requires at most as many inducing points as there are
    # training samples in the fold.
    num_inducing = min(params["num_inducing"], train_features.size(0))
    inducing_points = select_inducing_points(
        train_features, num_inducing, seed=seed
    )

    return VariationalGP(
        inducing_points=inducing_points,
        lr=params["lr"],
        num_inducing=num_inducing,
    )


def _gather_targets(loader: DataLoader) -> torch.Tensor:
    """Collect all targets from an unshuffled loader in order.

    Args:
        loader (DataLoader): The data loader to collect targets.

    Returns:
        torch.Tensor: A tensor of targets in the active device.
    """
    targets = [batch_y for _, batch_y, _ in loader]
    return torch.cat(targets, dim=0).to(DEVICE)


def train_and_score(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    monitor: str,
    patience: int,
) -> tuple[float, dict[str, float]]:
    """Train a model on one fold and return its best validation score.

    Args:
        model (BaseModel): The model to train.
        train_loader (DataLoader): The fold's training loader.
        val_loader (DataLoader): The fold's validation loader.
        epochs (int): The maximum number of epochs.
        lr (float): The learning rate.
        monitor (str): The validation metric to optimise.
        patience (int): The early-stopping patience in epochs.

    Returns:
        tuple[float, dict[str, float]]: A tuple containing the best monitored
                                        validation score observed during
                                        training, and the full set of metrics
                                        recorded at the same epoch.
    """
    higher_is_better = monitor in HIGHER_IS_BETTER
    best_score = float("-inf") if higher_is_better else float("inf")
    best_metrics: dict[str, float] = {}
    patience_counter = 0

    val_targets = _gather_targets(val_loader)

    for _ in range(epochs):
        model.backward_pass(x_train=train_loader, y_train=None, lr=lr)
        metrics = model.evaluate(val_loader, val_targets)
        score = metrics[monitor]

        is_improvement = (
            score > best_score if higher_is_better else score < best_score
        )

        if is_improvement:
            best_score = score
            best_metrics = metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_score, best_metrics


def make_objective(
    model_name: str,
    dataset: TensorDataset,
    features: torch.Tensor,
    n_splits: int,
    epochs: int,
    patience: int,
    monitor: str,
    seed: int,
) -> Callable[[optuna.Trial], float]:
    """Make the Optuna objective closure for the given configuration.

    Args:
        model_name (str): The model name.
        dataset (TensorDataset): The combined chronological CV pool.
        features (torch.Tensor): The combined feature tensor for inducing
                                 points.
        n_splits (int): The number of expanding-window CV folds.
        epochs (int): The maximum training epochs per fold.
        patience (int): The early-stopping patience per fold.
        monitor (str): The validation metric to optimise.
        seed (int): The base seed for reproducibility.

    Returns:
        Callable[[optuna.Trial], float]: An objective function.
    """
    input_size = features.shape[-1]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_indices = list(splitter.split(np.arange(len(dataset))))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)

        fold_scores: list[float] = []
        fold_metrics: list[dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            # Re-seed per fold so model initialisation is deterministic given
            # the trial, while remaining independent across folds.
            set_seeds(seed + fold)

            train_loader = DataLoader(
                Subset(dataset, train_idx.tolist()),
                batch_size=params["batch_size"],
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx.tolist()),
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
            )

            model = build_model(
                model_name,
                params,
                features[train_idx],
                input_size,
                seed + fold,
            )

            score, metrics = train_and_score(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=params["lr"],
                monitor=monitor,
                patience=patience,
            )
            fold_scores.append(score)
            fold_metrics.append(metrics)

            fold_profile = " | ".join(
                f"{name}: {value:.6f}" for name, value in metrics.items()
            )
            LOGGER.info(
                f"Trial {trial.number} | fold {fold + 1}/{n_splits} | "
                f"{fold_profile}"
            )
            trial.set_user_attr(f"fold_{fold}_metrics", metrics)
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        metric_names = fold_metrics[0].keys()
        avg_metrics = {
            name: float(np.mean([fm[name] for fm in fold_metrics]))
            for name in metric_names
        }
        trial.set_user_attr("metrics", avg_metrics)

        trial_profile = " | ".join(
            f"{name}: {value:.6f}" for name, value in avg_metrics.items()
        )
        LOGGER.info(
            f"Trial {trial.number} complete | CV-mean | {trial_profile}"
        )

        return float(np.mean(fold_scores))

    return objective


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Tune model hyperparameters via chronological "
        "cross-validation and Bayesian Optimisation."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "vgp", "variational_gp"],
        help="The model to tune.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="The number of Bayesian Optimisation trials. Defaults to 30.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="The number of expanding-window CV folds. Defaults to 5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="The maximum training epochs per fold. Defaults to 50.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="The early-stopping patience per fold. Defaults to 10.",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="NLL",
        choices=["RMSE", "NLL", "ECE", "Hit_Rate"],
        help="The validation metric to optimise. Defaults to 'NLL'.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="gp",
        choices=["gp", "tpe"],
        help="The Optuna sampler. Defaults to 'gp'.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="The wall-clock budget in seconds for the whole study.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"The deterministic seed value to apply. Defaults to {SEED}.",
    )

    args = parser.parse_args()

    if args.model == "vgp":
        args.model = "variational_gp"

    if DEBUG:
        LOGGER.debug(f"Parsed arguments: {vars(args)}")

    set_seeds(args.seed)

    try:
        dataset, features = build_combined_dataset()
    except FileNotFoundError as e:
        LOGGER.error(f"Failed to load dataset partitions: {e}")
        LOGGER.info(
            "Please make sure you have downloaded and preprocessed the data "
            "first!"
        )
        raise FileNotFoundError("Dataset partitions not found.")

    direction = "maximize" if args.monitor in HIGHER_IS_BETTER else "minimize"

    if args.sampler == "gp":
        sampler = GPSampler(seed=args.seed)
    else:
        sampler = TPESampler(seed=args.seed)

    LOGGER.info(
        f"Starting Optuna study | model='{args.model}' | "
        f"sampler='{args.sampler}' | direction='{direction}' | "
        f"optimising '{args.monitor}' over {args.n_splits} expanding folds | "
        f"{args.n_trials} trials."
    )

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=MedianPruner(n_warmup_steps=1),
        study_name=f"{args.model}_{args.monitor}",
    )

    objective = make_objective(
        model_name=args.model,
        dataset=dataset,
        features=features,
        n_splits=args.n_splits,
        epochs=args.epochs,
        patience=args.patience,
        monitor=args.monitor,
        seed=args.seed,
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    LOGGER.info("Hyperparameter search complete.")
    LOGGER.info(f"Best {args.monitor}: {study.best_value:.6f}")
    LOGGER.info(f"Best hyperparameters: {study.best_params}")

    best_metrics = study.best_trial.user_attrs.get("metrics", {})
    if best_metrics:
        profile = " | ".join(
            f"{name}: {value:.6f}" for name, value in best_metrics.items()
        )
        LOGGER.info(f"Best trial full metric profile (CV mean): {profile}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"{args.model}_tuning_report.json"

    report = {
        "configuration": {
            "model": args.model,
            "sampler": args.sampler,
            "monitor": args.monitor,
            "direction": direction,
            "n_trials": args.n_trials,
            "n_splits": args.n_splits,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(DEVICE),
        },
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs.get("metrics", {}),
        "n_complete_trials": len(
            study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        ),
        "n_pruned_trials": len(
            study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
        ),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    LOGGER.info(f"Saved tuning report to: {report_path}")


if __name__ == "__main__":
    main()

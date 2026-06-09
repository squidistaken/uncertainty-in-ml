# UML: S&P 500 Forecasting and Risk Quantification using Gaussian Processes

Repository for the Uncertainty in Machine Learning course (WBAI065-05) at the University of Groningen.

We forecast next-step S&P 500 (SPY) returns and quantify the predictive uncertainty around those forecasts. Two models are compared:

 * **Variational GP** (`variational_gp`/`vgp`): Stochastic variational Gaussian Process that produces calibrated predictive distributions.
 * **LSTM** (`lstm`): Deterministic recurrent baseline.

Both share a common training, evaluation, and diagnostic-plotting pipeline, and are scored with point metrics (RMSE, directional Hit Rate) and probabilistic metrics (NLL, ECE).

## Development

We use [uv](https://docs.astral.sh/uv/) for project management.
1. Clone the project.
2. Synchronise the project.

```bash
uv sync
```

3. Create a copy of [example.config.yaml](example.config.yaml) and rename it to `config.yaml`. Update the configuration, if desired.

## Command Line Interface (CLI)

The project can be run via a CLI, for convenient usage and testing. Every pipeline stage is available as a subcommand of a single entry point:

```bash
uv run main.py <command> [options]
```
 * `download`: Download the raw dataset from Kaggle.
 * `preprocess`: Preprocess the data and engineer features.
 * `train`: Train a model.
 * `tune`: Tune model hyperparameters via cross-validation.

Run `uv run main.py <command> --help` for command-specific options. Each stage can also be invoked directly as a module (e.g. `uv run -m src.training.train ...`); the options below apply to both forms.

### Downloading Data

#### Option 1: Download Script Using Kaggle API

```bash
uv run main.py download [--force]
```
 * `--force`: Forces a redownload of the data, in the event of missing or corrupted raw data. Defaults to `False`.

This requires a Kaggle API token, configured either via the `kaggle` section of `config.yaml` or a token file: https://www.kaggle.com/settings/api

#### Option 2: Manual Download

Dataset: https://www.kaggle.com/datasets/yousefeddin/s-and-p-500-stock-price-end-of-2024

 * Extract the archive and place the `.csv` file in the directory `<DATA_DIR>/raw/`

### Preprocessing and Feature Engineering

```bash
uv run main.py preprocess [--window-size] [--train-end-year] [--val-end-year]
```
 * `--window-size`: The size of the sliding window for sequence generation. Defaults to 21.
 * `--train-end-year`: The last year of the training split. Defaults to 2016.
 * `--val-end-year`: The last year of the validation split. Defaults to 2020.

## Training

```bash
uv run main.py train --model [--epochs] [--lr] [--batch-size] [--num-inducing] [--hidden-size] [--num-layers] [--dropout] [--patience] [--monitor] [--seed]
```
 * `--model`: The model: `["lstm", "vgp", "variational_gp"]`. Required.
 * `--epochs`: The number of epochs to train. Defaults to 100.
 * `--lr`: The initial learning rate. Defaults to 0.01.
 * `--batch-size`: The training batch size. Defaults to 64.
 * `--num-inducing`: The number of inducing points. Exclusive for Variational GP. Defaults to 100.
 * `--hidden-size`: The size of the LSTM hidden state. Exclusive for LSTM. Defaults to 64.
 * `--num-layers`: The number of stacked LSTM layers. Exclusive for LSTM. Defaults to 2.
 * `--dropout`: The dropout probability between stacked LSTM layers. Exclusive for LSTM. Defaults to 0.2.
 * `--mc-samples`: The number of MC-Dropout forward passes used to estimate predictive uncertainty at evaluation. Exclusive for LSTM. Set to 1 for a deterministic (homoscedastic) model. Defaults to 50.
 * `--patience`: The number of epochs to wait for early stopping. Defaults to 10.
 * `--monitor`: The validation metric to track for early stopping: `["RMSE", "NLL", "ECE", "Hit_Rate"]`. Defaults to "NLL".
 * `--seed`: The deterministic seed value to apply. Defaults to the `seed` set in `config.yaml`.

Training saves the trained model to `<MODELS_DIR>/`, an evaluation report (`<MODEL>_evaluation_report.json`) and a set of plots. to `<RESULTS_DIR>/`. 

### Recommended Hyperparameters

The following configurations were selected via cross-validated [hyperparameter tuning](#hyperparameter-tuning), optimising for validation NLL.

#### LSTM

```bash
uv run main.py train --model lstm \
  --lr 0.0028824672368677117 --batch-size 32 \
  --hidden-size 128 --num-layers 3 --dropout 0.24703173405578172 \
  --epochs 100 --patience 10 --monitor NLL
```

#### Variational GP

```bash
uv run main.py train --model vgp \
  --lr 0.014542214474659556 --batch-size 64 --num-inducing 100 \
  --epochs 100 --patience 10 --monitor NLL
```

## Hyperparameter Tuning

Tune hyperparameters via chronological (expanding-window) cross-validation and Bayesian Optimisation with [Optuna](https://optuna.org/).

```bash
uv run main.py tune --model [--n-trials] [--n-splits] [--epochs] [--patience] [--monitor] [--sampler] [--timeout] [--seed]
```
 * `--model`: The model to tune: `["lstm", "vgp", "variational_gp"]`. Required.
 * `--n-trials`: The number of Bayesian Optimisation trials. Defaults to 30.
 * `--n-splits`: The number of expanding-window CV folds. Defaults to 5.
 * `--epochs`: The maximum training epochs per fold. Defaults to 50.
 * `--patience`: The early-stopping patience per fold. Defaults to 10.
 * `--monitor`: The validation metric to optimise: `["RMSE", "NLL", "ECE", "Hit_Rate"]`. Defaults to "NLL".
 * `--sampler`: The Optuna sampler: `["gp", "tpe"]`. Defaults to "gp".
 * `--timeout`: The wall-clock budget in seconds for the whole study. Defaults to no limit.
 * `--seed`: The deterministic seed value to apply. Defaults to the `seed` set in `config.yaml`.

The best configuration and metrics are written to `<RESULTS_DIR>/<MODEL>_tuning_report.json`.

## Tensorboard Dashboard

```bash
tensorboard --logdir logs/tensorboard
```

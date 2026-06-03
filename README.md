# UML: S&P 500 Forecasting and Risk Quantification using Gaussian Processes

Repository for the Uncertainty in Machine Learning course (WBAI065-05) at the University of Groningen.

## Development

We use [uv](https://docs.astral.sh/uv/) for project management.
1. Clone the project.
2. Synchronise the project.

```bash
uv sync
```

3. Create a copy of [example.config.yaml](example.config.yaml) and rename it to config.yaml. Update the configuration, if desired.

## Command Line Interface (CLI)

The project can be run via a CLI, for convenient usage and testing.

### Downloading Data

#### Option 1: Download Script Using Kaggle API

```bash
uv run -m src.data.download_data [--force]
```
 * `--force`: Forces a redownload of the data, in the event of missing or corrupted raw data. Defaults to `False`. 

This requires a Kaggle API token to be set up: https://www.kaggle.com/settings/api

#### Option 2: Manual Download

Dataset: https://www.kaggle.com/datasets/yousefeddin/s-and-p-500-stock-price-end-of-2024

 * Extract the archive and place the `.csv` file in the directory `<DATA_DIR>/raw/`

### Preprocessing and Feature Engineering

```bash
uv run -m src.features.preprocess_data [--window-size] [--train-end-year] [--val-end-year]
```

## Training 

```bash
uv run -m src.train.train_model --model [--epochs] [--lr] [--batch-size] [--num-inducing] [--patience] [--monitor] [--seed]
```
 * `--model`: The model: `["lstm", "vgp", "variational_gp"]`. Required.
 * `--epochs`: The number of epochs to train. Defaults to 100.
 * `--lr`: The initial learning rate. Defaults to 0.001.
 * `--batch-size`: The training batch size. Defaults to 32.
 * `--num-inducing`: The number of inducing points. Exclusive for Variational GP. Defaults to 10.
 * `--patience`: The number of epochs to wait for early stopping. Defaults to 10.
 * `--monitor`: The validation metric to track for early stopping: `["RMSE", "NLL", "ECE", "Hit_Rate"]`. Defaults to "NLL."
 * `--seed`: The deterministic seed value to apply. Defaults to `SEED`.

## Tensorboard Dashboard

```bash
tensorboard --logdir logs/tensorboard
```


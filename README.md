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


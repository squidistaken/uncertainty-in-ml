# UML: S&P 500 Forecasting and Risk Quantification using Gaussian Processes

This project aims to predict the next-day log returns of the S&P 500 index and quantify market risk using Gaussian Processes. It is part of the Uncertainty in Machine Learning course at the University of Groningen.

## Development

1. Clone the project.
2. We use uv for package management.

```bash
uv sync
```
3. Clone [example.config.yaml][example.config.yaml] as `config.yaml`. Modify the configuration, if needed.
4. Set up a Kaggle API token on your device (for downloading data): https://kaggle.com/settings/api

## CLI

### Downloading Data
```bash
uv run -m src.data.download_data [--force]
```

### Preprocessing
```bash
uv run -m src.data.preprocess_data [--window-size] [--train-end-year] [--val-end-year]
```

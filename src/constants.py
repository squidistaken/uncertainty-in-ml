from pathlib import Path
from src.utils.logger import Logger
import yaml
import torch
import os

ROOT_DIR = Path(__file__).parent.parent

with open(ROOT_DIR / "config.yaml", "r") as config_file:
    data = yaml.load(config_file, Loader=yaml.SafeLoader)
    paths = data["paths"]

if "kaggle" in data:
    os.environ["KAGGLE_USERNAME"] = data["kaggle"]["username"]
    os.environ["KAGGLE_KEY"] = data["kaggle"]["key"]

DATA_DIR = ROOT_DIR / Path(paths["data"])
MODELS_DIR = ROOT_DIR / Path(paths["models"])
LOGS_DIR = ROOT_DIR / Path(paths["logs"])
RESULTS_DIR = ROOT_DIR / Path(paths["results"])
DEBUG = data["debug"]
LOGGER = Logger("uml")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
SEED = data["seed"]

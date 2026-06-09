import random

import numpy as np
import torch
from sklearn.cluster import KMeans

from src.constants import DEVICE, LOGGER, SEED


def set_seeds(seed: int = SEED) -> None:
    """Fix random seeds across environments for deterministic reproducibility.

    Args:
        seed (int): The seed value to apply. Defaults to the project SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    LOGGER.info(f"Fixed deterministic environment seeds using: {seed}")


def select_inducing_points(
    features: torch.Tensor, num_inducing: int, seed: int = SEED
) -> torch.Tensor:
    """Sample a subset of training features to initialise GP inducing points.

    Args:
        features (torch.Tensor): The training features.
        num_inducing (int): The number of inducing points to select.
        seed (int): The seed used for the K-Means initialisation. Defaults to
                    the project SEED.

    Returns:
        torch.Tensor: A tensor containing the initial inducing locations on the
                      active device.
    """
    X_flat = features.reshape(features.size(0), -1).cpu().numpy()

    kmeans = KMeans(n_clusters=num_inducing, random_state=seed, n_init=10)
    kmeans.fit(X_flat)

    inducing_points = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float32
    ).to(DEVICE)

    LOGGER.info(f"Selected {num_inducing} inducing points.")

    return inducing_points

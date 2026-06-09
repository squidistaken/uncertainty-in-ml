from typing import Union
from src.models.base_model import BaseModel
import gpytorch
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.constants import DEVICE


def epistemic_uncertainty(model: BaseModel, X: torch.Tensor) -> np.ndarray:
    """Calculate per-point epistemic (model) uncertainty in scaled space.

    Args:
        model (BaseModel): The model.
        X (torch.Tensor): The batch input sequences.

    Returns:
        np.ndarray: A per-point epistemic standard deviation.
    """
    X = X.to(DEVICE)

    if isinstance(model, gpytorch.models.ApproximateGP):
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            flat = X.reshape(X.size(0), -1) if X.dim() > 2 else X
            latent = model(flat)

            return latent.variance.sqrt().cpu().numpy()

    with torch.no_grad():
        model.forward_pass(X)

    epistemic = getattr(model, "last_epistemic_variance", None)
    if epistemic is None:
        raise ValueError(
            "Model exposes no epistemic uncertainty estimate "
            "(enable MC-Dropout with mc_samples > 1)."
        )
    return epistemic.sqrt().cpu().numpy()


def amplify(X: torch.Tensor, factor: float) -> torch.Tensor:
    """Synthesise an OOD batch by amplifying the input magnitude.

    Args:
        X (torch.Tensor): The in-distribution input batch.
        factor (float): The amplification factor.

    Returns:
        torch.Tensor: An amplified OOD batch.
    """
    return X * factor


def uncertainty_vs_severity(
    model: BaseModel, X: torch.Tensor, scales: list[float]
) -> list[float]:
    """Calculate mean epistemic uncertainty as OOD severity increases.

    Args:
        model (BaseModel): The model.
        X (torch.Tensor): The in-distribution input batch.
        scales (list[float]): The amplification factors to sweep.

    Returns:
        list[float]: A list of mean epistemic STD at each amplification factor.
    """
    return [
        float(epistemic_uncertainty(model, amplify(X, k)).mean())
        for k in scales
    ]


def separability(
    model: BaseModel, X_id: torch.Tensor, ood_factor: float = 8.0
) -> dict[str, Union[np.ndarray, float]]:
    """Score how well epistemic uncertainty separates ID from OOD inputs.

    Args:
        model (BaseModel): The model.
        X_id (torch.Tensor): The in-distribution input batch.
        ood_factor (float): The amplification factor defining the OOD batch.

    Returns:
        dict[str, Union[np.ndarray, float]]: A dictionary containing AUROC,
                                             ID STD and OOD STD, and OOD
                                             factor.
    """
    id_std = epistemic_uncertainty(model, X_id)
    ood_std = epistemic_uncertainty(model, amplify(X_id, ood_factor))

    scores = np.concatenate([id_std, ood_std])
    labels = np.concatenate([np.zeros(len(id_std)), np.ones(len(ood_std))])
    auroc = (
        float(roc_auc_score(labels, scores))
        if scores.std() > 0
        else float("nan")
    )

    return {
        "auroc": auroc,
        "id_std": id_std,
        "ood_std": ood_std,
        "ood_factor": ood_factor,
    }

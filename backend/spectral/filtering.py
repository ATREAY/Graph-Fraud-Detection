import numpy as np
import torch


def spectral_filter_features(
    x: torch.Tensor,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    mode: str = "low",
    cutoff: int = 10,
):
    """
    Spectral filtering of node features.

    x        : (N, F) node features
    eigvals  : (k,) eigenvalues
    eigvecs  : (N, k) eigenvectors
    mode     : "low" or "high"
    cutoff   : number of frequencies to keep
    """

    # Project features to spectral domain
    x_np = x.cpu().numpy()
    coeffs = eigvecs.T @ x_np   # (k, F)

    if mode == "low":
        mask = np.zeros_like(coeffs)
        mask[:cutoff, :] = coeffs[:cutoff, :]
    elif mode == "high":
        mask = np.zeros_like(coeffs)
        mask[cutoff:, :] = coeffs[cutoff:, :]
    else:
        raise ValueError("mode must be 'low' or 'high'")

    # Reconstruct features
    x_filtered = eigvecs @ mask
    return torch.tensor(x_filtered, dtype=torch.float)

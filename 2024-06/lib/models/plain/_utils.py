import torch


def minmax_scale(X: torch.Tensor, dim=0) -> torch.Tensor:
    X_min = X.min(dim=dim, keepdims=True).values
    X_max = X.max(dim=dim, keepdims=True).values
    X_std = (X - X_min) / (X_max - X_min)
    return X_std

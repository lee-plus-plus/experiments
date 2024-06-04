import numpy as np
import scipy.sparse as sp
import torch


def to_spmatrix(X, copy=False):
    if isinstance(X, np.ndarray):
        return sp.csc_matrix(X)
    if isinstance(X, sp.spmatrix):
        return X.copy().tocsc() if copy else X.tocsc()
    if isinstance(X, torch.Tensor):
        return sp.csc_matrix(X.cpu().detach().numpy())

    raise TypeError(f"unable to transform {type(X)} to scipy.spmatrix")


def to_ndarray(X, copy=False):
    if isinstance(X, sp.spmatrix):
        return X.toarray()
    if isinstance(X, np.ndarray):
        return X.copy() if copy else X
    if isinstance(X, torch.Tensor):
        return X.cpu().detach().numpy()

    raise TypeError(f"unable to transform ‘{type(X)}’ to numpy.ndarray")


def to_tensor(X, copy=False):
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X)
    if isinstance(X, torch.Tensor):
        return X.clone() if copy else X

    return to_tensor(to_ndarray(X, copy=False))

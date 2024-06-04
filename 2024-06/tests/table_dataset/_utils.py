import numpy as np
import torch
from torch.utils.data import Dataset


def assert_is_torch_Tensor(x, shape=None, dtype=None):
    assert type(x) is torch.Tensor

    if shape:
        assert x.shape == torch.Size(shape)
    if dtype:
        assert x.dtype == dtype


def assert_is_numpy_ndarray(x, shape=None, dtype=None):
    assert type(x) is np.ndarray

    if shape:
        assert x.shape == shape
    if dtype:
        assert x.dtype == dtype


def assert_is_equal(*elems):
    pivot = elems[0]
    for elem in elems:
        assert type(elem) == type(pivot) and elem == pivot


def assert_is_parital_labels(y_true, y_partial):
    assert not ((y_true == 1) & (y_partial == 0)).any()


def assert_is_valid_mll_dataset(dataset: Dataset) -> None:
    assert type(dataset.num_classes) is int
    assert type(dataset.num_samples) is int
    assert type(dataset.dim_features) is int
    assert type(dataset.category_name) is dict

    step = max(len(dataset) // 5, 1)  # sampling, for efficiency
    for index in range(0, len(dataset), step):
        feature, label = dataset[index]
        assert_is_torch_Tensor(feature, shape=[dataset.dim_features], dtype=torch.float32)
        assert_is_torch_Tensor(label, shape=[dataset.num_classes], dtype=torch.int32)


def assert_is_valid_pml_dataset(dataset: Dataset) -> None:
    assert type(dataset.num_classes) is int
    assert type(dataset.num_samples) is int
    assert type(dataset.dim_features) is int
    assert type(dataset.category_name) is dict

    step = max(len(dataset) // 5, 1)  # sampling, for efficiency
    for index in range(0, len(dataset), step):
        feature, label, partial_label = dataset[index]
        assert_is_torch_Tensor(feature, shape=[dataset.dim_features], dtype=torch.float32)
        assert_is_torch_Tensor(label, shape=[dataset.num_classes], dtype=torch.int32)
        assert_is_torch_Tensor(partial_label, shape=[dataset.num_classes], dtype=torch.int32)
        assert_is_parital_labels(label, partial_label)
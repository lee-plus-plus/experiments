import numpy as np
import torch
from random import choice, choices
from os.path import join, expanduser

from lib.table_dataset import (
    supported_datasets,
    build_dataset,
)
from ._utils import (
    assert_is_valid_mll_dataset,
    assert_is_valid_pml_dataset,
)


def test_factory():
    base = expanduser('~/table_dataset')

    assert len(supported_datasets()) > 0

    for name in choices(supported_datasets(), k=5):
        print(name)
        train_dataset, valid_dataset = build_dataset(name, base=base)
        assert_is_valid_mll_dataset(train_dataset)
        assert_is_valid_mll_dataset(valid_dataset)

        train_dataset, valid_dataset = build_dataset(name, base=base, 
            add_partial_noise=True, noise_rate=0.2)
        assert_is_valid_pml_dataset(train_dataset)
        assert_is_valid_pml_dataset(valid_dataset)



import numpy as np
import torch


def get_split_indices(num_samples, split_ratio, shuffle=True):
    indices = list(range(num_samples))
    split_ratio = (np.cumsum([0] + split_ratio) * num_samples).astype(int)

    if shuffle:
        np.random.shuffle(indices)

    splited_indices = [
        indices[st: ed]
        for st, ed in zip(split_ratio[:-1], split_ratio[1:])
    ]
    return splited_indices


def add_partial_noise(
    labels: torch.Tensor,
    noise_rate: float,
) -> torch.Tensor:
    '''add partial noise into labels with uniform probability
    '''
    noise = torch.rand(labels.shape) < noise_rate
    partial_labels = (labels.int() | noise.int()).int()
    return partial_labels

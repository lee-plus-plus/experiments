import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import os


def init_cuda_environment(seed=123, device='0'):
    cudnn.benchmark = True
    cudnn.deterministic = True

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = device


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

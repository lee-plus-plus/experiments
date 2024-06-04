import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import os
import sys

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


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

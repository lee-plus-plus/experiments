import numpy as np
import torch
from torchvision.transforms import Compose
from PIL import Image
from lib.transforms import (
    Normalize,
    UnNormalize,
    RandomCutout,
    imagenet_mean,
    imagenet_std,
)


def random_rgb_image(image_size=(224, 224)):
    shape = image_size + (3,)
    arr = np.random.randint(0, 256, shape)
    img = Image.fromarray(arr.astype('uint8')).convert('RGB')
    return img


def assert_is_PIL_Image(x, size=None):
    assert type(x) is Image.Image

    if size:
        assert x.size == size


def test_unnormalize():
    x = torch.rand((3, 224, 224))
    t = Compose([
        Normalize(),
        UnNormalize(),
    ])

    assert (torch.norm(t(x) - x)) / torch.norm(x) < 1e-5


def test_randomcutout():
    img = random_rgb_image(image_size=(224, 224))
    t = RandomCutout(cutout_factor=0.5)
    img_augmented = t(img)

    assert_is_PIL_Image(img_augmented, size=img.size)

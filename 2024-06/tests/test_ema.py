import torch
import torchvision
from lib.ema import ExponentialMovingAverage


def assert_is_tensor_equal(x, y):
    assert torch.equal(x, y)


def assert_is_tensor_not_equal(x, y):
    assert not torch.equal(x, y)


def get_params(model):
    return list(model.parameters())[0][0][0].detach()


def test_ema():
    # two randomly-weighted models
    model_src = torchvision.models.resnet18().requires_grad_(False)
    model_tgt = torchvision.models.resnet18().requires_grad_(False)

    assert_is_tensor_not_equal(get_params(model_src), get_params(model_tgt))

    model_ema = ExponentialMovingAverage(model_src, decay=0.9, device='cpu')
    assert_is_tensor_equal(get_params(model_ema), get_params(model_src))

    # assert model_ema.update_parameters(model_tgt) is same as letting
    # model_ema.params = decay * model_ema.params + (1 - decay) * model_tgt.params
    params = get_params(model_src).clone()
    for i in range(10):
        params = 0.9 * params + 0.1 * get_params(model_tgt)
        model_ema.update_parameters(model_tgt)

        assert_is_tensor_equal(get_params(model_ema), params)

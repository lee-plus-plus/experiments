import torch
import torchvision
import timm


'''support the following models:

- torchvision.models.resnet18
- torchvision.models.resnet34
- torchvision.models.resnet50
- torchvision.models.resnet101
- torchvision.models.resnet152
- timm.create_model('tresnet_m')
- timm.create_model('tresnet_l')
- timm.create_model('tresnet_xl')
'''


def build_cnn_backbone(name, *, pretrained=False):
    if 'tresnet' in name:
        assert name in ["tresnet_m", "tresnet_l", "tresnet_xl"]
        cnn_backbone = timm.create_model(name, pretrained=pretrained)

    elif 'resnet' in name:
        assert name in ["resnet18", "resnet34",
                        "resnet50", "resnet101", "resnet152"]

        weights = getattr(torchvision.models, f'ResNet{name[6:]}_Weights').DEFAULT
        cnn_backbone = getattr(torchvision.models, name)(
            weights=weights if pretrained else None)

    else:
        raise ValueError("unsupport cnn_backbone name")

    return cnn_backbone


def cnn_backbone_info(model):
    name = type(model).__name__

    if name == 'ResNet':
        layer_name_featuremap = 'layer4'
        dim_featuremap = model.layer4[-1].conv1.in_channels
        downsample_ratio = 32

        layer_name_fc = 'fc'
        dim_fc = model.fc.in_features
        dim_out = model.fc.out_features

    elif name == 'TResNet':
        layer_name_featuremap = 'body'
        dim_featuremap = model.body[-1][-1].conv1.in_channels
        downsample_ratio = 32

        layer_name_fc = 'head.fc'
        dim_fc = model.head.fc.in_features
        dim_out = model.head.fc.out_features

    return {
        "layer_name_featuremap": layer_name_featuremap,
        "dim_featuremap": dim_featuremap,
        "downsample_ratio": downsample_ratio,
        "layer_name_fc": layer_name_fc,
        "dim_fc": dim_fc,
        "dim_out": dim_out,
    }

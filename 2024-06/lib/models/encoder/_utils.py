import torch
from torchvision import transforms
from ...transforms import Normalize, UnNormalize, RandomCutout
from ...backbone import build_cnn_backbone, cnn_backbone_info, set_layer_by_name
from ...table_dataset.base import StorableTableMllDataset


def build_transform(name, *, image_size=224):
    if name == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Normalize(),
        ])
    else:
        raise ValueError(f'unsupported transform name ‘{name}’')


def build_encoder(backbone_name, pretrained=True):
    model = build_cnn_backbone(name=backbone_name, pretrained=pretrained)
    dim_embed = cnn_backbone_info(model)['dim_fc']
    fc_name = cnn_backbone_info(model)['layer_name_fc']
    set_layer_by_name(model, fc_name, torch.nn.Identity())
    return model, dim_embed


def save_dataset(features, labels, category_name, filename=None):
    dataset = StorableTableMllDataset(
        features=features, labels=labels,
        feature_names=None, label_names=category_name
    )

    if filename.endswith('.arff'):
        dataset.to_arff(filename)
    elif filename.endswith('.mat'):
        dataset.to_mat(filename)
    elif filename.endswith('.pickle'):
        dataset.to_pickle(filename)
    else:
        raise ValueError(f'unsupported file type ‘{filename}’')

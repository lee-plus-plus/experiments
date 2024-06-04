import os
from copy import deepcopy
import random
import time
from copy import deepcopy
import torch
from torch import nn
from ...backbone import (
    build_cnn_backbone,
    cnn_backbone_info,
    set_layer_by_name
)


def build_model(backbone_name, num_classes, *, pretrained=True):
    # load cnn backbone
    model = build_cnn_backbone(name=backbone_name, pretrained=pretrained)

    # replace fully-connected layer
    info = cnn_backbone_info(model)
    set_layer_by_name(
        model, info['layer_name_fc'],
        nn.Linear(info['dim_fc'], num_classes)
    )
    return model


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

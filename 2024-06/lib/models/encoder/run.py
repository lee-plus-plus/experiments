import os
import argparse
import torch
from datetime import datetime
from ...dataset import build_dataset, supported_multilabel_datasets
from ...utils import init_cuda_environment, str2bool
from ._utils import build_transform, build_encoder, save_dataset
from .main import collect_outputs


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='encoder')

    parser.add_argument('--dataset', default='voc2007', type=str,
                        choices=supported_multilabel_datasets())
    parser.add_argument('--divide', default='train', type=str,
                        choices=['train', 'test', 'undivided'])
    parser.add_argument('--backbone', default='resnet101', type=str,
                        choices=['resnet18', 'resnet50', 'resnet101', 'tresnet_l', 'tresnet_xl'])
    parser.add_argument('--output-filename', default='output.pickle', type=str, 
                        help='output embedding filename')


    parser.add_argument('--verbose', default=False, type=str2bool)
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device be used')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed for initialization')

    args = parser.parse_args(args)
    return args


def run(args=None):
    args = parse_args(args)
    print(__file__)
    print(args)

    init_cuda_environment(seed=args.seed, device=args.gpu)

    print(f'- dataset: {args.dataset} ({args.divide})')
    dataset = build_dataset(
        name=args.dataset, divide=args.divide, add_index=True, 
        transforms=[build_transform('none')],  # will be over-written
        flatten=True
    )

    # Setup model
    print(f'- model: {args.backbone}')
    model, dim_embedding = build_encoder(backbone_name=args.backbone, pretrained=True)\

    print(f'- num_samples: {dataset.num_samples}')
    print(f'- num_classes: {dataset.num_classes}')
    print(f'- dim_embedding: {dim_embedding}')

    # Actuall Training
    print('begin encoding...')
    embeddings = collect_outputs(model, dataset, verbose=args.verbose)

    print('begin saving...')
    save_dataset(
        features=embeddings.numpy(), 
        labels=dataset.labels.numpy(), 
        category_name=list(dataset.category_name.values()), 
        filename=args.output_filename
    )
    print(f'saved at {args.output_filename}')
    print('done')



if __name__ == '__main__':
    run(args=None)

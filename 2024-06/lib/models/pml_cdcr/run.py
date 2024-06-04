import os
import argparse
import torch
from torchvision import transforms
from datetime import datetime

from ._utils import build_model
from .main import train_model

from ...dataset import build_dataset
from ...utils import init_cuda_environment


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='PML-CDCR')

    parser.add_argument('--dataset', default='coco2014', type=str,
                        choices=['coco2014', 'coco2017', 'voc2007', 'voc2012', 'nus_wide', 'rand'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone', default='resnet101', type=str,
                        choices=['resnet18', 'resnet101', 'tresnet_l', 'tresnet_xl'])
    parser.add_argument('--ema-decay', default=0.82)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--image-size', default=224, type=int,
                        metavar='N', help='input image size')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--noise-rate', type=float, default=0.2,
                        help='corruption rate, should be less than 1')
    parser.add_argument('--threshold', default=0.6, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('--warmup-epochs', type=int, default=15)
    parser.add_argument('--cdcr-epochs', type=int, default=40)
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device be used')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed for initialization')

    # parse arguments, decide loaded path, saved path
    args = parser.parse_args(args)
    args.current_time = str(datetime.now())

    return args


def run(args=None):
    args = parse_args(args)
    print(__file__)
    print(args)

    init_cuda_environment(seed=args.seed, device=args.gpu)

    print('creating dataset...')

    train_dataset = build_dataset(
        name=args.dataset, divide='train', base=os.path.expanduser('~/dataset/'),
        add_index=True, add_partial_noise=True, noise_rate=args.noise_rate,
        transforms=[transforms.ToTensor()], flatten=True
    )
    valid_dataset = build_dataset(
        name=args.dataset, divide='test', base=os.path.expanduser('~/dataset/'),
        add_index=True, add_partial_noise=False,
        transforms=[transforms.ToTensor()], flatten=True
    )
    args.num_classes = train_dataset.num_classes

    print("len(train_dataset)): ", len(train_dataset))
    print("len(valid_dataset)): ", len(valid_dataset))

    # Pytorch Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    print('done\n')

    # Setup model
    print('creating model...')
    model = build_model(args.backbone, args.num_classes, pretrained=True)
    model = model.cuda()

    # Actuall Training
    train_model(
        model, train_loader, valid_loader,
        warmup_epochs=args.warmup_epochs, cdcr_epochs=args.cdcr_epochs,
        lr=args.lr, ema_decay=args.ema_decay, 
        threshold=args.threshold, image_size=args.image_size
    )

    print('train done.')



if __name__ == '__main__':
    run(args=None)

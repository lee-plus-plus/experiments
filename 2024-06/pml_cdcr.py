import os
import argparse
import torch
import sys
from tqdm import tqdm
from copy import deepcopy
from randaugment import RandAugment
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import normalize, mse_loss, binary_cross_entropy
from torchmetrics import MetricCollection

from lib.dataset import build_dataset, supported_multilabel_datasets
from lib.transforms import RandomCutout
from lib.metric import (
    MultilabelHammingDistance,
    MultilabelRankingLoss,
    MultilabelAveragePrecision,
    MultilabelAveragePrecision2,
    MultilabelF1Score,
)
from lib.backbone import (
    build_cnn_backbone,
    cnn_backbone_info,
    set_layer_by_name,
    MlpNet
)
from lib.checkpointer import Checkpointer
from lib.utils import init_cuda_environment, str2bool, best_gpu
from lib.meter import AverageMeter
from lib.ema import ExponentialMovingAverage as ModelEma
from lib.amp import GradScaler, autocast


@torch.no_grad()
def curriculum_disambiguation(model, train_loader, threshold):
    true_labels = train_loader.dataset.labels
    partial_labels = train_loader.dataset.partial_labels
    confidences = torch.zeros_like(true_labels).float()

    model.eval()
    for (img_w, img_s, y_true, y_partial, idxs) in train_loader:
        with autocast():
            y_score, x = model(img_w.cuda())
        confidences[idxs] = y_score.cpu()

    weights = ((confidences > threshold) | (partial_labels == 0)).float()
    return weights


@torch.no_grad()
def paritial_confusion_matrix(labels, partial_labels, confidences):
    preds = confidences.cpu()[partial_labels.cpu() == 1]
    target = labels.cpu()[partial_labels.cpu() == 1]
    total = target.numel()

    TP, FP, TN, FN = [
        ((preds == p) & (target == t)).sum() / total
        for p, t in [(1, 1), (1, 0), (0, 0), (0, 1)]
    ]
    # return TP, FP, TN, FN
    return f'TP={TP:.0%}, FP={FP:.0%}, TN={TN:.0%}, FN={FN:.0%}'


def weighted_mse_loss(input, target, weight=None):
    loss = mse_loss(input.float(), target.float(), reduction='none')
    if weight is not None:
        loss *= weight
    return loss.mean()


@torch.no_grad()
def evaluate_mAP(model, valid_loader):
    num_classes = valid_loader.dataset.num_classes
    metric = MultilabelAveragePrecision(num_labels=num_classes).cuda()

    model.eval()
    for (img, y_true, y_partial, idxs) in valid_loader:
        img, y_true = img.cuda(), y_true.cuda()

        with autocast():
            y_score, x = model(img)

        metric.update(y_score, y_true)

    return metric.cpu().compute().item()


@torch.no_grad()
def evaluate_in_detail(model, valid_loader):
    num_classes = valid_loader.dataset.num_classes

    metric = MetricCollection({
        'f1_score': MultilabelF1Score(num_classes),
        'h_loss': MultilabelHammingDistance(num_classes, threshold=0.5),
        'mAP': MultilabelAveragePrecision(num_classes),
        'mAP2': MultilabelAveragePrecision2(num_classes),
        'r_loss': MultilabelRankingLoss(num_classes),
    }).cuda()

    model.eval()
    for (img, y_true, y_partial, idxs) in valid_loader:
        img, y_true = img.cuda(), y_true.cuda()

        with autocast():
            y_score, x = model(img)

        metric.update(y_score, y_true)

    return metric.cpu().compute()


def build_transform(name, *, image_size=224):
    if name == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Normalize(),
        ])
    elif name == 'weak_augment':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalize(),
        ])
    elif name == 'strong_augment':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            RandomCutout(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            # Normalize(),
        ])
    elif name == 'valid':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Normalize(),
        ])
    else:
        raise ValueError(f'unsupported transform name ‘{name}’')


def build_model(backbone_name, num_classes, dim_embed, *, pretrained=True):
    class MyModel(torch.nn.Module):
        '''
        img -[backbone]-> encoding -┬-[classific_projector]----> y_logit
                                    └-[contrastive_projector]--> y_embed
        '''

        def __init__(self, backbone_name, num_classes, dim_embed, pretrained):
            super().__init__()
            dim_encoding = 512
            backbone = build_cnn_backbone(
                name=backbone_name, pretrained=pretrained)
            info = cnn_backbone_info(backbone)
            set_layer_by_name(
                backbone, info['layer_name_fc'],
                torch.nn.Linear(info['dim_fc'], dim_encoding)
            )

            classific_projector = MlpNet(
                dim_encoding, num_classes, [dim_encoding])
            contrastive_projector = torch.nn.Linear(dim_encoding, dim_embed)

            self.backbone_name = backbone_name
            self.pretrained = pretrained
            self.dim_encoding = dim_encoding
            self.dim_embed = dim_embed
            self.num_classes = num_classes
            self.backbone = backbone
            self.classific_projector = classific_projector
            self.contrastive_projector = contrastive_projector

        def forward(self, x):
            batch_size, _, H, W = x.shape  # (batch_size, 3, H, W)

            x = self.backbone(x)  # (batch_size, dim_encoding)

            y_logit = self.classific_projector(x)  # (batch_size, num_classes)
            y_embed = self.contrastive_projector(x)  # (batch_size, dim_embed)
            y_embed = normalize(y_embed, dim=-1)

            return torch.sigmoid(y_logit).float(), y_embed.float()

        def __repr__(self):
            return f'MyModel(backbone_name={self.backbone_name}, ' \
                   f'num_classes={self.num_classes}, ' \
                   f'dim_embed={self.dim_embed}, ' \
                   f'pretrained={self.pretrained})'

    model = MyModel(backbone_name, num_classes,
                    dim_embed=dim_embed, pretrained=pretrained)
    return model


def train_model(
    model, train_loader, valid_loader,
    *,
    # train setting
    show_progress, lr, epochs, use_augmentation, use_ema, optimizer,
    # cdcr setting
    use_cdcr, warmup_epochs, threshold,
    export_checkpoints,
):
    if use_ema:
        ema = ModelEma(model, decay=0.80 ** (1 / len(train_loader))).cuda()
    model.cuda()

    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.9, weight_decay=5e-5)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr, weight_decay=5e-5)
    scaler = GradScaler()

    dataset = train_loader.dataset
    dataset.transforms = [
        build_transform('weak_augment'),
        build_transform('strong_augment')
    ]

    X = torch.zeros((args.num_samples, args.dim_embed))
    Y_true = dataset.labels
    Y_partial = dataset.partial_labels
    Y_score = Y_partial.clone().float()
    Y_pseudo = Y_partial.clone().float()
    W = torch.ones((args.num_samples, args.num_classes))

    with torch.no_grad():
        for img_w, img_s, y_true, y_partial, idxs in tqdm(
            train_loader, disable=not show_progress, leave=False, mininterval=1
        ):
            with autocast():
                y_score, x = model(img_w.cuda())
            X[idxs] = x.cpu()
            Y_score[idxs] = y_score.cpu()

    def save_fn(score, model):
        checkpoint = deepcopy({
            'epoch': epoch,
            'mAP': score,
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'Y_score': Y_score.detach().cpu(),
            'Y_pseudo': Y_pseudo.detach().cpu(),
        })

        if epoch == 0:
            checkpoint['labels'] = deepcopy(dataset.labels)
            checkpoint['partial_labels'] = deepcopy(dataset.partial_labels)
            checkpoint['valid_labels'] = deepcopy(dataset.labels)

        return checkpoint

    def load_fn(checkpoint):
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return checkpoint

    checkpointer = Checkpointer(
        patience=5 * (2 if use_ema else 1), save_fn=save_fn, load_fn=load_fn,
        save_every_epoch=bool(export_checkpoints))
    train_meter = AverageMeter()

    # begin training
    # --------------
    for epoch in range(epochs):
        if checkpointer.early_stop():
            print('early stopping.')
            break

        if use_cdcr and (epoch >= warmup_epochs):
            W = ((Y_score > threshold) | (Y_partial == 0)).float()
            # W = curriculum_disambiguation(model, train_loader, threshold)

        model.train()
        for img_w, img_s, y_true, y_partial, idxs in tqdm(
            train_loader, disable=not show_progress, leave=False, mininterval=1
        ):
            img = (img_s if use_augmentation else img_w).cuda()
            y_target = (y_partial).cuda()
            weight = W[idxs].cuda()

            with autocast():
                y_score, x = model(img)
            loss = binary_cross_entropy(
                y_score, y_target.float(), weight=weight,
                reduction='mean')

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                with autocast():
                    y_score, x = model(img_w.cuda())
                X[idxs] = x.cpu()
                Y_score[idxs] = y_score.cpu()

            if use_ema:
                ema.update_parameters(model)
            train_meter.update(loss=loss.item())

        mAP = evaluate_mAP(model, valid_loader)
        if use_ema:
            mAP_ema = evaluate_mAP(ema.module, valid_loader)
        if epoch % 1 == 0:
            print(
                f'Epoch {epoch}: mAP {mAP:.2%}, '
                f"{f'mAP_ema {mAP_ema:.2%}, ' if use_ema else ''}"
                f'selection: {(W * Y_partial).sum() / Y_partial.sum():.0%} '
                f'({paritial_confusion_matrix(Y_true, Y_partial, W)}), '
                f'{str(train_meter)}'
            )

        checkpointer.update(score=mAP, model=model)
        if use_ema:
            checkpointer.update(score=mAP_ema, model=ema.module)

    ckpt = checkpointer.load()
    print(f'load model from epoch {ckpt["epoch"]:d} '
          f'with highest mAP {ckpt["mAP"]:.2%}')
    print(evaluate_in_detail(model, valid_loader))

    if export_checkpoints:
        torch.save(checkpointer.checkpoints, export_checkpoints)


def main(args, train_model_func):
    # initailize dataset, dataloader
    # ------------------------------

    print('loading dataset...', end=' ')
    train_dataset = build_dataset(
        name=args.dataset, divide='train',
        add_index=True, add_partial_noise=True, noise_rate=args.noise_rate,
        transforms=[build_transform('none')],  # will be over-written
        flatten=True
    )
    valid_dataset = build_dataset(
        name=args.dataset, divide='test',
        add_index=True, add_partial_noise=True, noise_rate=args.noise_rate,
        transforms=[build_transform('valid')],
        flatten=True
    )
    print(train_dataset)

    args.num_samples = train_dataset.num_samples
    args.num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
    )

    print('loading model...', end=' ')
    model = build_model(
        backbone_name=args.backbone,
        num_classes=train_dataset.num_classes,
        dim_embed=args.dim_embed,
        pretrained=args.pretrained
    )
    print(model)

    print('start training')

    train_model_func(
        model, train_loader, valid_loader,
        # train setting
        show_progress=args.show_progress, lr=args.lr, epochs=args.epochs,
        use_augmentation=args.use_augmentation, use_ema=args.use_ema,
        optimizer=args.optimizer,
        # cdcr setting
        use_cdcr=args.use_cdcr,
        threshold=args.threshold, warmup_epochs=args.warmup_epochs,
        export_checkpoints=args.export_checkpoints

    )

    os.makedirs('checkpoint/', exist_ok=True)
    torch.save(model.state_dict(),
               f'checkpoint/model_{__file__.split(".")[0]}.pth')
    print('done.')


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='pml_cdcr_run.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add = parser.add_argument

    group = parser.add_argument_group('model setting')
    group.add = group.add_argument
    group.add('--backbone', default='resnet101', type=str,
              choices=['resnet50', 'resnet101', 'tresnet_l', 'tresnet_xl'])
    group.add('--pretrained', default=True, type=str2bool)
    group.add('--dim-embed', default=256, type=int)

    group = parser.add_argument_group('dataset setting')
    group.add = group.add_argument
    group.add('--dataset', default='rand', type=str,
              choices=supported_multilabel_datasets())
    group.add('--noise-rate', default=0.2, type=float)
    group.add('--image-size', default=224,
              type=int, choices=[224, 448])

    # cdcr hyper-parameters
    group = parser.add_argument_group('PML-CDCR hyper-parameters')
    group.add = group.add_argument
    group.add('--use-cdcr', type=str2bool, default=False)
    group.add('--warmup-epochs', type=int, default=5)
    group.add('--threshold', type=float, default=0.6)

    group = parser.add_argument_group('training setting')
    group.add = group.add_argument
    group.add('--epochs', default=20, type=int)
    group.add('--use-augmentation', type=str2bool, default=False)
    group.add('--use-ema', type=str2bool, default=False)
    group.add('--batch-size', default=32, type=int)
    group.add('--lr', default=1e-04, type=float)
    group.add('--optimizer', default='Adam',
              type=str, choices=['SGD', 'Adam'])

    group = parser.add_argument_group('misc setting')
    group.add = group.add_argument
    group.add('--seed', help='random seed', default=123, type=int)
    group.add('--gpu', type=str, default=str(best_gpu()))
    group.add('--export-checkpoints', default='', type=str)
    group.add('--show-progress', type=str2bool,
              default=(os.isatty(sys.stdout.fileno())))

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    # print(__file__)
    # print(datetime.now().strftime(f'%Y-%m-%d %H:%M:%S'))
    print(f'{args = }')

    init_cuda_environment(seed=args.seed, device=args.gpu)

    main(args, train_model_func=train_model)

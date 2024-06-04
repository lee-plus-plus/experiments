import torch
import random
import torchvision.transforms as transforms
from randaugment import RandAugment
from copy import deepcopy

from ._utils import add_weight_decay
from ._eval import evaluate_mAP, evaluate_in_detail

from ...loss.asl import AsymmetricBceLoss
from ...amp import GradScaler, autocast
from ...ema import ExponentialMovingAverage as ModelEma
from ...transforms import RandomCutout


@torch.no_grad()
def evaluate_disambigation(labels, partial_labels, weights):
    preds = weights[partial_labels == 1]
    target = labels[partial_labels == 1]
    total = target.numel()
    TP = ((preds == 1) & (target == 1)).sum() / total
    FP = ((preds == 1) & (target == 0)).sum() / total
    TN = ((preds == 0) & (target == 0)).sum() / total
    FN = ((preds == 0) & (target == 1)).sum() / total
    return TP, FP, TN, FN


@torch.no_grad()
def confidence_selection(confidences, targets, threshold, use_diff=False):
    weights = torch.zeros_like(confidences)
    zero_targets_mask = targets == 0

    if not use_diff:
        weights[zero_targets_mask] = 1
        confident_mask = (targets == 1) & (confidences >= threshold)
        weights[confident_mask] = 1

    else:
        weights[zero_targets_mask] = 1
        confident_mask = (targets == 1) & (confidences >= 0)
        confidences_true = torch.where(
            confident_mask, confidences, torch.tensor(0.0))
        confidences_true_sum = torch.sum(confidences_true, dim=0)
        confidences_true_num = torch.sum(targets != 0, dim=0)
        confidences_true_mean = confidences_true_sum / confidences_true_num
        confidences_mean = torch.sum(
            confidences_true_sum) / torch.sum(confidences_true_num)
        confidences_tmp = confidences_true - confidences_true_mean
        threshold -= confidences_mean
        weights[confidences_tmp >= threshold] = 1

    return weights.float()


@torch.no_grad()
def curriculum_disambiguation(model, train_loader, threshold):
    true_labels = train_loader.dataset.labels
    partial_labels = train_loader.dataset.partial_labels
    confidences = torch.zeros_like(true_labels).float()

    for x_aug, x, y, y_partial, idxs in train_loader:
        with autocast():
            y_score = torch.sigmoid(model(x.cuda())).cpu().float()
        confidences[idxs] = y_score
    weights = confidence_selection(confidences, partial_labels, threshold)

    TP, FP, TN, FN = evaluate_disambigation(
        true_labels, partial_labels, weights)
    print(f"disambiguation: TP={TP:.1%}, FP={FP:.1%}, TN={TN:.1%}, FN={FN:.1%}")

    return weights


def train_model_warmup(
    model, train_loader, valid_loader,
    epochs, lr, ema_decay, image_size
):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        RandomCutout(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_loader.dataset.transforms = [train_transform]
    valid_loader.dataset.transforms = [valid_transform]

    ema = ModelEma(model, ema_decay ** (1 / len(train_loader)))

    criterion = AsymmetricBceLoss(0, 0, clip=0, no_focal_grad=False)
    parameters = add_weight_decay(model, weight_decay=1e-4)  # useless?
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.2
    )

    # begin training
    # --------------

    highest_mAP, checkpoint = 0, None
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        for i, (x_aug, y, y_partial, idxs) in enumerate(train_loader):
            x_aug = x_aug.cuda()
            y_partial = y_partial.cuda()

            with autocast():
                output = model(x_aug).float()

            loss = criterion(output, y_partial)
            model.zero_grad()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_scale_changed:
                scheduler.step()

            ema.update_parameters(model)

            if i % 100 == 0:
                print(
                    f'Epoch [{epoch}/{epochs}], '
                    f'Step [{i:d}/{len(train_loader):d}], '
                    f'LR {scheduler.get_last_lr()[0]:.1e}, '
                    f'Loss: {loss.item():.1f}'
                )
        
        print('starting validation')
        mAP_reg = evaluate_mAP(model, valid_loader)
        mAP_ema = evaluate_mAP(ema, valid_loader)
        mAP_score = max(mAP_reg, mAP_ema)

        model.train()
        ema.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            model_best = model if mAP_reg > mAP_ema else ema.module
            checkpoint = deepcopy({
                'epoch': epoch, 'mAP': mAP_score,
                'state_dict': model_best.state_dict()
            })

        print(f'mAP regular = {mAP_reg:.2%}, mAP EMA = {mAP_ema:.2%}')

    # use best checkpoint
    # -------------------

    model.load_state_dict(checkpoint['state_dict'])
    print(f'load model with highest mAP {highest_mAP:.2%}')
    print(evaluate_in_detail(model, valid_loader))


def train_model_cdcr(
    model, train_loader, valid_loader,
    epochs, lr, ema_decay, threshold, image_size
):
    # initialization
    # --------------

    train_transform_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        RandomCutout(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()
    ])
    train_transform_2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    train_loader.dataset.transforms = [train_transform_1, train_transform_2]
    valid_loader.dataset.transforms = [valid_transform]

    ema = ModelEma(model, ema_decay ** (1 / len(train_loader)))

    criterion = AsymmetricBceLoss(0, 0, clip=0, no_focal_grad=False)
    parameters = add_weight_decay(model, weight_decay=1e-4)  # useless?
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.2
    )

    # begin training
    # --------------

    highest_mAP, checkpoint = 0, None
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.eval()
        weights = curriculum_disambiguation(model, train_loader, threshold)

        model.train()
        for i, (x_aug, x, y, y_partial, idxs) in enumerate(train_loader):
            x_aug = x_aug.cuda()
            y_partial = y_partial.cuda()
            weight = weights[idxs].cuda()

            with autocast():
                output = model(x_aug).float()

            loss = criterion(output, y_partial, weight)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_scale_changed:
                scheduler.step()

            ema.update_parameters(model)

            if i % 100 == 0:
                print(
                    f'Epoch [{epoch}/{epochs}], '
                    f'Step [{i:d}/{len(train_loader):d}], '
                    f'LR {scheduler.get_last_lr()[0]:.1e}, '
                    f'Loss: {loss.item():.1f}'
                )

        print('starting validation')
        mAP_reg = evaluate_mAP(model, valid_loader)
        mAP_ema = evaluate_mAP(ema, valid_loader)
        mAP_score = max(mAP_reg, mAP_ema)

        model.train()
        ema.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            model_best = model if mAP_reg > mAP_ema else ema.module
            checkpoint = {
                'epoch': epoch, 'mAP': mAP_score,
                'state_dict': model_best.state_dict()
            }

        print(f'mAP regular = {mAP_reg:.2%}, mAP EMA = {mAP_ema:.2%}')

    # use best checkpoint
    # -------------------

    model.load_state_dict(checkpoint['state_dict'])
    print(f'load model with highest mAP {highest_mAP:.2%}')
    print(evaluate_in_detail(model, valid_loader))


def train_model(
    model, train_loader, valid_loader,
    warmup_epochs, cdcr_epochs,
    lr, ema_decay, threshold, image_size
):
    print('begin first stage training')
    train_model_warmup(
        model, train_loader, valid_loader,
        warmup_epochs, lr, ema_decay, image_size
    )
    print('end first stage training')

    print('begin second stage training')
    train_model_cdcr(
        model, train_loader, valid_loader,
        cdcr_epochs, lr, ema_decay, threshold, image_size
    )
    print('end second stage training')

    print(evaluate_in_detail(model, valid_loader))

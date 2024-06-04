import torch

from torchmetrics import MetricCollection
from torch.cuda.amp import autocast

from ...metric import (
    MultilabelHammingDistance,
    MultilabelRankingLoss,
    MultilabelAveragePrecision,
    MultilabelAveragePrecision2,
    MultilabelF1Score,
)
from ...utils import get_model_device, ignore_warnings


@torch.no_grad()
def evaluate(model, valid_loader, metric):
    device = get_model_device(model)
    model.eval()
    metric.reset()

    for batch in valid_loader:
        x, y_true = batch[0], batch[1]
        with autocast():
            y_score = torch.sigmoid(model(x.to(device))).cpu()
        metric(y_score, y_true)

    result = metric.compute()
    metric.reset()

    return result


@ignore_warnings
def evaluate_mAP2(model, valid_loader):
    num_classes = valid_loader.dataset.num_classes
    metric = MultilabelAveragePrecision2(num_labels=num_classes)
    return evaluate(model, valid_loader, metric)


@ignore_warnings
def evaluate_mAP(model, valid_loader):
    num_classes = valid_loader.dataset.num_classes
    metric = MultilabelAveragePrecision(num_labels=num_classes)
    return evaluate(model, valid_loader, metric)


@ignore_warnings
def evaluate_in_detail(model, valid_loader, threshold=0.5):
    num_classes = valid_loader.dataset.num_classes
    metric = MetricCollection([
        MultilabelAveragePrecision(num_labels=num_classes),
        MultilabelAveragePrecision2(num_labels=num_classes),
        MultilabelHammingDistance(num_labels=num_classes, threshold=threshold),
        MultilabelRankingLoss(num_labels=num_classes),
        MultilabelF1Score(num_labels=num_classes),
    ])
    return evaluate(model, valid_loader, metric)

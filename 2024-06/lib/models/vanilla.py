'''template for multi-label / partial-multi-label model implementations
'''
import torch
from ._utils import AverageMeter


def train_epoch(model, train_loader, criterion, optimizer):
    meter = AverageMeter()
    model.train()
    for i, (x, y, idxs) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()

        y_logit = model(x)
        loss = criterion(y_logit, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter.update(loss=loss.item())

    return meter.average_value()


def eval_epoch(model, valid_loader, criterion):
    meter = AverageMeter()
    model.eval()
    for i, (x, y, idxs) in enumerate(valid_loader):
        x = x.cuda()
        y = y.cuda()

        y_logit = model(x)
        score = criterion(y_logit, y)

        meter.update(score=loss.item())

    return meter.average_value()

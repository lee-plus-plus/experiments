import torch
from torch.nn import MSELoss, BCELoss
from copy import deepcopy

from ...checkpointer import Checkpointer
from ._eval import evaluate_mAP, evaluate_in_detail


def build_criterion(name):
    if name == 'bce':
        criterion = BCELoss()
    elif name == 'mse':
        criterion = MSELoss()
    else:
        raise ValueError("unsupported criterion name ‘{}’".format(name))
    return criterion


def train_model(
    model, train_loader, valid_loader, criterion_name,
    lr, epochs,
):

    criterion = build_criterion(criterion_name)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=5e-5
    )

    def save_fn(score, model):
        return deepcopy({
            'epoch': epoch, 'mAP': mAP,
            'state_dict': model.state_dict()
        })

    def load_fn(checkpoint):
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return checkpoint

    checkpointer = Checkpointer(patience=10, save_fn=save_fn, load_fn=load_fn)

    for epoch in range(epochs):
        if checkpointer.early_stop():
            break

        model.train()
        for i, (x, y_true, y_partial, idxs) in enumerate(train_loader):

            x, y_partial = x.cuda(), y_partial.cuda()
            y_score = torch.sigmoid(model(x))
            loss = criterion(y_score, y_partial.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mAP = evaluate_mAP(model, valid_loader)
        checkpointer.update(score=mAP, model=model)
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: mAP {mAP:.2%}')

    checkpoint = checkpointer.load()  # use best checkpoint
    print(f'load model from epoch {checkpoint["epoch"]:d} '
          f'with highest mAP {checkpoint["mAP"]:.2%}')
    print(evaluate_in_detail(model, valid_loader))

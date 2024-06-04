import torch
from torch import nn


# Asymmetric Loss For Multi-Label Classification
def asymmetric_bce_loss(
    y_logit, y_target,
    gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
    no_focal_grad=True,
    weight=None, reduce='mean',
):
    y_score = torch.sigmoid(y_logit)
    y_score_pos = y_score
    y_score_neg = 1 - y_score

    # Probability Shifting for Negative Prediction
    y_score_neg = (y_score_neg + clip).clamp(max=1)

    # Binary Cross Entropy
    los_pos = -y_target * torch.log(y_score_pos.clamp(min=eps))
    los_neg = -(1 - y_target) * torch.log(y_score_neg.clamp(min=eps))
    loss = los_pos + los_neg  # (batch_size, num_classes)

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        base = y_score_pos * y_target + y_score_neg * (1 - y_target)
        gamma = gamma_pos * y_target + gamma_neg * (1 - y_target)
        focusing_weight = torch.pow(1 - base, gamma)

        if no_focal_grad:
            focusing_weight.detach_()

        loss *= focusing_weight

    if weight is not None:
        loss *= weight

    if reduce == 'mean':
        loss = loss.sum(dim=1).mean()
    elif reduce == 'sum':
        loss = loss.sum()
    elif reduce == 'none':
        pass
    return loss


class AsymmetricBceLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, no_focal_grad=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.no_focal_grad = no_focal_grad

    def forward(self, y_logit, y_target, weight=None, reduce='mean'):
        '''Parameters
        ----------
        y_logit: (batch_size, num_classes)
        y_target: (batch_size, num_classes)

        loss_pos = ((1 - y_score_pos) ** gamma_pos) * log(y_score_pos)
        loss_neg = ((1 - y_score_neg) ** gamma_neg) * log(y_score_neg)
        loss = y_target * loss_pos + (1 - target) * loss_neg
        '''

        return asymmetric_bce_loss(
            y_logit, y_target,
            gamma_neg=self.gamma_neg, gamma_pos=self.gamma_pos,
            clip=self.clip, eps=self.eps, 
            no_focal_grad=self.no_focal_grad,
            weight=weight, reduce=reduce, 
        )
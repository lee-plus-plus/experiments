import torch
from torch import nn


def self_excluded_log_softmax(logits, dim):
    mask = 1 - torch.eye(*logits.shape, device=logits.device)
    exp_logits = torch.exp(logits) * mask
    log_proba = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    return log_proba


def info_nce_loss(query, keys, positive_mask, temperature=0.07):
    # query: (num_query, dim_embed)
    # keys: (num_keys, dim_embed)
    # positive_mask: (num_query, num_keys)
    assert query.shape[1] == keys.shape[1], "dim of embedding is not consistant"
    assert query.shape[0] == positive_mask.shape[0], "shape of positive_mask is not aligned"
    assert keys.shape[0] == positive_mask.shape[1], "shape of positive_mask is not aligned"
    assert positive_mask.min() >= 0, "negative value of positive_mask is lack of interpretation"

    assert 0 not in positive_mask.max(dim=1).values, "no positive pair"

    logit = query @ keys.t() / temperature  # (n, m)
    loss = (-positive_mask * self_excluded_log_softmax(logit, dim=1)
            ).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-12)
    loss = loss.mean()

    return loss


# supervised contrastive learning loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(features, labels):
        # features: (batch_size, dim_embed)
        # labels: (batch_size, )
        batch_size, dim_embed = features.shape
        labels = labels.reshape(batch_size, 1)

        # mask[i, j] <=> labels[i] == 1 and labels[j] == 1 and i != j
        mask = (labels.t() * labels) * (1 - torch.eye(batch_size))

        loss = info_nce_loss(features, features, mask,
                             temperature=self.temperature)
        loss *= (temperature / base_temperature)

        return loss

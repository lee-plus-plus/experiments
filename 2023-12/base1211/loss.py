import torch
from torch.nn.functional import mse_loss, cross_entropy, binary_cross_entropy, log_softmax
from sklearn.metrics import (
    hamming_loss as hamming_loss_numpy,
    label_ranking_loss as label_ranking_loss_numpy,
    average_precision_score as avg_precison_score_numpy,
)
from leemultilearn.metrics import (
    hamming_loss as hamming_loss_numpy,
    label_ranking_loss as label_ranking_loss_numpy,
    average_precision_score as average_precision_score_numpy
)
from sklearn.utils import check_array, check_consistent_length


def average_precision_score_numpy(y_true, y_score, *, sample_weight=None):
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)

    ap = 0.0
    for i in range(y_true.shape[0]):
        ap += avg_precison_score_numpy(y_true[i], y_score[i])
    ap /= y_true.shape[0]
    return ap


@torch.no_grad()
def hamming_loss(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return hamming_loss_numpy(y_true, y_pred)


@torch.no_grad()
def label_ranking_loss(y_true, y_pred_proba):
    y_true = y_true.cpu().detach().numpy()
    y_pred_proba = y_pred_proba.cpu().detach().numpy()
    return label_ranking_loss_numpy(y_true, y_pred_proba)


@torch.no_grad()
def average_precision_score(y_true, y_pred_proba):
    y_true = y_true.cpu().detach().numpy()
    y_pred_proba = y_pred_proba.cpu().detach().numpy()
    return average_precision_score_numpy(y_true, y_pred_proba)



def info_nce_loss(query, keys, positive_mask, temperature=0.07):
    # query: (n, d)
    # keys: (m, d)
    # positive_mask: (n, m)
    assert query.shape[1] == keys.shape[1], "dimension of embedding is not consistant"
    assert query.shape[0] == positive_mask.shape[0], "shape of positive_mask is not aligned"
    assert keys.shape[0] == positive_mask.shape[1], "shape of positive_mask is not aligned"
    assert positive_mask.min() >= 0, "negative value of positive_mask is lack of interpretation"
    
    assert 0 not in positive_mask.max(dim=1).values, "no positive pair"
    
    logit = query @ keys.t() / temperature # (n, m)
    loss = (-positive_mask * log_softmax(logit, dim=1)).sum(dim=1) / positive_mask.sum(dim=1)
    loss = loss.mean()
    
    return loss
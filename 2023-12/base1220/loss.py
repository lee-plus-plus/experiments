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

    logit = query @ keys.t() / temperature  # (n, m)
    loss = (-positive_mask * log_softmax(logit, dim=1)
            ).sum(dim=1) / positive_mask.sum(dim=1)
    loss = loss.mean()

    return loss


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, y_logit, y_target, weight=None, reduce='mean'):
        '''
        Parameters
        ----------
        y_logit: (batch_size, num_classes)
        y_target: (batch_size, num_classes)
        
        loss_pos = ((1 - y_score_pos) ** gamma_pos) * log(y_score_pos)
        loss_neg = ((1 - y_score_neg) ** gamma_neg) * log(y_score_neg)
        loss = y_target * loss_pos + (1 - target) * loss_neg
        '''

        # Calculating Probabilities
        y_score = torch.sigmoid(y_logit)
        y_score_pos = y_score
        y_score_neg = 1 - y_score

        # Probability Shifting for Negative Prediction
        y_score_neg = (y_score_neg + self.clip).clamp(max=1)

        # Binary Cross Entropy
        los_pos = y_target * torch.log(y_score_pos.clamp(min=self.eps))
        los_neg = (1 - y_target) * torch.log(y_score_neg.clamp(min=self.eps))
        loss = los_pos + los_neg # (batch_size, num_classes)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            base = y_score_pos * y_target + y_score_neg * (1 - y_target) 
            gamma = self.gamma_pos * y_target + self.gamma_neg * (1 - y_target)
            focusing_weight = torch.pow(1 - base, gamma)
            loss *= focusing_weight

        if weight is not None:
            loss *= weight

        if reduce == 'mean':
            loss = loss.sum(dim=1).mean()
        elif reduce == 'sum':
            loss = loss.sum()
        elif reduce == 'none':
            pass
        return -loss.sum()


def self_excluded_log_softmax(logits, dim):
    mask = 1 - torch.eye(*logits.shape)
    exp_logits = torch.exp(logits) * mask
    log_proba = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    return log_proba


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

        mask = (labels.t() * labels) # mask[i, j] <=> labels[i] and labels[j]
        mask = mask * (1 - torch.eye(batch_size)) # and i != j
        
        logits = features @ features.t() / temperature
        logits = logits - logits.max(dim=1, keepdims=True).values.detach() # for numerical stability, no effect
        log_proba = self_excluded_log_softmax(logits, dim=1)
        mean_log_proba = (mask * log_proba).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        
        loss = - (temperature / base_temperature) * mean_log_proba
        loss = loss.mean()
        
        return loss


class MultiSupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.2, base_temperature=0.2):
        super(MultiSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = 1e-5

    def forward(self, features, labels=None, mask=None, multi=True):
        # features: [batch_size, n_view, num_classes, dim]
        # labels: [batch_size, num_classes]

        device = features.device
        batch_size, n_view, num_classes, dim = features.shape
        n_instance = batch_size * n_view

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # [batch_size * n_view, num_classes, dim]

        anchor_feature = contrast_feature # [batch_size * n_view, num_classes, dim]
        anchor_count = contrast_count

        # intra-class: 同类不同图片之间的对比

        # mask_intra[class, a*batch_size+i, b*batch_size+j] == 0 <=> labels[i, class] == labels[j, class] == 1 and i != j
        mask_intra = labels.t().unsqueeze(-1) @ labels.t().unsqueeze(1) # [num_classes, batch_size, batch_size]
        mask_intra = mask_intra.repeat(1, anchor_count, contrast_count) # [num_classes, batch_size*n_view, batch_size*n_view]
        mask_intra = mask_intra * (1 - torch.eye(batch_size * anchor_count))

        # num_classes * [batch_size*n_view, dim] @ [dim, batch_size*n_view] => [num_classes, batch_size*n_view, batch_size*n_view]
        logits_intra = anchor_feature.permute(1, 0, 2) @ contrast_feature.permute(1, 2, 0) / self.temperature
        logits_intra -= torch.max(logits_intra, dim=0, keepdim=True).values.detach() # 为了数值稳定
        logits_intra = mask_intra * logits_intra # [num_classes, batch_size*n_view, batch_size*n_view]

        # inter-class: 类间不同图片之间的对比

        mask_inter = labels.reshape(-1, 1) * labels.reshape(1, -1)
        mask_inter = mask_inter.repeat(anchor_count, contrast_count) # [batch_size*n_view*num_classes, batch_size*n_view*num_classes]
        mask_inter = mask_intra * (1 - torch.eye(batch_size*n_view*num_classes))

        all_features = contrast_feature.reshape(-1, dim) # [batch_size*n_view*num_classes, dim]
        logits_inter = all_features @ all_features.t() / self.temperature # [batch_size*n_view*num_classes, batch_size*n_view*num_classes]
        
        # 为了数值稳定 (类似于做 softmax?)
        logits_inter = logits_inter.reshape(-1, batch_size*n_view, num_classes)
        logits_inter = logits_inter - torch.max(logits_inter, dim=-2, keepdim=True).values.detach()
        logits_inter = logits_inter.reshape(batch_size*n_view*num_classes, batch_size*n_view*num_classes)

        logits_inter = torch.exp(logits_inter) * mask_inter

        # 分母 [batch_size*n_view*num_classes, batch_size*n_view*num_classes]
        logits_inter = logits_inter.sum(1, keepdim=True)
        logits_inter[logits_inter <= 0] = 1
        logits_inter = all_mask*torch.log(logits_inter)

        # 分子 [batch_size*n_view*num_classes, batch_size*n_view]
        logits_intra = (mask_intra*logits_intra).transpose(1, 0).reshape(
            batch_size*contrast_count*num_classes, batch_size*contrast_count)

        # 计算对数似然除以正的均值
        # [batch_size*n_view*num_classes, batch_size*n_view] - [batch_size*n_view*num_classes, 1]
        log_prob = logits_intra - logits_inter.sum(1, keepdim=True)
        # print(logits_intra.shape,     logits_inter.sum(1, keepdim=True).shape)
        mean_log_prob_pos = (log_prob).sum() / mask_intra.sum()

        # 计算loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(batch_size, anchor_count, num_classes).mean()
        return loss / batch_size

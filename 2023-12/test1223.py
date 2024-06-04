# coding: utf-8

# # MLL dataset with MulCon based on SADCL

# 修复了 COCO2017上数据集的bug，发现之前读取的数据只有正常数据的 1/10，
# 可能之前的实验结果并不具有参考意义. 

# In[1]:


import torch
from torch import nn
import torchvision
from tqdm import tqdm
from base1220 import (
    VocDataset, CocoDataset, NusWideDataset, 
    split_trainval_dataset,
    mse_loss, cross_entropy, binary_cross_entropy, 
    hamming_loss, label_ranking_loss, average_precision_score,
    MlpNet,
    AverageMeter, LastMeter, 
    init_cuda_environment, UnNormalize,
)
from typing import Dict
from torch.nn.functional import normalize, log_softmax


# In[2]:


from randaugment import RandAugment
from os.path import join
import torchvision.transforms as transforms


# ## Dataset

# In[3]:


def build_dataset(name, divide, base='/hdd/lzm/dataset', *, image_size=224):
    assert name in ['voc2007', 'voc2012', 'coco2014', 'coco2017', 'nuswide']
    assert divide in ['train', 'test']
    
    # default transform
    if divide == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if name in ['voc2007', 'voc2012']:
        year = name[-4:]
        divide = {
            ('2007', 'train'): 'trainval', ('2007', 'test'): 'test', 
            ('2012', 'train'): 'train', ('2012', 'test'): 'val', 
        }[(year, divide)]
        dataset = VocDataset(
            root_dir=join(base, name), 
            year=year,
            divide=divide, 
            transform=transform
        )
            
    elif name in ['coco2014', 'coco2017']:
        divide = {'train': 'train', 'test': 'val'}[divide]
        year = name[-4:]
        dataset = CocoDataset(
            img_dir=join(base, f'{name}/{divide}{year}/'),
            ann_path=join(base, f'{name}/annotations/instances_{divide}{year}.json'), 
            transform=transform
        )
            
    elif name in ['nuswide']:
        dataset = NusWideDataset(
            root_dir=join(base, name), 
            divide=divide, 
            transform=transform
        )
        
    return dataset


# In[4]:


# # test:
# dataset = build_dataset('coco2014', 'train')


# In[5]:


# # test:
# img, lbl = dataset[int(torch.randint(0, len(dataset), (1, )))]
# t = transforms.Compose([
#     UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.ToPILImage(),
# ])
# print(f'labels: {[dataset.category_name[int(elem)] for elem in lbl.nonzero().flatten()]}')
# t(img)


# ## Model

# In[6]:


class IntermediateLayerExtracter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layer: str) -> None:
        if return_layer not in [name for name, _ in model.named_children()]:
            raise ValueError("return_layer are not present in model")
        layers = dict()
        for name, module in model.named_children():
            layers[name] = module
            if name == return_layer:
                break

        super().__init__(layers)
        self.return_layer = return_layer

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        return x


# In[7]:


def build_cnn_backbone(name, *, pretrained=False):
    # build cnn backbone that return the feature-map (from immediate layer)
    pretrained_weight = {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101": torchvision.models.ResNet101_Weights.IMAGENET1K_V2,
        "resnet152": torchvision.models.ResNet152_Weights.IMAGENET1K_V2,
    }[name] if pretrained else False
    base_model = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
    }[name](pretrained=pretrained_weight)
    
    layer_name = "layer4"
    dim_channel = {
        "resnet18": 512,
        "resnet34": 512, 
        "resnet50": 2048, 
        "resnet101": 2048, 
        "resnet152": 2048, 
    }[name]
    downsample_ratio = 32
    
    backbone = IntermediateLayerExtracter(base_model, layer_name)

    return backbone, dim_channel, downsample_ratio


# In[8]:


# CNN backbone + multi-head attention + classification head
class MultiVisualEncoder(torch.nn.Module):
    def __init__(self, num_classes, dim_embed, *, backbone='resnet18', pretrained=False):
        super(MultiVisualEncoder, self).__init__()
        
        # CNN encoder
        # (batch_size 3, H, W) -> cnn_encoder -> (batch_size, dim_channel, H/32, W/32)
        cnn_encoder, dim_channel, downsample_ratio = build_cnn_backbone(name=backbone, pretrained=pretrained)
        
        # transformer 
        # ( key:   (batch_size, H/32 * W/32, dim_embed), 
        #   pos:   (batch_size, H/32 * W/32, dim_embed),
        #   query: (batch_size, num_classes, dim_embed), 
        # ) -> transformer -> (batch_size, num_classes, dim_embed)
        query_embed = torch.nn.Embedding(num_classes, dim_embed)
        transformer = torch.nn.Transformer(d_model=dim_embed, nhead=4, batch_first=True)
        
        # adapter from CNN-encoded feature to key-embedding 
        # (batch_size, H/32 * W/32, dim_channel) -> key_embed_projector -> (batch_size, H/32 * W/32, dim_embed)
        key_embed_projector = torch.nn.Linear(dim_channel, dim_embed)
        
        self.dim_embed = dim_embed
        self.dim_channel = dim_channel
        self.num_classes = num_classes
        self.cnn_encoder = cnn_encoder
        self.query_embed = query_embed
        self.transformer = transformer
        self.key_embed_projector = key_embed_projector
        # self.positional_embedder = positional_embedder      

    def forward(self, x):
        batch_size, _, H, W = x.shape # (batch_size, 3, H, W)
        
        x = self.cnn_encoder(x) # (batch_size, dim_channel, H/32, W/32)
        x = x.flatten(2, 3).permute(0, 2, 1) # (batch_size, H/32 * W/32, dim_channel)
    
        key = self.key_embed_projector(x) # (batch_size, H/32 * W/32, dim_embed)
        query = self.query_embed.weight.repeat(batch_size, 1, 1) # (batch_size, num_classes, dim_embed)
        features = self.transformer(key, query) # (batch_size, num_classes, dim_embed)
        
        return features


# In[9]:


# test
# m = MultiVisualEncoder(num_classes=80, dim_embed=2048, pretrained=False).cuda()
# x = torch.rand((32, 3, 224, 224)).cuda()
# y = m.forward(x)
# y.shape


# In[10]:


class MyNet(torch.nn.Module):
    def __init__(self, num_classes, dim_encoding, dim_contrastive, *, backbone='resnet18', pretrained=False):
        super(MyNet, self).__init__()
        
        self.num_classes = num_classes
        self.dim_encoding = dim_encoding
        self.dim_contrastive = dim_contrastive
        
        self.visual_encoder = MultiVisualEncoder(
            num_classes, dim_encoding, backbone=backbone, pretrained=pretrained
        )
        self.classific_projector = MlpNet(dim_encoding, 1, hidden_layer_sizes=[dim_encoding])
        self.contrastive_projector = torch.nn.Linear(dim_encoding, dim_contrastive)
        
    def forward(self, x):
        batch_size, _, H, W = x.shape # (batch_size, 3, H, W)
        x = self.visual_encoder(x) # (batch_size, num_classes, dim_encoding)
        
        y_logit = self.classific_projector(x).squeeze(-1) # (batch_size, num_classes)
        y_embed = self.contrastive_projector(x) # (batch_size, num_classes, dim_contrastive)
        y_embed = normalize(y_embed, dim=-1)
        
        return y_logit, y_embed


# In[11]:


# # test
# model = MyNet(num_classes=80, dim_encoding=2048, dim_contrastive=512).cuda()
# x = torch.rand((32, 3, 224, 224)).cuda()
# y_logit, y_embed = model(x)
# y_logit.shape, y_embed.shape


# ## Loss

# In[12]:


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, y_logit, y_target, *, weight=None, reduce='mean'):
        '''
        loss_pos = ((1 - y_score_pos) ** gamma_pos) * log(y_score_pos)
        loss_neg = ((1 - y_score_neg) ** gamma_neg) * log(y_score_neg)
        loss = y_target * loss_pos + (1 - target) * loss_neg
        '''

        batch_size, num_classes = y_logit.shape
        assert y_logit.shape == y_target.shape
        
        # Calculating Probabilities
        y_score = torch.sigmoid(y_logit)
        y_score_pos = y_score
        y_score_neg = 1 - y_score

        # Probability Shifting for Negative Prediction
        y_score_neg = (y_score_neg + self.clip).clamp(max=1)

        # Binary Cross Entropy
        los_pos = -y_target * torch.log(y_score_pos.clamp(min=self.eps))
        los_neg = -(1 - y_target) * torch.log(y_score_neg.clamp(min=self.eps))
        loss = los_pos + los_neg # (batch_size, num_classes)

        # Asymmetric Focusing
        if self.gamma_neg != 0 or self.gamma_pos != 0:
            with torch.no_grad():
                base = y_score_pos * y_target + y_score_neg * (1 - y_target) 
                gamma = self.gamma_pos * y_target + self.gamma_neg * (1 - y_target)
                focusing_weight = torch.pow(1 - base, gamma)
            loss *= focusing_weight

        if weight is not None:
            loss *= weight

        if reduce == 'mean':
            loss = loss.mean()
        elif reduce == 'sum':
            loss = loss.sum(dim=1).mean()
        elif reduce == 'none':
            pass
        return loss


# In[13]:


def asymmetric_loss(y_logit, y_target, weight=None, reduce='mean', **kwargs):
    return AsymmetricLoss(**kwargs)(y_logit, y_target, weight=weight, reduce=reduce)


# In[14]:


# # test
# y_logit = torch.rand((64, 80))
# y_target = (torch.rand((64, 80)) < 0.2).float()
# print(torch.nn.BCEWithLogitsLoss()(y_logit, y_target))
# print(AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0.0)(y_logit, y_target))
# print(AsymmetricLoss(gamma_neg=0.01, gamma_pos=0.01, clip=0.0)(y_logit, y_target))
# print(AsymmetricLoss(gamma_neg=0.01, gamma_pos=0.01, clip=0.05)(y_logit, y_target))


# In[15]:


def self_excluded_log_softmax(logits, dim):
    mask = 1 - torch.eye(*logits.shape)
    exp_logits = torch.exp(logits) * mask
    log_proba = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    return log_proba


# In[16]:


def sup_con_loss(features, labels, *, temperature=0.2, base_temperature=0.2):
    # features: (batch_size, dim_embed)
    # labels: (batch_size, 1)
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


# In[17]:


def sup_con_loss2(features1, features2, labels1, labels2, *, temperature=0.2, base_temperature=0.2):
    size1, dim_embed = features1.shape
    size2, dim_embed = features2.shape
    labels1 = labels1.reshape(size1, 1)
    labels2 = labels2.reshape(size2, 1)
    
    mask = labels1 * labels2.t() # [size1, size2]
    
    logits = features1 @ features2.t() / temperature # [size1, size2]
    logits = logits - logits.max(dim=1, keepdims=True).values.detach() # for numerical stability, no effect
    log_proba = log_softmax(logits, dim=1) # [size1, size2]
    mean_log_proba = (mask * log_proba).sum(dim=1) / (mask.sum(dim=1) + 1e-12) # [size1]
    
    loss = - (temperature / base_temperature) * mean_log_proba
    loss = loss.mean()
    
    return loss


# In[18]:


def multi_sup_con_loss(features, labels, *, temperature=0.2, base_temperature=0.2):
    # features: (batch_size, dim_embed)
    # labels: (batch_size, 1)
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


# In[19]:


# test
# features = normalize(torch.rand(4, 64), dim=1)
# labels = torch.tensor([0, 1, 1, 1])
# temperature = 0.2
# sup_con_loss(features, labels)


# ## Train & Eval

# ### Vanilla

# In[20]:


from torch.nn.functional import binary_cross_entropy_with_logits


# In[21]:


def train_epoch_vanilla(
    train_loader, optimizer, model, 
    embed_queue, y_true_queue, idx_queue, size_queue, 
    show_progress=False
):
    # global train_loader, optimizer, model
    # global embed_queue, y_true_queue, idx_queue, size_queue
    meter = AverageMeter()
    model.train()
    
    for image, y_true in tqdm(train_loader, leave=False, disable=not show_progress):
        image = image.cuda() # (batch_size, 3, H, W)
        y_true = y_true.cuda() # (batch_size, num_classes)
        batch_size, num_classes = y_true.shape

        y_logit, embed = model(image) # (batch_size, num_classes)
        y_score = torch.sigmoid(y_logit)
        y_pred = y_score > 0.5

        loss = asymmetric_loss(y_logit, y_true)
        # loss = binary_cross_entropy_with_logits(y_logit, y_true)
        
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update queue
        with torch.no_grad():
            embed_queue[idx_queue: idx_queue + batch_size, :, :] = embed.detach()
            y_true_queue[idx_queue: idx_queue + batch_size, :] = y_true.detach()
            idx_queue = (idx_queue + batch_size) % size_queue

        # record
        meter.update(
            loss=loss.item(), 
            h_loss=hamming_loss(y_true, y_pred),
            r_loss=label_ranking_loss(y_true, y_score),
            mAP=average_precision_score(y_true, y_score),
        )

    return meter


# In[22]:


def eval_epoch_vanilla(test_loader, optimizer, model, show_progress=False):
    # global test_loader, optimizer, model
    meter = AverageMeter()
    model.eval()
    
    for image, y_true in tqdm(test_loader, leave=False, disable=not show_progress):
        image = image.cuda() # (batch_size, 3, H, W)
        y_true = y_true.cuda() # (batch_size, num_classes)
        batch_size, num_classes = y_true.shape

        y_logit, embed = model(image) # (batch_size, num_classes)
        y_score = torch.sigmoid(y_logit)
        y_pred = y_score > 0.5

        loss = asymmetric_loss(y_logit, y_true)
        # loss = binary_cross_entropy_with_logits(y_logit, y_true)

        # record
        meter.update(
            loss=loss.item(), 
            h_loss=hamming_loss(y_true, y_pred),
            r_loss=label_ranking_loss(y_true, y_score),
            mAP=average_precision_score(y_true, y_score),
        )

    return meter


# ### MulCon

# In[23]:


def train_epoch_mulcon(
    train_loader, optimizer, model, 
    embed_queue, y_true_queue, idx_queue, size_queue, 
    show_progress=False, weight_contrastive=10
):
    # global train_loader, optimizer, model
    # global embed_queue, y_true_queue, idx_queue, size_queue
    meter = AverageMeter()
    model.train()
    
    for image, y_true in tqdm(train_loader, leave=False, disable=not show_progress):
        image = image.cuda() # (batch_size, 3, H, W)
        y_true = y_true.cuda() # (batch_size, num_classes)
        batch_size, num_classes = y_true.shape

        y_logit, embed = model(image) # (batch_size, num_classes), (batch_size, num_classes, dim_embedding)
        dim_embedding = embed.shape[-1]
        
        y_score = torch.sigmoid(y_logit)
        y_pred = y_score > 0.5

        loss_clf = asymmetric_loss(y_logit, y_true)
        
        loss_con = 0.0
        for c in range(num_classes):
            embed_one_class = embed[:, c, :]
            embed_queue_one_class = embed_queue[:, c, :]
            label_one_class = y_true[:, c]
            label_queue_one_class = y_true_queue[:, c]
            loss_con += sup_con_loss2(
                embed_one_class, embed_queue_one_class, 
                label_one_class, label_queue_one_class
            )
        loss_con /= num_classes
        
        loss = 1.0 * loss_clf + weight_contrastive * loss_con
        
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update queue
        with torch.no_grad():
            embed_queue[idx_queue: idx_queue + batch_size, :, :] = embed.detach()
            y_true_queue[idx_queue: idx_queue + batch_size, :] = y_true.detach()
            idx_queue = (idx_queue + batch_size) % size_queue
        
        # record
        meter.update(
            loss_clf=loss_clf.item(),
            loss_con=loss_con.item(),
            loss=loss.item(), 
            h_loss=hamming_loss(y_true, y_pred),
            r_loss=label_ranking_loss(y_true, y_score),
            mAP=average_precision_score(y_true, y_score),
        )

    return meter


# In[24]:


def eval_epoch_mulcon(
    test_loader, optimizer, model,
    embed_queue, y_true_queue, idx_queue, size_queue, 
    show_progress=False, weight_contrastive=10
):
    # global test_loader, optimizer, model
    # global embed_queue, y_true_queue, idx_queue, size_queue
    meter = AverageMeter()
    model.eval()
    
    for image, y_true in tqdm(test_loader, leave=False, disable=not show_progress):
        image = image.cuda() # (batch_size, 3, H, W)
        y_true = y_true.cuda() # (batch_size, num_classes)
        batch_size, num_classes = y_true.shape

        y_logit, embed = model(image) # (batch_size, num_classes)
        dim_embedding = embed.shape[-1]
        
        y_score = torch.sigmoid(y_logit)
        y_pred = y_score > 0.5

        loss_clf = asymmetric_loss(y_logit, y_true)
        
        loss_con = 0.0
        for c in range(num_classes):
            embed_one_class = embed[:, c, :]
            embed_queue_one_class = embed_queue[:, c, :]
            label_one_class = y_true[:, c]
            label_queue_one_class = y_true_queue[:, c]
            
            loss_con += sup_con_loss2(
                embed_one_class, embed_queue_one_class, 
                label_one_class, label_queue_one_class
            )
        loss_con /= num_classes
        
        loss = 1.0 * loss_clf + weight_contrastive * loss_con

        # record
        meter.update(
            loss_clf=loss_clf.item(),
            loss_con=loss_con.item(),
            loss=loss.item(), 
            h_loss=hamming_loss(y_true, y_pred),
            r_loss=label_ranking_loss(y_true, y_score),
            mAP=average_precision_score(y_true, y_score),
        )

    return meter


# ## Begin

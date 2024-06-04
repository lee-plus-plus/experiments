import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
import torch
import torchvision
from os.path import join
import torchvision.transforms as transforms


class StrongRandomAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = self.get_augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = self.Cutout(img, cutout_val)  # for fixmatch
        return img

    def get_augment_list(self):
        l = [
            (self.AutoContrast, 0, 1),
            (self.Brightness, 0.05, 0.95),
            (self.Color, 0.05, 0.95),
            (self.Contrast, 0.05, 0.95),
            (self.Equalize, 0, 1),
            (self.Identity, 0, 1),
            (self.Posterize, 4, 8),
            (self.Rotate, -30, 30),
            (self.Sharpness, 0.05, 0.95),
            (self.ShearX, -0.3, 0.3),
            (self.ShearY, -0.3, 0.3),
            (self.Solarize, 0, 256),
            (self.TranslateX, -0.3, 0.3),
            (self.TranslateY, -0.3, 0.3)
        ]
        return l

    def AutoContrast(self, img, _):
        return PIL.ImageOps.autocontrast(img)

    def Brightness(self, img, v):
        assert v >= 0.0
        return PIL.ImageEnhance.Brightness(img).enhance(v)

    def Color(self, img, v):
        assert v >= 0.0
        return PIL.ImageEnhance.Color(img).enhance(v)

    def Contrast(self, img, v):
        assert v >= 0.0
        return PIL.ImageEnhance.Contrast(img).enhance(v)

    def Equalize(self, img, _):
        return PIL.ImageOps.equalize(img)

    def Invert(self, img, _):
        return PIL.ImageOps.invert(img)

    def Identity(self, img, v):
        return img

    def Posterize(self, img, v):  # [4, 8]
        v = int(v)
        v = max(1, v)
        return PIL.ImageOps.posterize(img, v)

    def Rotate(self, img, v):  # [-30, 30]
        #assert -30 <= v <= 30
        # if random.random() > 0.5:
        #    v = -v
        return img.rotate(v)

    def Sharpness(self, img, v):  # [0.1,1.9]
        assert v >= 0.0
        return PIL.ImageEnhance.Sharpness(img).enhance(v)

    def ShearX(self, img, v):  # [-0.3, 0.3]
        #assert -0.3 <= v <= 0.3
        # if random.random() > 0.5:
        #    v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

    def ShearY(self, img, v):  # [-0.3, 0.3]
        #assert -0.3 <= v <= 0.3
        # if random.random() > 0.5:
        #    v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

    def TranslateX(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        #assert -0.3 <= v <= 0.3
        # if random.random() > 0.5:
        #    v = -v
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

    # [-150, 150] => percentage: [-0.45, 0.45]
    def TranslateXabs(self, img, v):
        #assert v >= 0.0
        # if random.random() > 0.5:
        #    v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

    def TranslateY(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        #assert -0.3 <= v <= 0.3
        # if random.random() > 0.5:
        #    v = -v
        v = v * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

    # [-150, 150] => percentage: [-0.45, 0.45]
    def TranslateYabs(self, img, v):
        #assert 0 <= v
        # if random.random() > 0.5:
        #    v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

    def Solarize(self, img, v):  # [0, 256]
        assert 0 <= v <= 256
        return PIL.ImageOps.solarize(img, v)

    def CutoutAbs(self, img, v):  # [0, 60] => percentage: [0, 0.2]
        # assert 0 <= v <= 20
        if v < 0:
            return img
        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img

    # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    def Cutout(self, img, v):
        assert 0.0 <= v <= 0.5
        if v <= 0.:
            return img

        v = v * img.size[0]
        return self.CutoutAbs(img, v)


def get_transform(img_size=224, name='basic'):
    if name == 'basic':
        t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
            transforms.ToTensor(),
            transforms.Normalize((0.2443, 0.2394, 0.2425),
                                 (0.2443, 0.2394, 0.2425))
        ])

    elif name == 'weak':
        t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.2443, 0.2394, 0.2425),
                                 (0.2443, 0.2394, 0.2425))
        ])

    elif name == 'strong':
        t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            StrongRandomAugment(n=3, m=5),
            transforms.ToTensor(),
            transforms.Normalize((0.2443, 0.2394, 0.2425),
                                 (0.2443, 0.2394, 0.2425))
        ])

    else:
        raise ValueError("unsupported transform name ‘{}’".format(name))

    return t

# ## Dataset


@torch.no_grad()
def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K)
    transition_matrix[np.where(
        ~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
    # print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy(
            (random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    # print("Finish Generating Candidate Label Sets!\n")
    return partialY


def split_trainval_dataset(dataset, split_ratio=[0.5, 0.5], shuffle=True, seed=None):
    indices = list(range(len(dataset)))
    split_ratio = (np.cumsum([0] + split_ratio) * len(dataset)).astype(int)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    splited_indices = [indices[begin: end]
                       for begin, end in zip(split_ratio[:-1], split_ratio[1:])]
    splited_sampler = [torch.utils.data.SubsetRandomSampler(
        idxs) for idxs in splited_indices]

    return splited_sampler


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root='/hdd/lzm/dataset/coco/', partial_rate=0.1, img_size=224):
        print("loading dataset ...")
        base_dataset = torchvision.datasets.CocoDetection(
            root=join(root, f'train2017'),
            annFile=join(root, f'annotations/instances_train2017.json')
        )

        # many unlabeled samples are included, filter them out
        targets = [base_dataset._load_target(
            id) for id in range(len(base_dataset))]
        labels = [sorted(list({seg_box['category_id']
                               for seg_box in elem})) for elem in targets]
        available_idxs, labels = zip(*[
            (i, lbl) for i, lbl in enumerate(labels) if len(lbl) > 0
        ])

        # some label value is not used, re-mapping them (labels -> y_indexs)
        lbl2idx = {lbl: idx for idx, lbl in enumerate(set(sum(labels, [])))}
        y_indexs = [[lbl2idx[lbl] for lbl in elem] for elem in labels]

        # generate ground-truth multi-label target (Y_true)
        num_samples, dim_labels = len(labels), len(lbl2idx)
        Y_true = torch.zeros(num_samples, dim_labels, requires_grad=False)
        for row in range(num_samples):
            Y_true[row, y_indexs[row]] = 1

        # generate candidate label set (Y_candidate)
        Y_candidate = Y_true.clone()
        Y_candidate[torch.rand_like(Y_candidate) < partial_rate] = 1

        self.base_dataset = base_dataset
        self.available_idxs = available_idxs
        self.Y_true = Y_true
        self.Y_candidate = Y_candidate
        self.dim_labels = dim_labels

        self.basic_transform = get_transform(img_size=img_size, name='basic')
        self.weak_transform = get_transform(img_size=img_size, name='weak')
        self.strong_transform = get_transform(img_size=img_size, name='strong')

    def _get_mean_and_std(self, images):
        to_tensor = torchvision.transforms.ToTensor()
        mean_, std_ = [], []

        for image in images:
            image = to_tensor(image)
            mean_.append(image.mean(dim=[1, 2]))
            std_.append(image.std(dim=[1, 2]))

        mean_ = torch.vstack(mean_).mean(dim=0)
        std_ = torch.vstack(std_).mean(dim=0)
        return std_, std_

    def __len__(self):
        return len(self.available_idxs)

    @torch.no_grad()
    def __getitem__(self, index):
        image_index = self.available_idxs[index]
        image = self.base_dataset._load_image(image_index)
        image_basic = self.basic_transform(image)
        image_weak_aug = self.weak_transform(image)
        image_strong_aug = self.strong_transform(image)

        y_true = self.Y_true[index, :]
        y_candidate = self.Y_candidate[index, :]

        return index, image_basic, image_weak_aug, image_strong_aug, y_true, y_candidate

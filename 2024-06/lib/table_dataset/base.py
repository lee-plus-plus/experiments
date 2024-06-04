import numpy as np
import scipy.sparse as sp
import re
import pickle
import arff
from scipy.io import loadmat, savemat
from dataclasses import dataclass
from typing import Union, List
from .convert import to_ndarray
from .convert import to_tensor
from torch.utils.data import Dataset


@dataclass
class StorableTableMllDataset:
    features: Union[sp.spmatrix, np.ndarray]
    labels: Union[sp.spmatrix, np.ndarray]
    feature_names: List[str] = None
    label_names: List[str] = None

    def __post_init__(self):
        assert isinstance(self.features, (sp.spmatrix, np.ndarray))
        assert isinstance(self.labels, (sp.spmatrix, np.ndarray))

        self.feature_names = self.feature_names or \
            [f'Attr{i}' for i in range(1, self.features.shape[1] + 1)]
        self.label_names = self.label_names or \
            [f'Class{j}' for j in range(1, self.labels.shape[1] + 1)]

        assert self.features.shape[0] == self.labels.shape[0]
        assert self.features.shape[1] == len(self.feature_names)
        assert self.labels.shape[1] == len(self.label_names)

        if isinstance(self.features, np.ndarray) or \
           isinstance(self.labels, np.ndarray):
            self.features = to_ndarray(self.features)
            self.labels = to_ndarray(self.labels)
        assert type(self.features) == type(self.labels)

    def __add__(self, dataset: 'StorableTableMllDataset') -> 'StorableTableMllDataset':
        assert (self.feature_names == dataset.feature_names) & \
               (self.label_names == self.label_names)

        vstack = sp.vstack if self.is_sparse else np.vstack
        features = vstack([self.features, dataset.features])
        labels = vstack([self.labels, dataset.labels])

        return StorableTableMllDataset(
            features, labels,
            self.feature_names, self.label_names)

    def to_dense(self):
        return StorableTableMllDataset(
            to_ndarray(self.features), to_ndarray(self.labels), 
            self.feature_names, self.label_names)

    @property
    def num_samples(self):
        return self.features.shape[0]

    @property
    def dim_features(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return self.labels.shape[1]

    @property
    def is_sparse(self):
        return isinstance(self.features, sp.spmatrix)

    @classmethod
    def from_arff(cls, filename: str) -> 'StorableTableMllDataset':
        with open(filename, 'r') as file:
            arff_frame = arff.load(file, encode_nominal=True)
        assert {'relation', 'attributes', 'data'}.issubset(arff_frame)

        # get meta infomation
        _, name, info = re.split('(.+): (.*)', arff_frame['relation'])[:3]
        info_args = info.split()
        info = {key: value for key, value in zip(
            info_args[0::2], info_args[1::2])}

        assert {'-m', '-d', '-q', '-label_location',
                '-is_sparse'}.issubset(info)
        num_samples, dim_features, num_classes = int(
            info['-m']), int(info['-d']), int(info['-q'])
        label_location = info['-label_location']
        assert label_location in ['begin', 'end']

        data = np.array(arff_frame['data'])  # arff.load guarantee to be dense
        attr_names = [name for name, value_domain in arff_frame['attributes']]
        assert data.shape == (num_samples, dim_features + num_classes)

        if label_location == "begin":
            labels = data[:, :num_classes]
            features = data[:, num_classes:]
            label_names = attr_names[:num_classes]
            feature_names = attr_names[num_classes:]
        elif label_location == "end":
            features = data[:, :num_classes]
            labels = data[:, num_classes:]
            feature_names = attr_names[:num_classes]
            label_names = attr_names[num_classes:]

        return cls(
            features=features,
            labels=labels,
            feature_names=feature_names,
            label_names=label_names,
        )

    @classmethod
    def from_mat(cls, filename: str) -> 'StorableTableMllDataset':
        data_dict = loadmat(filename)
        return cls(
            features=data_dict['data'],
            labels=data_dict['target'].T,
            feature_names=data_dict['feature_names'],
            label_names=data_dict['label_names'],
        )

    @classmethod
    def from_pickle(cls, filename: str) -> 'StorableTableMllDataset':
        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
        return cls(
            features=data_dict['data'],
            labels=data_dict['target'],
            feature_names=data_dict['feature_names'],
            label_names=data_dict['label_names'],
        )

    def to_arff(self, filename: str):
        relation = (f'data: -m {self.num_samples} -d {self.dim_features} '
                    f'-q {self.num_samples} -label_location {"end"} '
                    f'-is_sparse {self.is_sparse}')

        feature_attrs = [(elem, 'NUMERIC') for elem in self.feature_names]
        label_attrs = [(elem, ['0', '1']) for elem in self.label_names]
        attributes = feature_attrs + label_attrs
        data = (sp.hstack if self.is_sparse else np.hstack)(
            [self.features, self.labels])

        arff_data = arff.dumps({
            u'relation': relation,
            u'attributes': attributes,
            u'data': data
        })
        with open(filename, 'w') as file:
            file.write(arff_data)

    def to_mat(self, filename: str):
        data_dict = {
            'data': self.features,
            'target': self.labels.T,
            'feature_names': self.feature_names,
            'label_names': self.label_names
        }
        savemat(filename, data_dict)

    def to_pickle(self, filename: str):
        data_dict = {
            'data': self.features,
            'target': self.labels,
            'feature_names': self.feature_names,
            'label_names': self.label_names
        }
        with open(filename, 'wb') as file:
            pickle.dump(data_dict, file)


@dataclass
class StorableTablePmlDataset:
    features: Union[sp.spmatrix, np.ndarray]
    labels: Union[sp.spmatrix, np.ndarray]
    partial_labels: Union[sp.spmatrix, np.ndarray]
    feature_names: List[str]
    label_names: List[str]

    def __post_init__(self):
        assert isinstance(self.features, (sp.spmatrix, np.ndarray))
        assert isinstance(self.labels, (sp.spmatrix, np.ndarray))
        assert isinstance(self.partial_labels, (sp.spmatrix, np.ndarray))

        self.feature_names = self.feature_names or [
            f'Attr{i}' for i in range(1, self.features.shape[1] + 1)]
        self.label_names = self.label_names or [
            f'Class{j}' for j in range(1, self.labels.shape[1] + 1)]

        assert isinstance(self.feature_names, (list, tuple))
        assert isinstance(self.label_names, (list, tuple))
        assert self.features.shape[0] == self.labels.shape[0]
        assert self.features.shape[1] == len(self.feature_names)
        assert self.labels.shape[1] == len(self.label_names)
        assert self.partial_labels.shape == self.labels.shape

        if isinstance(self.features, np.ndarray) or \
           isinstance(self.labels, np.ndarray) or \
           isinstance(self.partial_labels, np.ndarray):
            self.features = to_ndarray(self.features)
            self.labels = to_ndarray(self.labels)
            self.partial_labels = to_ndarray(self.partial_labels)
        assert type(self.features) == type(
            self.labels) == type(self.partial_labels)

    def __add__(self, dataset: 'StorableTablePmlDataset') -> 'StorableTablePmlDataset':
        assert (self.feature_names == dataset.feature_names) & \
               (self.label_names == self.label_names)

        vstack = sp.vstack if self.is_sparse else np.vstack
        features = vstack([self.features, dataset.features])
        labels = vstack([self.labels, dataset.labels])
        partial_labels = vstack([self.partial_labels, dataset.partial_labels])

        return StorableTablePmlDataset(
            features, labels, partial_labels,
            self.feature_names, self.label_names)

    def to_dense(self):
        return StorableTablePmlDataset(
            to_ndarray(self.features), to_ndarray(self.labels), 
            to_ndarray(self.partial_labels), self.feature_names, self.label_names)

    @property
    def num_samples(self):
        return self.features.shape[0]

    @property
    def dim_features(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return self.labels.shape[1]

    @property
    def is_sparse(self):
        return isinstance(self.features, sp.spmatrix)

    @classmethod
    def from_mat(cls, filename: str) -> 'StorableTablePmlDataset':
        data_dict = loadmat(filename)
        return cls(
            features=data_dict['data'],
            labels=data_dict['target'].T,
            partial_labels=data_dict['candidate_labels'].T,
            feature_names=data_dict.get('feature_names'),
            label_names=data_dict.get('label_names'),
        )

    @classmethod
    def from_pickle(cls, filename: str) -> 'StorableTablePmlDataset':
        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
        return cls(
            features=data_dict['data'],
            labels=data_dict['target'],
            partial_labels=data_dict['candidate_labels'],
            feature_names=data_dict.get('feature_names'),
            label_names=data_dict.get('label_names'),
        )

    @classmethod
    def to_mat(self, filename: str):
        data_dict = {
            'data': self.features,
            'target': self.label.T,
            'candidate_labels': self.partial_labels.T,
            'feature_names': self.feature_names,
            'label_names': self.label_names
        }
        savemat(filename, data_dict)

    @classmethod
    def to_pickle(self, filename: str):
        data_dict = {
            'data': self.features,
            'target': self.labels,
            'candidate_labels': self.partial_labels,
            'feature_names': self.feature_names,
            'label_names': self.label_names
        }
        with open(filename, 'wb') as file:
            pickle.dump(data_dict, file)


class TableMllDataset(Dataset):
    def __init__(self, features, labels, category_name=None):
        if category_name is None:
            category_name = {i: str(i) for i in range(self.num_classes)}
        elif isinstance(category_name, list):
            category_name = dict(zip(range(labels.shape[1]), category_name))

        self.features = to_tensor(features).float()
        self.labels = to_tensor(labels).int()
        self.category_name = category_name

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = self.features[index, :]
        y = self.labels[index, :]
        return x, y

    def __repr__(self):
        return f'TableMllDataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'

    @property
    def num_samples(self):
        return self.labels.shape[0]

    @property
    def dim_features(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return self.labels.shape[1]

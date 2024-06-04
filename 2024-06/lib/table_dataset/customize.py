import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from .utils import add_partial_noise
from .convert import to_tensor
from typing import Type, Optional
import abc


# abstract class
class IndexedDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class PartiallyNoisyDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class GetItemsDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class FeatureScaledDataset(metaclass=abc.ABCMeta):
    pass


def indexed_dataset_class(
    dataset_class: Type[Dataset],
) -> Type[Dataset]:
    '''let the dataset iterate with index
    '''
    class _IndexedDataset(dataset_class, IndexedDataset):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            return result + (index,)

    return _IndexedDataset


def partially_noisy_dataset_class(
    dataset_class: Type[Dataset],
    noise_rate: float,
    noise_override: bool = False,
    partial_labels: Optional[torch.Tensor] = None,
) -> Type[Dataset]:
    '''let the dataset iterate with partial labels,
    if partial labels are given, use it, otherwise,
    generate uniformed partial noise
    '''
    class _PartiallyNoisyDataset(dataset_class, PartiallyNoisyDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)
            if partial_labels is not None:
                self.partial_labels = to_tensor(partial_labels).int()
            if (partial_labels is None) or noise_override:
                self.partial_labels = add_partial_noise(
                    self.labels, noise_rate)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            return result + (self.partial_labels[index],)

    return _PartiallyNoisyDataset


def getitems_dataset_class(dataset_class: Type[Dataset]) -> Type[Dataset]:
    '''let the dataset support __getitems__
    '''
    class _GetItemsDataset(dataset_class, GetItemsDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)

        def __getitems__(self, indices):
            return self.__getitem__(indices)

    return _GetItemsDataset


def feature_scaled_dataset_class(
    dataset_class: Type[Dataset],
    scaler: Optional[StandardScaler] = None,
) -> Type[Dataset]:
    '''standardize the feature of dataset,
    use provided scaler to standardize
    '''
    class _FeatureScaledDataset(dataset_class, FeatureScaledDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)
            self.features = torch.from_numpy(
                scaler.transform(self.features.numpy())).float()

    return _FeatureScaledDataset


def customize(
    dataset_class: Type[Dataset], *,
    add_index: bool = False,
    add_partial_noise: bool = False,
    noise_rate: float = 0.0,
    noise_override: bool = False,
    partial_labels: Optional[torch.Tensor] = None,
    getitems: bool = False,
    scale: bool = False,
    scaler: Optional[StandardScaler] = None,
):
    if add_partial_noise:
        dataset_class = partially_noisy_dataset_class(
            dataset_class, noise_rate, noise_override, partial_labels)
    if add_index:
        dataset_class = indexed_dataset_class(dataset_class)
    if getitems:
        dataset_class = getitems_dataset_class(dataset_class)
    if scale:
        dataset_class = feature_scaled_dataset_class(
            dataset_class, scaler)

    return dataset_class

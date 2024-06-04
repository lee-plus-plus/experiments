from sklearn.preprocessing import StandardScaler
from os.path import expanduser, basename, join
from .mll import supported_mll_datasets, load_mll_data
from .pml import supported_pml_datasets, load_pml_data
from .convert import to_tensor
from .base import TableMllDataset
from .customize import customize
from ..utils import ignore_warnings


def supported_datasets():
    return list(supported_pml_datasets()) + list(supported_mll_datasets())


@ignore_warnings
def build_dataset(
    name, 
    base=expanduser('~/table_dataset'), **kwargs,
):
    if name in supported_pml_datasets():
        train_data = load_pml_data(
            name, 'train', base=join(base, 'pml')).to_dense()
        test_data = load_pml_data(
            name, 'test', base=join(base, 'pml')).to_dense()

        Yp_train = to_tensor(train_data.partial_labels).int()
        Yp_test = to_tensor(test_data.partial_labels).int()

    elif name in supported_mll_datasets():
        train_data = load_mll_data(
            name, 'train', base=join(base, 'mll')).to_dense()
        test_data = load_mll_data(
            name, 'test', base=join(base, 'mll')).to_dense()

        Yp_train = None
        Yp_test = None

    else:
        raise ValueError(f'unsupported dataset name ‘{name}’')

    # standardization
    scaler = StandardScaler().fit(train_data.features)

    def _customize(dataset_class, **kwargs):
        return customize(dataset_class, **kwargs)

    train_dataset = _customize(
        TableMllDataset, partial_labels=Yp_train, scaler=scaler, **kwargs)(
        train_data.features, train_data.labels, train_data.label_names)
    test_dataset = _customize(
        TableMllDataset, partial_labels=Yp_test, scaler=scaler, **kwargs)(
        test_data.features, test_data.labels, test_data.label_names)

    return train_dataset, test_dataset

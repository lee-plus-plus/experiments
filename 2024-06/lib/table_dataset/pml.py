import re
from os.path import basename, join, expanduser
from glob import glob
from .base import StorableTablePmlDataset, TableMllDataset


def supported_pml_dataset_divide_pairs(base=expanduser('~/table_dataset/pml')):
    filenames = glob(join(base, '*', '*-*.mat'))
    name_divide_pairs = [tuple(re.split(r'(.+)\-(.+)\.mat',
                                        basename(filename))[1:3])
                         for filename in filenames]
    name_divide_pairs = sorted(list(set(name_divide_pairs)))
    return name_divide_pairs


def supported_pml_datasets(base=expanduser('~/table_dataset/pml')):
    return sorted(list(set([
        name for name, divide in supported_pml_dataset_divide_pairs()])))


def supported_pml_divides(base=expanduser('~/table_dataset/pml')):
    return sorted(list(set([
        divide for name, divide in supported_pml_dataset_divide_pairs()])))


def load_pml_data(set_name, divide, base=expanduser('~/table_dataset/pml')):
    filename = join(base, f'{set_name}/{set_name}-{divide}.pickle')
    return StorableTablePmlDataset.from_pickle(filename)

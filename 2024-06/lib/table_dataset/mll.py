import re
from os.path import expanduser, basename, join
from glob import glob
from .base import StorableTableMllDataset, TableMllDataset


def supported_mll_dataset_divide_pairs(base=expanduser('~/table_dataset/mll')):
    filenames = glob(join(base, '*', '*.arff'))
    name_divide_pairs = [tuple(re.split(r'(.+)\-(.+)\.arff',
                                        basename(filename))[1:3])
                         for filename in filenames]
    name_divide_pairs = sorted(list(set(name_divide_pairs)))
    return name_divide_pairs


def supported_mll_datasets(base=expanduser('~/table_dataset/mll')):
    return sorted(list(set([
        name for name, divide in supported_mll_dataset_divide_pairs()])))


def supported_mll_divides(base=expanduser('~/table_dataset/mll')):
    return sorted(list(set([
        divide for name, divide in supported_mll_dataset_divide_pairs()])))


def load_mll_data(set_name, divide, base=expanduser('~/table_dataset/mll')):
    filename = join(base, f'{set_name}', f'{set_name}-{divide}.pickle')
    return StorableTableMllDataset.from_pickle(filename)



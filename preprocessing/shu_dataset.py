import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
import json
from pathlib import Path
from functools import lru_cache
from collections import defaultdict


@lru_cache(maxsize=None)
def _shared_environment(data_dir):
    """Reuse one LMDB connection per path in this notebook process.

    The DataLoaders below use num_workers=0 (PyTorch's default).
    Restart the kernel after editing/reloading this module or rebuilding the data.
    """
    return lmdb.open(data_dir, readonly=True, lock=False,
                     readahead=True, meminit=False)


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = _shared_environment(os.path.realpath(data_dir))
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        # print(label)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


def select_training_keys(dataset, fraction, seed):
    """Deterministic nested subsets with approximately preserved session/class ratios."""
    if not 0 < fraction <= 1:
        raise ValueError('train_fraction must be in (0, 1].')
    if fraction == 1:
        return list(dataset.keys)
    groups = defaultdict(list)
    with dataset.db.begin(write=False) as txn:
        for key in dataset.keys:
            label = int(pickle.loads(txn.get(key.encode()))['label'])
            session = key.rsplit('-', 1)[0]
            groups[(session, label)].append(key)
    rng = np.random.default_rng(seed)
    ranked = []
    for group, keys in sorted(groups.items()):
        shuffled = rng.permutation(keys)
        for i, key in enumerate(shuffled):
            ranked.append(((i + 0.5) / len(keys), group, key))
    ranked.sort()
    count = int(len(dataset) * fraction)
    if count < 1:
        raise ValueError('The requested fraction selects no training samples.')
    selected = {item[2] for item in ranked[:count]}
    return [key for key in dataset.keys if key in selected]


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        # Restrict only the training split; validation/test retain all original keys.
        manifest_path = getattr(self.params, 'train_keys', None)
        fraction = getattr(self.params, 'train_fraction', 1.0)
        if manifest_path and fraction != 1.0:
            raise ValueError('Use either train_keys or train_fraction, not both.')
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text())
            keys = manifest['keys']
            if not keys or len(keys) != len(set(keys)) or not set(keys) <= set(train_set.keys):
                raise ValueError('Training manifest must contain unique keys from the training split.')
            train_set.keys = keys
        else:
            train_set.keys = select_training_keys(train_set, fraction, getattr(self.params, 'seed', 3407))
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader

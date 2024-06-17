import os
import torch
from torch.utils.data import Dataset, DataLoader
import lmdb
import pickle
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import string


class LMDBDataset(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path), readonly=True, lock=False, readahead=False,
                             meminit=False, map_size=1024 * 1024 * 1024 * 1024)

    def __len__(self):
        with self.env.begin(write=False) as txn:
            return txn.stat()['entries']

    def __getitem__(self, index):
        with self.env.begin(write=False, buffers=True) as txn:
            key = f'{index:08}'.encode('ascii')
            data = txn.get(key)
            image, target, input_length, label_length = pickle.loads(data)
            return image, target, input_length, label_length


def create_lmdb_dataset(dataset, db_path):
    env = lmdb.open(db_path, subdir=os.path.isdir(db_path), map_size=1024 * 1024 * 1024 * 1024, readonly=False,
                    meminit=False, map_async=True)
    txn = env.begin(write=True)
    for idx, (image, target, input_length, label_length) in enumerate(dataset):
        data = pickle.dumps((image, target, input_length, label_length))
        key = f'{idx:08}'.encode('ascii')
        txn.put(key, data)
    txn.commit()
    env.close()


if __name__ == '__main__':
    from NamedFilesDataset import NamedFilesDataset

    characters = string.digits + string.ascii_lowercase
    width, height, n_len, n_classes = 192, 64, 6, len(characters)
    n_input_length = 12
    __dataset_basepath = os.path.join(os.path.dirname(__file__), "..", '..', 'data', 'synthetic')

    # Create dataset
    train_set = NamedFilesDataset(os.path.join(__dataset_basepath, "train"), characters, width, height, n_input_length, n_len)
    valid_set = NamedFilesDataset(os.path.join(__dataset_basepath, "valid"), characters, width, height, n_input_length,
                                  n_len)

    # Create LMDB files
    train_db_path = os.path.join(__dataset_basepath, "train.lmdb")
    valid_db_path = os.path.join(__dataset_basepath, "valid.lmdb")

    create_lmdb_dataset(train_set, train_db_path)
    create_lmdb_dataset(valid_set, valid_db_path)

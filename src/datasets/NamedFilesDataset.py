import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
import random
import string
from PIL import Image


class NamedFilesDataset(Dataset):
    def __init__(self,
                 dataset_dirpath: str,
                 characters: list[str] | str,
                 width: int,
                 height: int,
                 input_length: int,
                 label_length: int):
        super(NamedFilesDataset, self).__init__()
        self.dataset_dirpath = dataset_dirpath
        self.files = os.listdir(dataset_dirpath)
        self.characters = characters
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        image = to_tensor(Image.open(os.path.join(self.dataset_dirpath, filename)).resize((self.width, self.height)))
        label = os.path.splitext(filename)[0]
        target = torch.tensor([self.characters.find(x) for x in label], dtype=torch.long)
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length


if __name__ == '__main__':
    characters = string.digits + string.ascii_lowercase
    width, height, n_len, n_classes = 192, 64, 6, len(characters)
    n_input_length = 12
    __dataset_basepath = os.path.join(os.path.dirname(__file__), "..", '..', 'data', 'dataset')
    dataset = NamedFilesDataset(characters=characters,
                                dataset_dirpath=os.path.join(__dataset_basepath, "train"),
                                width=width,
                                height=height,
                                input_length=n_input_length,
                                label_length=n_len)
    image, target, input_length, label_length = dataset[0]
    print(''.join([characters[x] for x in target]), input_length, label_length)
    pic = to_pil_image(image)
    pic.show()

    batch_size = 128
    train_set = NamedFilesDataset(characters=characters,
                                  dataset_dirpath=os.path.join(__dataset_basepath, "train"),
                                  width=width,
                                  height=height,
                                  input_length=n_input_length,
                                  label_length=n_len)
    valid_set = NamedFilesDataset(characters=characters,
                                  dataset_dirpath=os.path.join(__dataset_basepath, "valid"),
                                  width=width,
                                  height=height,
                                  input_length=n_input_length,
                                  label_length=n_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
import random
import string


class GeneratedCaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(GeneratedCaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        image = to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length


if __name__ == '__main__':
    characters = '-' + string.digits + string.ascii_uppercase
    width, height, n_len, n_classes = 192, 64, 4, len(characters)
    n_input_length = 12
    dataset = GeneratedCaptchaDataset(characters=characters, length=1, width=width, height=height,
                                      input_length=n_input_length,
                                      label_length=n_len)
    image, target, input_length, label_length = dataset[0]
    print(''.join([characters[x] for x in target]), input_length, label_length)
    pic = to_pil_image(image)
    pic.show()

    batch_size = 128
    train_set = GeneratedCaptchaDataset(characters, 1000 * batch_size, width, height, n_input_length, n_len)
    valid_set = GeneratedCaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())

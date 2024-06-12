import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from CaptchaDataset import CaptchaDataset, decode, decode_target, characters


class Network(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Network, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


def train(model, optimizer, epochs, dataloader, device: torch.device):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc = calc_acc(target, output)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            pbar.set_description(f'Epoch: {epochs} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


def valid(model, epochs, dataloader, device: torch.device):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss = loss.item()
            acc = calc_acc(target, output)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pbar.set_description(f'Test : {epochs} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


class ModelTrainer:
    def __init__(self,
                 class_count: int,
                 input_shape: tuple[int, int, int],
                 device: torch.device,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 epochs: int = 30,
                 ):
        self.class_count = class_count
        self.input_shape = input_shape
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs

    def optimise(self, learning_rate: float, epochs: int):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)
        for epoch in range(1, epochs + 1):
            train(model=self.model, optimizer=optimizer, epochs=epochs, dataloader=self.train_dataloader,
                  device=self.device)
            valid(self.model, epochs=epochs, dataloader=self.valid_dataloader, device=self.device)

    def train(self):
        self.model = Network(self.class_count, input_shape=self.input_shape)
        self.model.to(self.device)

        # Train and optimise model on provided no of epochs
        self.optimise(learning_rate=1e-3, epochs=self.epochs)
        self.optimise(learning_rate=1e-4, epochs=int(self.epochs / 2))

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)


if __name__ == '__main__':
    width, height, n_len, n_classes = 192, 64, 4, len(characters)
    n_input_length = 12
    dataset = CaptchaDataset(characters=characters, length=1, width=width, height=height, input_length=n_input_length,
                             label_length=n_len)

    batch_size = 128
    train_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
    valid_set = CaptchaDataset(characters, 50 * batch_size, width, height, n_input_length, n_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())

    # start model training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ModelTrainer(
        class_count=len(characters),
        input_shape=(3, 64, 192),
        device=device,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        epochs=30
    )
    model.train()
    model.save_model("data/models/captcha.pt")

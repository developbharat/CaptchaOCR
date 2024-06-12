import os.path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from common import Network, decode_target, decode
import numpy as np
import torch.nn.functional as F


def calc_acc(target, output, classes):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true, classes) == decode(pred, classes) for true, pred in zip(target, output_argmax)])
    return a.mean()


def train(model, optimizer, epochs, dataloader, device: torch.device, classes: list[str]):
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
            acc = calc_acc(target, output, classes)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            pbar.set_description(f'Epoch: {epochs} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


def valid(model, epochs, dataloader, device: torch.device, classes: list[str]):
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
            acc = calc_acc(target, output, classes)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pbar.set_description(f'Test : {epochs} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


class ModelTrainer:
    def __init__(self,
                 classes: list[str] | str,
                 input_shape: tuple[int, int, int],
                 device: torch.device,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 epochs: int = 30,
                 checkpoint: str | None = None
                 ):
        self.classes = classes
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs
        self.model = Network(class_count=len(classes), input_shape=input_shape)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))

    def optimise(self, learning_rate: float, epochs: int):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)
        for epoch in range(1, epochs + 1):
            train(model=self.model, optimizer=optimizer, epochs=epochs, dataloader=self.train_dataloader,
                  device=self.device, classes=self.classes)
            valid(self.model, epochs=epochs, dataloader=self.valid_dataloader, device=self.device, classes=self.classes)

    def train(self):
        self.model.to(self.device)

        # Train and optimise model on provided no of epochs
        self.optimise(learning_rate=1e-3, epochs=self.epochs)
        self.optimise(learning_rate=1e-4, epochs=int(self.epochs / 2))

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    import string
    from datasets.GeneratedCaptchaDataset import GeneratedCaptchaDataset

    characters = '-' + string.digits + string.ascii_uppercase
    width, height, n_len, n_classes = 192, 64, 4, len(characters)
    n_input_length = 12
    dataset = GeneratedCaptchaDataset(characters=characters, length=1, width=width, height=height,
                                      input_length=n_input_length,
                                      label_length=n_len)

    batch_size = 128
    train_set = GeneratedCaptchaDataset(characters, 10 * batch_size, width, height, n_input_length, n_len)
    valid_set = GeneratedCaptchaDataset(characters, 5 * batch_size, width, height, n_input_length, n_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())

    # start model training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ModelTrainer(
        classes=characters,
        input_shape=(192, 64, 3),
        device=device,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        checkpoint=None,
        epochs=30
    )
    model.train()
    model.save_model("data/models/captcha.pt")

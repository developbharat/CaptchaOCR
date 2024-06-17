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


def train(model, optimizer, epochs, epoch, dataloader, device: torch.device, classes: list[str]):
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

            pbar.set_description(f'Epoch: {epoch}/{epochs} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
    return loss_mean, acc_mean

def valid(model, epochs, epoch, dataloader, device: torch.device, classes: list[str]):
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

            pbar.set_description(f'Test : {epoch}/{epochs} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
    return loss_mean, acc_mean

class ModelTrainer:
    def __init__(self,
                 classes: list[str] | str,
                 input_shape: tuple[int, int, int],
                 device: torch.device,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 epochs: int = 30,
                 checkpoint: str | None = None,
                 checkpoints_dirpath: str = "./checkpoints",
                 ):
        self.classes = classes
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs
        self.best_acc = 0.0
        self.best_model = ""
        self.checkpoints_dirpath = checkpoints_dirpath
        self.model = Network(class_count=len(classes), input_shape=input_shape)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))

    def optimise(self, learning_rate: float, epochs: int):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)
        for epoch in range(1, epochs + 1):
            _, acc1 = train(model=self.model, optimizer=optimizer, epochs=epochs, epoch=epoch, dataloader=self.train_dataloader,
                  device=self.device, classes=self.classes)
            _, acc2 = valid(self.model, epochs=epochs, epoch=epoch, dataloader=self.valid_dataloader, device=self.device, classes=self.classes)

            # store most accurate model to disk after each epoch
            if self.best_acc < acc1 + acc2:
                self.best_acc = acc1 + acc2
                self.best_model = os.path.join(self.checkpoints_dirpath, f"epoch-{epoch:.4f}-acc-{self.best_acc:.4f}.pt")
                self.save_model(self.best_model)

    def train(self):
        self.model.to(self.device)

        # Train and optimise model on provided no of epochs
        self.optimise(learning_rate=1e-3, epochs=self.epochs)
        self.optimise(learning_rate=1e-4, epochs=int(self.epochs / 2))

    def save_model(self, path: str, best: bool=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if best and self.best_model != "":
            os.rename(self.best_model, path)
        else:
            torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    import string
    from datasets.NamedFilesDataset import NamedFilesDataset
    from datasets.LMDBDataset import LMDBDataset

    characters = string.digits + string.ascii_lowercase
    width, height, n_len, n_classes = 192, 64, 6, len(characters)
    n_input_length = 12
    __dataset_basepath = os.path.join(os.path.dirname(__file__), "..", 'data', 'synthetic2')
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
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=0)

    # Create dataset and data loader for LMDB files
    # __dataset_basepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    # train_db_path = os.path.join(__dataset_basepath, "train.lmdb")
    # valid_db_path = os.path.join(__dataset_basepath, "valid.lmdb")
    # train_dataset = LMDBDataset(train_db_path)
    # valid_dataset = LMDBDataset(valid_db_path)

    # batch_size = 128
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)

    # start model training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ModelTrainer(
        classes=characters,
        input_shape=(192, 64, 3),
        device=device,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        checkpoint=None,
        checkpoints_dirpath=os.path.join(os.path.dirname(__file__), "..", 'data', 'checkpoints'),
        epochs=50
    )
    model.train()

    model_save_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'models', "captcha.pt")
    model.save_model(model_save_path, best=True)

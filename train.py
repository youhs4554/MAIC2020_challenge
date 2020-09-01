import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.baseline import BasicConv1d
from datasets.maic2020 import MAIC2020
from utils.metrics import *
import tqdm

if __name__ == "__main__":
    # Prepare Dataset
    train_ds = MAIC2020(infile="data/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                        transform=None, validation=False)
    val_ds = MAIC2020(infile="data/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                      transform=None, validation=True)

    train_dataloader = DataLoader(
        train_ds, batch_size=512, num_workers=4, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(
        val_ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=torch.cuda.is_available())

    # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
    # 6-layers 1d-CNNs
    model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])

    crit = nn.BCELoss()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 10

    for ep in range(n_epochs):
        for phase, dataloader in zip(['train', 'val'], [train_dataloader, val_dataloader]):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0
            running_auprc = 0.0
            running_auc = 0.0

            dataloader = iter(dataloader)

            for _ in tqdm.tqdm(range(len(dataloader)), desc=f"{phase.capitalize()} Loop"):
                x, y_true = next(dataloader)
                y_true = y_true.view(-1, 1).float()

                if torch.cuda.is_available():
                    x = x.cuda()
                    y_true = y_true.cuda()

                y_pred = model(x)

                loss = crit(y_pred, y_true)

                acc = compute_accuracy(y_true, y_pred)
                auprc = compute_auprc(y_true, y_pred)
                auc = compute_auc(y_true, y_pred)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_acc += acc
                running_auprc += auprc
                running_auc += auc

            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = running_acc / len(train_dataloader)
            epoch_auprc = running_auprc / len(train_dataloader)
            epoch_auc = running_auc / len(train_dataloader)

            print(
                f"[EP={ep}][{phase}] ====> LOSS: {epoch_loss:.4f}, ACCURACY: {epoch_acc:.4f}, AUPRC: {epoch_auprc:.4f}, AUC: {epoch_auc:.4f}")

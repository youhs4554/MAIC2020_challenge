import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import tqdm
import random
import os

from models.baseline import BasicConv1d
from datasets.maic2020 import MAIC2020
from utils.metrics import *
from utils.losses import BCE_with_class_weights
from utils.lr_scheduler import LR_Wramer

if __name__ == "__main__":
    BATCH_SIZE = 1024 * torch.cuda.device_count()
    VALIDATION_SPLIT = 0.1
    N_EPOCHS = 10
    ZERO_CROSSING = 65.0
    EXP_HOME = "./experiments"

    verion_ix = len(os.listdir(EXP_HOME))
    EXP_SUB_DIR = os.path.join(EXP_HOME, "version-{}".format(verion_ix))
    os.system(f"mkdir -p {EXP_SUB_DIR}")

    # Prepare Dataset
    ds = MAIC2020(infile="data/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                  transform=lambda x: (x-ZERO_CROSSING)/ZERO_CROSSING)

    # random shuffle
    random.seed(0)       # fix seed for reproducitivy
    ixs = torch.randperm(len(ds))
    ds = Subset(ds, ixs)

    split_loc = int(len(ixs)*VALIDATION_SPLIT)

    ixs = torch.arange(len(ds))
    train_ds = Subset(ds, ixs[split_loc:])
    val_ds = Subset(ds, ixs[:split_loc])

    print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=torch.cuda.is_available())

    # # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
    # # 6-layers 1d-CNNs
    # model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])

    # import models.resnet1d
    # from models.non_local.nl_conv1d import NL_Conv1d
    # backbone = models.resnet1d.resnet18()
    # model = NL_Conv1d(backbone=backbone)

    from models.shufflenet import shufflenet_v2
    model = shufflenet_v2()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    crit = BCE_with_class_weights(class_weights={0: 1, 1: 1})

    if torch.cuda.is_available():
        model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.9, nesterov=True, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    warmer = LR_Wramer(optimizer, scheduler=scheduler,
                       until=5*len(train_dataloader))

    best_score = 0.0
    best_ep = 1

    for ep in range(1, N_EPOCHS+1):
        for phase, dataloader in zip(['train', 'val'], [train_dataloader, val_dataloader]):
            if phase == "train":
                model.train()
            else:
                model.eval()

            # disable gradients to save memory during validation phase
            torch.set_grad_enabled(phase == "train")

            running_loss = 0.0

            dataloader = iter(dataloader)

            y_true = []
            y_pred = []
            y_score = []

            for _ in tqdm.tqdm(range(len(dataloader)), desc=f"{phase.capitalize()} Loop"):
                x, label = next(dataloader)
                label = label.view(-1, 1).float()

                if torch.cuda.is_available():
                    x = x.cuda()
                    label = label.cuda()

                out = model(x)
                loss = crit(out, label)

                y_score += out.flatten().detach().cpu().numpy().tolist()
                y_true += label.flatten().detach().cpu().numpy().tolist()
                y_pred += out.flatten().ge(0.5).float().detach().cpu().tolist()

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    warmer.step(epoch=ep)

                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = compute_accuracy(y_true, y_pred)
            epoch_auprc = compute_auprc(y_true, y_score)
            epoch_auc = compute_auc(y_true, y_score)

            print(
                f"[EP={ep}][{phase}] ====> LOSS: {epoch_loss:.6f}, ACCURACY: {epoch_acc:.6f}, AUPRC: {epoch_auprc:.6f}, AUC: {epoch_auc:.6f}")

            if phase == "val" and epoch_auprc > best_score:
                # remove previous best model
                prev_model = os.path.join(
                    EXP_SUB_DIR, f"epoch={best_ep}.pth")
                if os.path.exists(prev_model):
                    os.system(f"rm {prev_model}")

                # update best_score & best_ep
                best_score = epoch_auprc
                best_ep = ep

                # save best model weights
                model_path = os.path.join(EXP_SUB_DIR, f"epoch={best_ep}.pth")
                torch.save(model.state_dict(), model_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import tqdm
import random
import os
import math

from models.baseline import BasicConv1d
from datasets.maic2020 import MAIC2020_rec
from utils.metrics import *
from utils.losses import BCE_with_class_weights
from utils.lr_scheduler import LR_Warmer

if __name__ == "__main__":
    BATCH_SIZE = 512 * torch.cuda.device_count()
    VALIDATION_SPLIT = 0.2
    N_EPOCHS = 30

    # dataset statistics for input data normalization
    MEAN = 85.15754
    STD = 22.675957

    EXP_HOME = "./experiments_pretrain"
    DATA_ROOT = os.path.join("/data", ".cache", "datasets", "MAIC2020")

    verion_ix = len(os.listdir(EXP_HOME))
    EXP_SUB_DIR = os.path.join(EXP_HOME, "version-{}".format(verion_ix))
    os.system(f"mkdir -p {EXP_SUB_DIR}")

    # Prepare Dataset
    train_ds = MAIC2020_rec(infile="/data/.cache/datasets/MAIC2020/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                            transform=lambda x: (x-MEAN)/STD, use_ext=False, train=True)
    val_ds = MAIC2020_rec(infile="/data/.cache/datasets/MAIC2020/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                          transform=lambda x: (x-MEAN)/STD, use_ext=False, train=False)

    print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=16, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=16, shuffle=False, pin_memory=torch.cuda.is_available())

    model = nn.Transformer(
        d_model=100, nhead=10, num_encoder_layers=12)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    crit = nn.MSELoss()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = np.inf
    best_ep = 1

    for ep in range(1, N_EPOCHS+1):
        try:
            for phase, dataloader in zip(['train', 'val'], [train_dataloader, val_dataloader]):
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                # disable gradients to save memory during validation phase
                torch.set_grad_enabled(phase == "train")

                running_loss = 0.0

                dataloader = iter(dataloader)
                for _ in tqdm.tqdm(range(len(dataloader)), desc=f"{phase.capitalize()} Loop"):
                    src, tgt = next(dataloader)

                    if torch.cuda.is_available():
                        src = src.cuda()
                        tgt = tgt.cuda()

                    # right-shifted target
                    tgt_shift = F.pad(tgt, (1, 0), value=0.0)[..., :-1]

                    # each vector represents samples for 1 sec
                    src = src.view(-1, 20, 100).transpose(0, 1)  # (S,N,E)
                    tgt = tgt.view(-1, 60, 100).transpose(0, 1)  # (T,N,E)
                    # (T,N,E)
                    tgt_shift = tgt_shift.view(-1, 60, 100).transpose(0, 1)

                    # => (T,N,E)
                    out = model(src, tgt_shift)
                    loss = crit(out, tgt)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # warmer.step(epoch=ep)

                    running_loss += loss.item()

                epoch_loss = running_loss / len(dataloader)

                print(
                    f"[EP={ep}][{phase}] ====> Reconstruction LOSS: {epoch_loss:.6f}")

                if phase == "val" and epoch_loss < best_loss:
                    # remove previous best model
                    prev_model = os.path.join(
                        EXP_SUB_DIR, f"epoch={best_ep}.pth")
                    if os.path.exists(prev_model):
                        os.system(f"rm {prev_model}")

                    # update best_loss & best_ep
                    best_loss = epoch_loss
                    best_ep = ep

                    # save best model weights
                    model_path = os.path.join(
                        EXP_SUB_DIR, f"epoch={best_ep}.pth")
                    torch.save(model.state_dict(), model_path)
        except KeyboardInterrupt:
            break

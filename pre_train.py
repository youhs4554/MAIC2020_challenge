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
    N_EPOCHS = 50

    MEAN = 65.0
    STD = 65.0

    EXP_HOME = "./experiments_pretrain"
    DATA_ROOT = os.path.join("/data", ".cache", "datasets", "MAIC2020")

    verion_ix = len(os.listdir(EXP_HOME))
    EXP_SUB_DIR = os.path.join(EXP_HOME, "version-{}".format(verion_ix))
    os.system(f"mkdir -p {EXP_SUB_DIR}")

    # Prepare Dataset
    train_ds = MAIC2020_rec(infile="/data/.cache/datasets/MAIC2020/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                            transform=lambda x: (x-MEAN)/STD, use_ext=False, train=True, composition=True)
    val_ds = MAIC2020_rec(infile="/data/.cache/datasets/MAIC2020/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                          transform=lambda x: (x-MEAN)/STD, use_ext=False, train=False, composition=True)

    print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=16, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=16, shuffle=False, pin_memory=torch.cuda.is_available())

    from models.transformers.transformers import TransformerModel_MTL
    model = TransformerModel_MTL(
        n_cls=2, d_model=512, nhead=8, num_encoder_layers=6)

    rec_loss = nn.MSELoss()
    cls_loss = nn.BCELoss()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.95,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_dataloader) * N_EPOCHS,
    )
    warmer = LR_Warmer(optimizer, scheduler=scheduler,
                       until=5*len(train_dataloader))

    best_score = 0.0
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

                y_true = []
                y_pred = []
                y_score = []

                for _ in tqdm.tqdm(range(len(dataloader)), desc=f"{phase.capitalize()} Loop"):
                    src, tgt, label = next(dataloader)

                    if torch.cuda.is_available():
                        src = src.cuda()
                        tgt = tgt.cuda()
                        label = label.cuda()

                    # right-shifted target
                    tgt_shift = F.pad(tgt, (1, 0), value=0.0)[..., :-1]

                    # each vector represents samples for 1 sec
                    tgt = tgt.view(-1, 60, 100).transpose(0, 1)  # (T,N,E)

                    logits_out, decoder_out = model(src, tgt_shift)
                    loss = cls_loss(logits_out, label) + \
                        rec_loss(decoder_out, tgt.transpose(0, 1))

                    y_score += logits_out.flatten().detach().cpu().numpy().tolist()
                    y_true += label.flatten().detach().cpu().numpy().tolist()
                    y_pred += logits_out.flatten().ge(0.5).float().detach().cpu().tolist()

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        warmer.step()

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
                    model_path = os.path.join(
                        EXP_SUB_DIR, f"epoch={best_ep}.pth")
                    torch.save(model.state_dict(), model_path)
        except KeyboardInterrupt:
            break

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import tqdm
import random
import os
import math

from models.baseline import BasicConv1d
from datasets.maic2020 import MAIC2020
from utils.metrics import *
from utils.losses import BCE_with_class_weights
from utils.lr_scheduler import LR_Warmer
import torchaudio
import torchvision.transforms as F


if __name__ == "__main__":
    BATCH_SIZE = 4096 * torch.cuda.device_count()
    N_EPOCHS = 100

    # dataset statistics for input data normalization
    MEAN = 65.0
    STD = 65.0

    EXP_HOME = "./experiments"
    DATA_ROOT = os.path.join("/data", ".cache", "datasets", "MAIC2020")

    verion_ix = len(os.listdir(EXP_HOME))
    EXP_SUB_DIR = os.path.join(EXP_HOME, "version-{}".format(verion_ix))
    os.system(f"mkdir -p {EXP_SUB_DIR}")

    specgram_transform = F.Compose([
        torchaudio.transforms.Spectrogram(),
        torch.log2,
        F.Lambda(lambda x: ((x-x.min())/(x.max()-x.min())).float()),
        F.ToPILImage(),
        F.Grayscale(3),
        # F.Resize((224, 244)),
        F.ToTensor(),
        F.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare Dataset
    train_ds = MAIC2020(infile="/data/.cache/datasets/MAIC2020/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                        transform=lambda x: (x-MEAN)/STD, use_ext=False, train=True)
    val_ds = MAIC2020(infile="/data/.cache/datasets/MAIC2020/train_cases.csv", SRATE=100, MINUTES_AHEAD=5,
                      transform=lambda x: (x-MEAN)/STD, use_ext=False, train=False)

    print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=16, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=16, shuffle=False, pin_memory=torch.cuda.is_available())

    # # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
    # # 6-layers 1d-CNNs
    # model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])

    # import models.resnet1d
    # from models.non_local.nl_conv1d import NL_Conv1d
    # backbone = models.resnet1d.resnet50()
    # model = NL_Conv1d(backbone=backbone, squad="0,2,3,0", use_ext=False)

    # from torchvision.models import resnet101
    # model = resnet101(pretrained=True)
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, 1)

    from models.shufflenet import shufflenet_v2
    model = shufflenet_v2()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    crit = nn.BCELoss()
    # Todo. FocalLoss!

    if torch.cuda.is_available():
        model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.95,
        weight_decay=1e-3
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=math.floor(2/3*N_EPOCHS), gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_dataloader) * N_EPOCHS,
    )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[math.floor(1/3*N_EPOCHS), math.floor(2/3*N_EPOCHS)], gamma=0.1)
    warmer = LR_Warmer(optimizer, scheduler=scheduler,
                       until=10*len(train_dataloader))

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
                    x, label = next(dataloader)
                    label = label.view(-1, 1).float()

                    if torch.cuda.is_available():
                        x = x.cuda()
                        label = label.cuda()

                    # (b,1,2000) -> (b,20,100)
                    # opt1 > each vector : consecutive 20 samples
                    # x = x.view(-1, 100, 20).transpose(1, 2)

                    # opt2 > each K vector : 20 samples separated by 100 samples
                    # x = x.view(-1, 20, 100)
                    out = model(x)
                    loss = crit(out, label)

                    y_score += out.flatten().detach().cpu().numpy().tolist()
                    y_true += label.flatten().detach().cpu().numpy().tolist()
                    y_pred += out.flatten().ge(0.5).float().detach().cpu().tolist()

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

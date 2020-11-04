import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import tqdm
import random
import os
import math
import yaml
import glob
from natsort import natsorted

from models.strided_convnet import StridedConvNet
from models.baseline import BasicConv1d
import models.resnet1d
from models.non_local.nl_conv1d import NL_Conv1d
from models.shufflenet import shufflenet_v2

from models.transformers.transformers import TransformerModel_MTL, transformers
from models.ensemble import ensemble_of_shufflenet_stridedconv_resnet34NL

from datasets.maic2020 import MAIC2020, MAIC2020_rec, MAIC2020_image, prepare_test_dataset
from utils.metrics import *
from utils.losses import BCE_with_class_weights, WeightedFocalLoss
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.models.resnet as resnet_module
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def _build_model(arch):
    if arch == "conv1d_lenet":
        # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
        model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])
    if arch == "conv_strided":
        model = StridedConvNet()
    elif arch.startswith("conv1d_r"):
        suffix = "conv1d_r"
        resnet_arch = ("resnet" + arch[len(suffix):]).split("_")[0]

        if resnet_arch == "resnet34":
            squad = "0,2,2,0"
        elif resnet_arch == "resnet50":
            squad = "0,2,3,0"
        else:
            raise ValueError(f"not support {resnet_arch}")
        resnet = getattr(models.resnet1d, resnet_arch)(
            in_channels=1 if args.single_ch else 20)
        model = NL_Conv1d(resnet=resnet, squad=squad, use_ext=args.use_ext)
    elif arch == "shufflenet_v2":
        model = shufflenet_v2()
    elif arch == "transformer_basic":
        model = transformers()
    elif arch == "transformer_mtl":
        model = TransformerModel_MTL(
            n_cls=2, d_model=512, nhead=8, num_encoder_layers=6)
    elif arch.startswith("conv2d_r"):
        suffix = "conv2d_r"
        resnet_arch = ("resnet" + arch[len(suffix):]).split("_")[0]
        model = getattr(resnet_module, resnet_arch)()
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch.startswith("conv2d_efficient"):
        from efficientnet_pytorch import EfficientNet
        pretrained_name = arch.split("_")[1]
        model = EfficientNet.from_pretrained(pretrained_name, num_classes=1)
    elif arch == "ensemble":
        model = ensemble_of_shufflenet_stridedconv_resnet34NL()

    return model


class LitModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.pytorch_model = _build_model(hparams.arch)
        self.hparams = hparams

    def forward(self, *inputs):
        inputs = self.arrange_inputs(*inputs)
        if self.hparams.arch.endswith("mtl"):
            if self.trainer.testing:
                x_seg, = inputs
                logits_out = self.pytorch_model(x_seg)
                return logits_out

            else:
                x_seg, y_seg = inputs
                # right-shifted target
                y_seg_shift = F.pad(y_seg, (1, 0), value=0.0)[..., :-1]

                logits_out, decoder_out = self.pytorch_model(
                    x_seg, y_seg_shift)

                return logits_out, decoder_out
        elif self.hparams.use_ext:
            x_seg, ext = inputs
            # raise NotImplementedError("comming soon!")
            return self.pytorch_model(x_seg, ext)
        else:
            x_seg, = inputs

            return self.pytorch_model(x_seg)

    def arrange_inputs(self, *inputs):
        inputs = list(inputs)
        x_seg = inputs[0]
        if self.hparams.single_ch:
            x_seg = x_seg.view(-1, 1, 2000)
        else:
            if self.hparams.arch == "shufflenet_v2":
                # reshape 1d -> 2d
                x_seg = x_seg.view(-1, 1, 40, 50)
            elif "2d" in self.hparams.arch:
                # do nothing
                pass
            else:
                x_seg = x_seg.view(-1, 20, 100)
        # replace input data
        inputs[0] = x_seg
        return tuple(inputs)

    def step(self, batch):
        _losses = {}
        if self.hparams.arch.endswith("mtl"):
            x_seg, y_seg, label = batch
            logits_out, decoder_out = self(x_seg, y_seg)
            y_seg = y_seg.view(-1, 60, 100).transpose(0, 1)
            _losses["rec"] = nn.MSELoss()(decoder_out, y_seg)

        elif self.hparams.use_ext:
            x_seg, ext, label = batch
            logits_out = self(x_seg, ext)
        else:
            x_seg, label = batch
            logits_out = self(x_seg)

        cls_crit = nn.BCEWithLogitsLoss()
        # cls_crit = BCE_with_class_weights()
        if self.hparams.focal_loss:
            cls_crit = WeightedFocalLoss(smoothing=0.1)
        _losses["cls"] = cls_crit(logits_out, label.float())
        loss = sum(_losses.values())

        return loss, label, logits_out

    def eval_step(self, test_batch):
        if self.hparams.use_ext:
            x_seg, ext = torch.split(test_batch[0], [4, 2000], dim=1)
            logits_out = self(x_seg, ext)
        else:
            if len(test_batch) == 2:
                x_seg, _ = test_batch
            else:
                x_seg, = test_batch
            logits_out = self(x_seg)

        return logits_out

    def on_fit_start(self):
        # init tensorboard logger at the beginning of fit()
        self.tb = self.logger.experiment
        if self.hparams.focal_loss and not self.trainer.testing:
            print('\033[1m' + '\033[92m' + 'Focal Loss : ENABLED' + '\033[0m')

    def training_step(self, batch, batch_nb):
        loss, label, logits_out = self.step(batch)
        return {"loss": loss, "y_true": label.detach(), "y_score": torch.sigmoid(logits_out).detach()}

    def training_step_end(self, metrics):
        # aggregate losses from multi-gpu
        self.tb.add_scalar(
            "train_loss", metrics["loss"].mean().item(), self.trainer.global_step)
        return metrics

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean()
        auprc, auc = self.get_scores(outputs)

        log_dict = {"train_auprc": auprc, "train_auc": auc}

        self.tb.add_scalar("train_auprc", auprc.item(),
                           self.trainer.global_step)
        self.tb.add_scalar("train_auc", auc.item(), self.trainer.global_step)

        return {"train_loss": avg_loss, "log": log_dict}

    def validation_step(self, batch, batch_nb):
        loss, label, logits_out = self.step(batch)
        return {"loss": loss, "y_true": label.detach(), "y_score": torch.sigmoid(logits_out).detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean()
        auprc, auc = self.get_scores(outputs)

        self.tb.add_scalar("val_loss", avg_loss.item(),
                           self.trainer.global_step)
        self.tb.add_scalar("val_auprc", auprc.item(), self.trainer.global_step)
        self.tb.add_scalar("val_auc", auc.item(), self.trainer.global_step)

        log_dict = {"val_auprc": auprc, "val_auc": auc}

        return {"val_loss": avg_loss, "log": log_dict}

    def test_step(self, test_batch, batch_nb):
        logits_out = self.eval_step(test_batch)
        return {"y_score": torch.sigmoid(logits_out)}

    def test_end(self, outputs):
        results = []
        for o in outputs:
            results += o["y_score"].flatten().detach().cpu().numpy().tolist()

        return {"results": np.array(results)}

    def get_scores(self, outputs):
        y_true = torch.cat([o["y_true"] for o in outputs])
        y_score = torch.cat([o["y_score"] for o in outputs])

        auprc = compute_auprc(y_true.cpu().numpy(), y_score.cpu().numpy())
        auc = compute_auc(y_true.cpu().numpy(), y_score.cpu().numpy())

        # to tensor
        auprc = torch.tensor(auprc).to(y_score)
        auc = torch.tensor(auc).to(y_score)
        return auprc, auc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.pytorch_model.parameters(), lr=self.hparams.lr, momentum=0.95
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.hparams.n_epochs,
            pct_start=0.3,
            base_momentum=0.9 * 0.95,
            max_momentum=0.95,
        )
        return [optimizer], [scheduler]


def create_parser():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Device Targets')
    g.add_argument(
        '--cfg',
        dest='cfg',
        type=argparse.FileType(mode='r'))
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.cfg:
        data = yaml.load(args.cfg, Loader=yaml.FullLoader)
        delattr(args, 'cfg')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


def cross_validate(args, train_ds, test_ds, transform=None, n_folds=5):
    # define test dataloader
    test_dataloader = DataLoader(
        test_ds, batch_size=args.batch_per_gpu * torch.cuda.device_count(), num_workers=args.num_workers, shuffle=False, pin_memory=torch.cuda.is_available())

    from sklearn.model_selection import KFold, StratifiedKFold
    # TODO. Stratified CV
    y_train = np.concatenate([d.y_true for d in train_ds.datasets])
    kfold = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)

    for fold, (train_indices, valid_indices) in enumerate(kfold.split(np.ones((len(train_ds), 1)), y_train)):
        train_ds_ = torch.utils.data.Subset(train_ds, train_indices)
        valid_ds_ = torch.utils.data.Subset(train_ds, valid_indices)

        print(
            f"[Train] fold-{fold} : {np.unique(y_train[train_indices], return_counts=True)}")
        print(
            f"[Valid] fold-{fold} : {np.unique(y_train[valid_indices], return_counts=True)}")

        train_dataloader = DataLoader(
            train_ds, batch_size=args.batch_per_gpu * torch.cuda.device_count(), num_workers=args.num_workers, shuffle=True, pin_memory=torch.cuda.is_available())
        val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_per_gpu * torch.cuda.device_count(), num_workers=args.num_workers, shuffle=False, pin_memory=torch.cuda.is_available())

        tb_logger = pl_loggers.TensorBoardLogger(
            args.logdir, name=args.arch, version=f"fold_{fold+1}")
        checkpoint_callback = ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch}-{val_auprc:.4f}",
                                              save_top_k=args.save_top_k, monitor="val_auprc", mode='max')
        early_stop_callback = EarlyStopping(
            monitor='val_auprc',
            patience=5,
            verbose=True,
            mode='max'
        )
        from pytorch_lightning.callbacks import LearningRateLogger
        lr_logger = LearningRateLogger(logging_interval='step')

        # model
        model = LitModel(args)

        # training
        trainer = pl.Trainer(
            logger=tb_logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            callbacks=[lr_logger],
            num_sanity_val_steps=0, gpus=torch.cuda.device_count(), distributed_backend='dp',
            max_epochs=100)
        trainer.fit(model, train_dataloader, val_dataloader)

        predictions = trainer.test(model, test_dataloaders=test_dataloader)[
            0]["results"][:len(test_ds)]

        # save scores as .txt file
        print('\x1b[6;30;42m' +
              f"saving resulting file at {tb_logger.log_dir}/pred_y.txt..." + '\x1b[0m', flush=True, end="")
        np.savetxt(os.path.join(tb_logger.log_dir, "pred_y.txt"), predictions)
        print('\x1b[6;30;42m' + "Done!" + '\033[0m', flush=True, end="\n")


if __name__ == "__main__":
    # fix seed for reproductivity
    torch.manual_seed(0)

    args = parse_args(create_parser())

    from utils.data_augmentation import DA_Permutation, DA_TimeWarp, DA_Jitter, DA_Scaling

    def apply_random_augmentations(x, p=.5):
        if random.random() < p:
            x = DA_Permutation(x)
        if random.random() < p:
            x = DA_TimeWarp(x)
        if random.random() < p:
            x = DA_Jitter(x)
        # if random.random() < p:
        #     x = DA_Scaling(x, sigma=0.1)

        return x

    if args.transform == "offset":
        baseline = 65.0

        def transform(x, is_training):
            if is_training:
                x = apply_random_augmentations(x)
            return (x-baseline)/baseline
    elif args.transform == "standard":
        mean = 85.15754
        std = 22.675957

        def transform(x, is_training):
            if is_training:
                x = apply_random_augmentations(x)
            return (x-mean)/std
    elif args.transform == "image":
        # transform = TF.Compose([
        #     TF.Resize((224, 224)),
        #     TF.ToTensor(),
        #     TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        raise NotImplementedError(
            "The ImageTransform is not working anymore!")

    # TODO. Augmentation method for time-series data
    # code : https://nbviewer.jupyter.org/github/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb
    # paper : https://arxiv.org/pdf/1706.00527.pdf

    if "2d" in args.arch:
        data_init = MAIC2020_image
    elif args.arch.endswith("mtl"):
        # (current + future)
        data_init = MAIC2020_rec
    else:
        # current conly
        data_init = MAIC2020

    train_ds = data_init(SRATE=100, MINUTES_AHEAD=5,
                         transform=transform, use_ext=args.use_ext, train=True)
    val_ds = data_init(SRATE=100, MINUTES_AHEAD=5,
                       transform=transform, use_ext=args.use_ext, train=False)
    train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    test_ds = prepare_test_dataset(
        args.data_root, transform=transform, use_ext=args.use_ext, use_image="2d" in args.arch)
    print('\033[1m' + '\033[93m' + f"Train: {len(train_ds)}")
    print('\033[1m' + '\033[96m' + f"Validation: {len(val_ds)}" + '\033[0m')

    cross_validate(
        args, train_ds, test_ds, transform)

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

from models.baseline import BasicConv1d
import models.resnet1d
from models.non_local.nl_conv1d import NL_Conv1d
from models.shufflenet import shufflenet_v2

from models.transformers.transformers import TransformerModel_MTL, transformers


from datasets.maic2020 import MAIC2020, MAIC2020_rec, prepare_test_dataset
from utils.metrics import *
from utils.losses import WeightedFocalLoss
import torchaudio
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


def _build_model(arch):
    if arch == "conv1d_lenet":
        # Define model (preproduce baseline implemented at https://github.com/vitaldb/maic2020/blob/master/maic_data_cnn.py)
        model = BasicConv1d(dims=[1, 64, 64, 64, 64, 64, 64])
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
            raise NotImplementedError("comming soon!")
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

        cls_crit = nn.BCELoss()
        if self.hparams.focal_loss:
            cls_crit = WeightedFocalLoss()
        _losses["cls"] = cls_crit(logits_out, label.float())
        loss = sum(_losses.values())

        return loss, label, logits_out

    def eval_step(self, test_batch):
        if self.hparams.use_ext:
            x_seg, ext = torch.split(test_batch[0], [4, 2000], dim=1)
            logits_out = self(x_seg, ext)
        else:
            x_seg, = test_batch
            logits_out = self(x_seg)

        return logits_out

    def on_fit_start(self):
        # init tensorboard logger at the beginning of fit()
        self.tb = self.logger.experiment
        if self.hparams.focal_loss and not self.trainer.testing:
            print('\033[1m' + '\033[91m' + 'Focal Loss : ENABLED' + '\033[0m')

    def training_step(self, batch, batch_nb):
        loss, label, logits_out = self.step(batch)
        return {"loss": loss, "y_true": label.detach(), "y_score": logits_out.detach()}

    def training_step_end(self, metrics):
        # aggregate losses from multi-gpu
        self.tb.add_scalar(
            "train_loss", metrics["loss"].mean().item(), self.trainer.global_step)
        return metrics

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean()
        auprc, auc = self.get_scores(outputs)

        log_dict = {"train_auprc": auprc.item(), "train_auc": auc.item()}

        self.tb.add_scalar("train_auprc", auprc.item(),
                           self.trainer.global_step)
        self.tb.add_scalar("train_auc", auc.item(), self.trainer.global_step)

        return {"train_loss": avg_loss, "log": log_dict}

    def validation_step(self, batch, batch_nb):
        loss, label, logits_out = self.step(batch)
        return {"loss": loss, "y_true": label.detach(), "y_score": logits_out.detach()}

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
        return {"y_score": logits_out}

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

    # learning rate warm-up
    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        using_native_amp=None,
    ):
        # use lr proposed by lr_finder
        lr = self.hparams.lr
        until = 5 * len(self.train_dataloader())  # warm-up for 5 epochs
        # warm up lr
        if self.trainer.global_step < until:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / until)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * lr
        else:
            self.lr_sch[optimizer_idx].step()

        # log for learning rate
        lr_val = optimizer.param_groups[0]["lr"]
        self.tb.add_scalar("lr", lr_val, self.trainer.global_step)

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.pytorch_model.parameters(), lr=0.1, momentum=0.95, weight_decay=1e-3
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.train_dataloader()) * self.hparams.n_epochs,
            eta_min=self.hparams.lr * (1 / 16),
        )

        self.lr_sch = [scheduler]

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


if __name__ == "__main__":
    # fix seed for reproductivity
    torch.manual_seed(0)
    # # TODO.
    # parser.add_argument("--ensemble_infer", action="store_true",
    #                     help="ensemble top-k model results at inference stage(i.e. test)")

    args = parse_args(create_parser())

    """
        STEP 1 : prepare dataset
    """
    if args.transform == "offset":
        baseline = 65.0
        def transform(x): return (x-baseline)/baseline
    elif args.transform == "standard":
        mean = 85.15754
        std = 22.675957
        def transform(x): return (x-mean)/std

    if args.arch.endswith("mtl"):
        # (current + future)
        data_init = MAIC2020_rec
    else:
        # current conly
        data_init = MAIC2020

    train_ds = data_init(SRATE=100, MINUTES_AHEAD=5,
                         transform=transform, use_ext=args.use_ext, train=True)
    val_ds = data_init(SRATE=100, MINUTES_AHEAD=5,
                       transform=transform, use_ext=args.use_ext, train=False)
    test_ds = prepare_test_dataset(
        args.data_root, transform=transform, use_ext=args.use_ext)

    # class _dummyDS(torch.utils.data.Dataset):
    #     def __init__(self, arch, train=False):
    #         if train:
    #             N = 10000
    #         else:
    #             N = 2000
    #         self.x_seg = torch.randn(N, 1, 2000)
    #         self.y_seg = torch.randn(N, 1, 6000)
    #         self.label = torch.randint(low=0, high=2, size=(N, 1))
    #         self.arch = arch

    #     def __len__(self):
    #         return len(self.x_seg)

    #     def __getitem__(self, ix):
    #         if self.arch.endswith("mtl"):
    #             return self.x_seg[ix], self.y_seg[ix], self.label[ix]
    #         else:
    #             return self.x_seg[ix], self.label[ix]

    # train_ds = _dummyDS(arch=args.arch, train=True)
    # val_ds = _dummyDS(arch=args.arch, train=False)

    print('\033[1m' + '\033[93m' + f"Train: {len(train_ds)}")
    print('\033[1m' + '\033[96m' + f"Validation: {len(val_ds)}" + '\033[0m')

    train_dataloader = DataLoader(
        train_ds, batch_size=args.batch_per_gpu * torch.cuda.device_count(), num_workers=args.num_workers, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(
        val_ds, batch_size=args.batch_per_gpu * torch.cuda.device_count(), num_workers=args.num_workers, shuffle=False, pin_memory=torch.cuda.is_available())
    test_dataloader = DataLoader(
        test_ds, batch_size=args.batch_per_gpu * torch.cuda.device_count(), num_workers=args.num_workers, shuffle=False, pin_memory=torch.cuda.is_available())

    """
        STEP 2 : build model
    """
    model = LitModel(args)

    """
        STEP 3 : configure trainer -> fit
    """

    tb_logger = pl_loggers.TensorBoardLogger(args.logdir, name=args.arch)

    checkpoint_callback = ModelCheckpoint(
        os.path.join(args.logdir, tb_logger.name,
                     f"version_{tb_logger.version}", "checkpoints", "{epoch}-{val_auprc:.4f}"),
        save_top_k=args.save_top_k, monitor="val_auprc", mode='max')

    trainer = pl.Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0, gpus=torch.cuda.device_count(), distributed_backend='dp')
    trainer.fit(model, train_dataloader, val_dataloader)

    """
        STEP 4 : test and save a result file to submit
    """
    if args.ensemble_infer:
        print('\033[1m' + '\033[91m' + '\033[4m' +
              'Ensemble Infer : ENABLED' + '\033[0m')

        ckpt_dir = os.path.join(args.logdir, tb_logger.name,
                                f"version_{tb_logger.version}", "checkpoints", "*")
        results = []
        for ckpt in natsorted(glob.glob(ckpt_dir)):
            results.append(trainer.test(test_dataloaders=test_dataloader, ckpt_path=ckpt)[0][
                "results"]
            )
        results = np.mean(results, axis=0)

    else:
        best_ckpt = checkpoint_callback.best_model_path
        results = trainer.test(model, test_dataloaders=test_dataloader, ckpt_path=best_ckpt)[0][
            "results"]

    # save scores as .txt file
    exp_dir = os.path.join(args.logdir, tb_logger.name,
                           f"version_{tb_logger.version}")
    np.savetxt(os.path.join(exp_dir, "pred_y.txt"), results)

    print('\x1b[6;30;42m' + 'Done!' + '\x1b[0m' + '\033[0m')

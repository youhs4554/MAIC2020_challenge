import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

from models.base_models import NLBlockND
from tools.models.BERT.bert import BERT5


class NL_Conv1d(nn.Module):
    def __init__(self, resnet, squad, use_ext=False):
        super().__init__()

        self.use_ext = use_ext

        fdim = resnet.fc.in_features
        hidden_dim = fdim // 4

        del resnet.avgpool
        del resnet.fc

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        layerNames = ["layer{}".format(i) for i in range(1, 4+1)]

        # parse squad
        squad = dict(zip(layerNames, eval(squad)))

        self.netConfig = [64, 64, 128, 256, 512]
        if fdim == 2048:
            # deeper CNNs
            self.netConfig = [64, 256, 512, 1024, 2048]

        # pSG blocks
        self.block_a = self.make_layers("layer1", n_blocks=squad.get("layer1"))
        self.block_b = self.make_layers("layer2", n_blocks=squad.get("layer2"))
        self.block_c = self.make_layers("layer3", n_blocks=squad.get("layer3"))
        self.block_d = self.make_layers("layer4", n_blocks=squad.get("layer4"))

        if use_ext:
            # embedding layer for external data
            self.embedding = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, fdim),
                nn.ReLU(inplace=True)
            )

        # self.bert_pool = BERT5(fdim, 4, hidden=fdim, n_layers=1, attn_heads=8)

        # classifier
        self.classifier = nn.Linear(fdim * 2 if use_ext else fdim, 1)

    def make_layers(self, layerName, n_blocks=None):
        resnet_layer = getattr(self, layerName)
        if n_blocks is None:
            # if n_blocks is not given, any blocks are not included
            n_blocks = 0

        NL_idx = range(len(resnet_layer))[
            1::2][:n_blocks]  # default : offset=1, step=2

        if n_blocks > math.ceil(len(resnet_layer) / 2):
            NL_idx = range(len(resnet_layer))[:n_blocks]  # offset=0, step=1
        print(layerName, list(NL_idx))
        layers = nn.ModuleList()
        in_channels = self.netConfig[eval(layerName.split("layer")[-1])]

        for i in range(len(resnet_layer)):
            blck = resnet_layer[i]
            layers.append(blck)
            if i in NL_idx:
                # if i is included in NL_idx, attach NL-Block
                layers.append(
                    NLBlockND(in_channels=in_channels, dimension=1))

        return nn.Sequential(*layers)

    def _forward(self, *inputs):
        if len(inputs) == 2 and self.use_ext:
            x, ext = inputs
        else:
            x, = inputs

        x = self.stem(x)
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)

        # """
        #     Bert pooling
        # """
        # # N,C,D -> N,D,C
        # x = x.transpose(1, 2)

        # bert_outputs, _ = self.bert_pool(x)
        # x = bert_outputs[:, 0, :]

        x = F.adaptive_avg_pool1d(x, 1).flatten(1)

        if self.use_ext:
            # embedding of external data
            ext_emb = self.embedding(ext)

            x = torch.cat((x, ext_emb), dim=1)

        x = self.classifier(x)
        x = torch.sigmoid(x)  # sigmoid output

        return x

    def forward(self, *inputs, **kwargs):
        out = self._forward(*inputs, **kwargs)
        return out

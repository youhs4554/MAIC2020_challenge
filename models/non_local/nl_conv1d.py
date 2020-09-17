import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

from models.base_models import NLBlockND


class NL_Conv1d(nn.Module):
    def __init__(self, backbone, squad):
        super().__init__()

        fdim = backbone.fc.in_features

        del backbone.avgpool
        del backbone.fc

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        layerNames = ["layer{}".format(i) for i in range(1, 4+1)]

        # parse squad
        squad = dict(zip(layerNames, eval(squad)))

        self.netConfig = [64, 64, 128, 256, 512]
        # self.netConfig = [64, 256, 512, 1024, 2048]

        # pSG blocks
        self.block_a = self.make_layers("layer1", n_blocks=squad.get("layer1"))
        self.block_b = self.make_layers("layer2", n_blocks=squad.get("layer2"))
        self.block_c = self.make_layers("layer3", n_blocks=squad.get("layer3"))
        self.block_d = self.make_layers("layer4", n_blocks=squad.get("layer4"))

        # classifier
        self.classifier = nn.Linear(fdim, 1)

    def make_layers(self, layerName, n_blocks=None):
        backbone_layer = getattr(self, layerName)
        if n_blocks is None:
            # if n_blocks is not given, any blocks are not included
            n_blocks = 0

        NL_idx = range(len(backbone_layer))[
            1::2][:n_blocks]  # default : offset=1, step=2

        if n_blocks > math.ceil(len(backbone_layer) / 2):
            NL_idx = range(len(backbone_layer))[:n_blocks]  # offset=0, step=1
        print(layerName, list(NL_idx))
        layers = nn.ModuleList()
        in_channels = self.netConfig[eval(layerName.split("layer")[-1])]

        for i in range(len(backbone_layer)):
            blck = backbone_layer[i]
            layers.append(blck)
            if i in NL_idx:
                # if i is included in NL_idx, attach NL-Block
                layers.append(
                    NLBlockND(in_channels=in_channels, dimension=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)

        x = F.adaptive_avg_pool1d(x, (1,)).flatten(
            1)  # avg-pool along with time-axis
        x = self.classifier(x)
        x = torch.sigmoid(x)  # sigmoid output

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.base_models import NoneLocalBlockNd


class NL_Conv1d(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        fdim = backbone.fc.in_features

        del backbone.avgpool
        del backbone.fc

        self.backbone = nn.Sequential(OrderedDict(backbone.named_children()))
        self.non_local = NoneLocalBlockNd(in_channels=fdim, dimension=1)
        self.classifier = nn.Linear(fdim, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.non_local(x)  # do non-local operation on res5

        x = F.adaptive_avg_pool1d(x, (1,)).flatten(
            1)  # avg-pool along with time-axis
        x = self.classifier(x)
        x = torch.sigmoid(x)  # sigmoid output

        return x

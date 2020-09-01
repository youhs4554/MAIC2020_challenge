import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import ConvNd

# reproduce baseline model


class BasicConv1d(nn.Module):
    def __init__(self, dims=[64, 64, 64, 64, 64, 64], n_outputs=1):
        super().__init__()
        self.feature = self.make_layers(dims)
        self.classifier = nn.Linear(dims[-1], n_outputs)

    def make_layers(self, dims):
        layers = []
        for in_ch, out_ch in zip(dims[:-1], dims[1:]):
            m = ConvNd(in_ch, out_ch, kernel_size=3, dimension=1, bias=False)
            layers.append(m)

        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x):
        x = self.feature(x)
        x = F.adaptive_avg_pool1d(x, (1,)).flatten(
            1)  # avg-pool along with time-axis
        x = self.classifier(x)
        x = torch.sigmoid(x)  # sigmoid for BCE
        return x

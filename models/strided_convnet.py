import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import ConvNd


class StridedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvNd(1, 16, kernel_size=4, stride=3,
                            dimension=1, bias=False)  # (batch, 16, 666)
        self.conv2 = ConvNd(16, 32, kernel_size=4, stride=3,
                            dimension=1, bias=False)  # (batch, 32, 221)
        self.conv3 = ConvNd(32, 64, kernel_size=3, stride=2,
                            dimension=1, bias=False)  # (batch, 64, 110)
        self.conv4 = ConvNd(64, 64, kernel_size=3, stride=2, padding=1,
                            dimension=1, bias=False)  # (batch, 64, 55)
        self.conv5 = ConvNd(64, 64, kernel_size=2, stride=2, padding=1,
                            dimension=1, bias=False)  # (batch, 64, 28)
        self.conv6 = ConvNd(64, 64, kernel_size=2, stride=2, padding=1,
                            dimension=1, bias=False)  # (batch, 64, 15)
        self.conv7 = ConvNd(64, 64, kernel_size=2, stride=1,
                            dimension=1, bias=False)  # (batch, 64, 14)

        self.classifier = nn.Linear(64, 1)

        for m in self.modules():
            # if isinstance(m, nn.Conv1d):
            #     nn.init.kaiming_normal_(
            #         m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight
                ),
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # GAP
        x = F.adaptive_avg_pool1d(x, 1).flatten(1)

        x = self.classifier(x)
        return x

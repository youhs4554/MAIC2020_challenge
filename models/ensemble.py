import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Sequential):
    def __init__(self, modelA, modelB, modelC):
        """
            modelA : shufflenet_v2,
            modelB : stridedConv
            modelC : nonlocal
        """

        super().__init__()
        dimA = modelA.fc.in_features
        dimB = modelB.classifier.in_features
        dimC = modelC.classifier.in_features

        modelA.fc = nn.Identity()
        modelB.classifier = nn.Identity()
        modelC.classifier = nn.Identity()

        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(dimA, 512)
        self.fc2 = nn.Linear(dimB, 512)
        self.fc3 = nn.Linear(dimC, 512)

        self.classifier = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight
                ),
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # suppose x is single-channel input
        xa = self.modelA(x.view(-1, 1, 40, 50))
        xb = self.modelB(x)
        xc = self.modelC(x)

        # feature embedding
        out_a = self.fc1(xa)
        out_b = self.fc2(xb)
        out_c = self.fc3(xc)

        out = out_a + out_b + out_c
        out = self.classifier(out)

        return out


def ensemble_of_shufflenet_stridedconv_resnet34NL():

    from .shufflenet import shufflenet_v2
    from .strided_convnet import StridedConvNet
    from .non_local.nl_conv1d import NL_Conv1d
    from .resnet1d import resnet18

    modelA = shufflenet_v2()
    modelB = StridedConvNet()
    modelC = NL_Conv1d(resnet=resnet18(in_channels=1),
                       squad="0,2,2,0", use_ext=False)

    ensemble = Ensemble(modelA, modelB, modelC)
    return ensemble

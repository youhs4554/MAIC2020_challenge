import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE_with_class_weights(nn.Module):
    def __init__(self, class_weights={0: 1, 1: 10}):
        super().__init__()
        self.class_weights = class_weights
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true):
        weight = torch.tensor(
            list(zip(sorted(self.class_weights.items(), key=lambda x: x[0])))[1]).flatten().to(y_pred.device)
        weight_ = weight[y_true.data.view(-1).long()].view_as(y_true)

        loss = self.criterion(y_pred, y_true)
        loss_class_weighted = loss * weight_
        loss_class_weighted = loss_class_weighted.mean()

        return loss_class_weighted


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2, smoothing=0.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, labels):
        with torch.no_grad():
            # label smoothing!
            targets = torch.ones_like(labels).mul(self.smoothing)
            targets.masked_fill_(labels.eq(1), 1.0-self.smoothing)

        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')

        at = self.alpha.gather(0, labels.long().data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

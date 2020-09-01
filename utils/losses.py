import torch
import torch.nn as nn


class BCE_with_class_weights(nn.Module):
    def __init__(self, class_weights={0: 1, 1: 10}):
        super().__init__()
        self.class_weights = class_weights
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, y_pred, y_true):
        weight = torch.tensor(
            list(zip(sorted(self.class_weights.items(), key=lambda x: x[0])))[1]).flatten().to(y_pred.device)
        weight_ = weight[y_true.data.view(-1).long()].view_as(y_true)

        loss = self.criterion(y_pred, y_true)
        loss_class_weighted = loss * weight_
        loss_class_weighted = loss_class_weighted.mean()

        return loss_class_weighted

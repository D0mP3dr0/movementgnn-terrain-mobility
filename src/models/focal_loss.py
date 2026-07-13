"""
FocalLoss - Loss for severe class imbalance (Lin et al., 2017).

Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

Where:
- pt: predicted probability for the correct class
- gamma: focusing parameter (down-weights easy examples)
- alpha: class-balancing weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [N, C]
            targets: Labels [N] (0-based)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

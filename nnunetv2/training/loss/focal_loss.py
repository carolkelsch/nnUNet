import torch
import torch.nn as nn
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal loss implementation.
    By deafult alpha is 1, not introducing precomputed differences based on labels. 
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.ndim == inputs.ndim:
            targets = targets[:, 0]
            targets = targets.long()

        # Use logits for numerical stability
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities
        pt = torch.exp(-ce_loss) 
        
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DC_and_Focal_loss(nn.Module):
    """
    Combination of Dice loss and Focal loss.
    """
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_dice=1, weight_focal=1,
        ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_and_Focal_loss, self).__init__()
        self.ignore_label = ignore_label
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

        self.focal = FocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        focal_loss = self.focal(net_output, target.long()) \
            if self.weight_focal != 0 else 0

        result = self.weight_focal * focal_loss + self.weight_dice * dc_loss
        return result
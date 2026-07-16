import torch
import torch.nn as nn
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    """
    Boundary Loss implementation for medical image segmentation.
    Expects pre-computed distance maps as the 'target'.
    Inspired by nnSAM (https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17481).
    """
    def __init__(self, apply_softmax=True):
        super(BoundaryLoss, self).__init__()
        self.apply_softmax = apply_softmax
        self.max_dist = 10.0
        # clipping distance is [-max_dist,+max_dist], make sure this
        # distance makes sense for the segmentation structure dimensions

    def forward(self, logits, gt_dist):
        """
        Args:
            logits: (B, C, H, W, [D]) - Raw network output
            gt_dist: (B, C, H, W, [D]) - Signed Distance Map
        """
        fg_logits = logits[:, 1:, ...] # discard background
        if self.apply_softmax:
            # If binary (C=1), use sigmoid. If multi-class, use softmax.
            probs = F.softmax(fg_logits, dim=1) if fg_logits.shape[1] > 1 else F.sigmoid(fg_logits)
        else:
            probs = logits

        pred_sdm = 1.0 - 2.0 * probs
        # The loss is the product of the probability map and the distance map
        # Points far from the boundary in the background have high positive distance
        # Points inside the object have negative/zero distance
        with torch.no_grad():
            gt_dist = torch.clamp(gt_dist, min=-self.max_dist, max=self.max_dist)
            gt_dist = gt_dist / self.max_dist

        loss = F.mse_loss(pred_sdm, gt_dist, reduction='mean')
        
        return loss

class DC_and_CE_and_BD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_bound=0.01,
        ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_and_CE_and_BD_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        
        self.ignore_label = ignore_label
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_bd = weight_bound

        self.boundary = BoundaryLoss()
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor, dist_maps: torch.Tensor):

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
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # target is the binary mask, dist_maps is the SDF
        bd_loss = self.boundary(net_output, dist_maps) if self.weight_bd != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_bd * bd_loss
        return result

class DC_and_BD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_ce=1, weight_dice=1, weight_bound=0.01,
        ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_and_BD_loss, self).__init__()
        
        self.ignore_label = ignore_label
        self.weight_dice = weight_dice
        self.weight_bd = weight_bound

        self.boundary = BoundaryLoss()
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor, dist_maps: torch.Tensor):

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

        # target is the binary mask, dist_maps is the SDF
        bd_loss = self.boundary(net_output, dist_maps) if self.weight_bd != 0 else 0

        result = self.weight_dice * dc_loss + self.weight_bd * bd_loss
        return result
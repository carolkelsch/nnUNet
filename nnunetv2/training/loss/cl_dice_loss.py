import torch
import torch.nn as nn
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn.functional as F
from nnunetv2.utilities.soft_skeleton import SoftSkeletonize

class SoftCLDiceLoss(nn.Module):
    """
    CLDice Loss implementation for medical image segmentation.
    Code from clDice (https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py).
    """

    """
    Number of iterations is related to the number of iterations of the soft skeletonize algorithm will
    perform to get the structure of the air tree. It is related to the dimension of the largest airways.

    15 was selected based on the AirRC dataset and the trachea sizes, for more info on
    the dataset check https://www.nature.com/articles/s41597-025-06074-6.

    """
    def __init__(self, apply_softmax=True, iter_=15, smooth = 1., exclude_background=False):
        super(SoftCLDiceLoss, self).__init__()
        self.apply_softmax = apply_softmax
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        """
        Args:
            y_pred: (B, C, H, W, [D]) - Raw network output
        """
        if self.apply_softmax:
            # If binary (C=1), use sigmoid. If multi-class, use softmax.
            y_pred = F.softmax(y_pred, dim=1) if y_pred.shape[1] > 1 else F.sigmoid(y_pred)
        
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        
        return cl_dice

class CE_and_CLD_loss(nn.Module):
    """
    Combination of Cross-entropy loss and clDice.
    """
    def __init__(self, cl_dice_kwargs, ce_kwargs, weight_ce=1, weight_cldice=1,
        ignore_label=None, dice_class=SoftCLDiceLoss):

        super(CE_and_CLD_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        
        self.ignore_label = ignore_label
        self.weight_cldice = weight_cldice
        self.weight_ce = weight_ce

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.cld = dice_class(**cl_dice_kwargs)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(CE_and_CLD_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        
        cld_loss = self.cld(net_output, target_dice) \
            if self.weight_cldice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_cldice * cld_loss
        return result

class DC_and_CLD_loss(nn.Module):
    """
    Combination of Dice loss and clDice.
    """
    def __init__(self, soft_dice_kwargs, cl_dice_kwargs, alpha, weight_dice=1, weight_cldice=1,
        ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_and_CLD_loss, self).__init__()
        
        self.ignore_label = ignore_label
        self.weight_dice = weight_dice
        self.weight_cldice = weight_cldice
        self.alpha = alpha

        self.cld = SoftCLDiceLoss(**cl_dice_kwargs)
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

        # target is the binary mask
        cld_loss = self.cld(net_output, target_dice) if self.weight_cldice != 0 else 0
        result = (1 - self.alpha) * dc_loss + self.alpha * cld_loss
        return result
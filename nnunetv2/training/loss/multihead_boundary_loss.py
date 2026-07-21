import torch
import torch.nn as nn
import torch.nn.functional as F

class MHBoundaryLoss(nn.Module):
    """
    Boundary Loss implementation for medical image segmentation.
    Expects pre-computed distance maps as the 'target'.
    """
    def __init__(self):
        super(MHBoundaryLoss, self).__init__()

    def forward(self, pred_sdm, gt_dist):
        """
        Args:
            logits: (B, C, H, W, [D]) - Raw network output
            gt_dist: (B, C, H, W, [D]) - Signed Distance Map from target
        """
        # If your regression head predicts all classes including background, 
        # slice out the background channel to perfectly align with your distance map shape.
        if pred_sdm.shape[1] > gt_dist.shape[1]:
            pred_sdm = pred_sdm[:, 1:, ...]

        loss = F.mse_loss(pred_sdm, gt_dist, reduction='mean')
        
        return loss

class TwinHeadSeparateLossWrapper(nn.Module):
    """
    Wrapper to handle deep supervision of cross-entropy and dice on the segmentation head and
    the boundary loss implemented in an added regression head.
    """
    def __init__(self, native_nnunet_ds_loss: nn.Module, lambda_boundary: float = 0.1):
        """
        Args:
            native_nnunet_ds_loss: The unaltered, default DeepSupervisionWrapper(DC_and_CE_loss)
                                   generated automatically by nnUNet.
            lambda_boundary: The scaling weight for your boundary regression.
        """
        super().__init__()
        self.nnunet_seg_loss = native_nnunet_ds_loss
        self.boundary_loss = MHBoundaryLoss()
        self.lambda_boundary = lambda_boundary

    def forward(self, model_outputs: tuple, targets: list, gt_sdf: torch.Tensor) -> dict:
        """
        Args:
            model_outputs: Tuple from our network -> (list_of_seg_logits, single_pred_sdf)
            targets: List of downsampled target masks for deep supervision tiers
            gt_sdf: Continuous precomputed full-resolution Tanh-SDF field
        """
        seg_outputs, pred_sdf = model_outputs

        # Compute standard multi-scale segmentation losses natively via nnUNet
        loss_seg = self.nnunet_seg_loss(seg_outputs, targets)

        # Compute boundary regression MSE completely independently at full resolution
        loss_boundary = self.boundary_loss(pred_sdf, gt_sdf)

        # Aggregate
        total_loss = loss_seg + (self.lambda_boundary * loss_boundary)

        return {
            'total_loss': total_loss,
            'seg_loss': loss_seg.detach(),
            'boundary_loss': loss_boundary.detach()
        }
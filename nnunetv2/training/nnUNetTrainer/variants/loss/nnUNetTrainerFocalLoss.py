import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.focal_loss import DC_and_Focal_loss
import numpy as np

class nnUNetTrainerFocalLoss(nnUNetTrainer):
    """
    Focal loss trainer for nnUNet.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        loss = DC_and_Focal_loss({'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_dice=1, weight_focal=1,
            ignore_label=self.label_manager.ignore_label
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

""" Variation of training length. """

class nnUNetTrainerFocalLoss_100epochs(nnUNetTrainerFocalLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerFocalLoss_200epochs(nnUNetTrainerFocalLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerFocalLoss_250epochs(nnUNetTrainerFocalLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
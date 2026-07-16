import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.cl_dice_loss import DC_and_CLD_loss
import numpy as np

class nnUNetTrainerCLDLoss(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        loss = DC_and_CLD_loss({'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'iter_': 15}, alpha=0.5, weight_dice=1, weight_cldice=1,
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


"""
Variations for the alpha hyperparameter to keep trainer name descriptive using the nnUNet logic.
"""
class nnUNetTrainerCLDLoss_alpha05(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        loss = DC_and_CLD_loss({'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'iter_': 15}, alpha=0.5, weight_dice=1, weight_cldice=1,
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

class nnUNetTrainerCLDLoss_alpha07(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        loss = DC_and_CLD_loss({'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'iter_': 15}, alpha=0.7, weight_dice=1, weight_cldice=1,
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

class nnUNetTrainerCLDLoss_alpha03(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        loss = DC_and_CLD_loss({'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'iter_': 15}, alpha=0.3, weight_dice=1, weight_cldice=1,
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

class nnUNetTrainerCLDLoss_alpha04(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        loss = DC_and_CLD_loss({'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'iter_': 15}, alpha=0.4, weight_dice=1, weight_cldice=1,
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

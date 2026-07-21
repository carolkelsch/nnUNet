import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainerEarlyStop import nnUNetTrainerEarlyStop
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerBDLoss import nnUNetTrainerBDLoss, nnUNetTrainer005BDLoss, nnUNetTrainer001BDLoss, nnUNetTrainer0005BDLoss, nnUNetTrainer008BDLoss, nnUNetTrainer006BDLoss, nnUNetTrainer007BDLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainer_MultiHeadBDLoss import nnUNetTrainerMultiHead001BDLoss, nnUNetTrainerMultiHead005BDLoss, nnUNetTrainerMultiHead0005BDLoss, nnUNetTrainerMultiHead008BDLoss, \
    nnUNetTrainerMultiHead001DCBDLoss, nnUNetTrainerMultiHead005DCBDLoss, nnUNetTrainerMultiHead0005DCBDLoss, nnUNetTrainerMultiHead008DCBDLoss, nnUNetTrainerMultiHead00005DCBDLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerFocalLoss import nnUNetTrainerFocalLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCLDLoss import nnUNetTrainerCLDLoss, nnUNetTrainerCLDLoss_alpha05, nnUNetTrainerCLDLoss_alpha03, nnUNetTrainerCLDLoss_alpha04, nnUNetTrainerCLDLoss_alpha07

""" Variations of trainers with different losses and early stopping strategy for training.
This nomination keeps nnUNet format for outputs, models output folders and inference outputs for validation. """

class nnUNetTrainerEarlyStop_BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerBDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_005BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainer005BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_006BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainer006BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_007BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainer007BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)


class nnUNetTrainerEarlyStop_008BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainer008BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_001BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainer001BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_0005BDLoss(nnUNetTrainerEarlyStop, nnUNetTrainer0005BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_FocalLoss(nnUNetTrainerEarlyStop, nnUNetTrainerFocalLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_CLDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerCLDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_CLDLoss_alpha07(nnUNetTrainerEarlyStop, nnUNetTrainerCLDLoss_alpha07):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_CLDLoss_alpha05(nnUNetTrainerEarlyStop, nnUNetTrainerCLDLoss_alpha05):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)


class nnUNetTrainerEarlyStop_CLDLoss_alpha03(nnUNetTrainerEarlyStop, nnUNetTrainerCLDLoss_alpha03):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_CLDLoss_alpha04(nnUNetTrainerEarlyStop, nnUNetTrainerCLDLoss_alpha04):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_0005MHBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead0005BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_001MHBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead001BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_005MHBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead005BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_008MHBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead008BDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_0005MHDCBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead0005DCBDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_001MHDCBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead001DCBDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_005MHDCBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead005DCBDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_008MHDCBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead008DCBDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)

class nnUNetTrainerEarlyStop_00005MHDCBDLoss(nnUNetTrainerEarlyStop, nnUNetTrainerMultiHead00005DCBDLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device, early_stopping=early_stopping)
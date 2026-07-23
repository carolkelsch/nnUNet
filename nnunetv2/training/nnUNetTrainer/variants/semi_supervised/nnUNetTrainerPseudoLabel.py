"""
Semi-supervised nnU-Net trainer: confidence-thresholded pseudo-labeling.

Then train with, e.g.:
    nnUNetv2_train 500 3d_fullres all -tr nnUNetTrainerPseudoLabel
"""

from typing import Union
import os
import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerPseudoLabel(nnUNetTrainer):
    # ----------------------- SSL hyperparameters -----------------------
    confidence_threshold: float = 0.90     # per-voxel softmax confidence cutoff
    unsup_loss_max_weight: float = 1.0     # asymptotic weight of the unsupervised loss
    rampup_epochs: int = 10                # epochs over which the weight ramps up
    unlabeled_batch_size: Union[int, None] = None  # defaults to labeled batch_size
    unlabeled_dataset_folder = None
    # ---------------------------------------------------------------------

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        #   nnUNet_preprocessed/<DatasetXXX_Name>/imagesTrUnlabeled
        unlabeled_folder = join(self.preprocessed_dataset_folder_base, "imagesTrUnlabeled")
        print(unlabeled_folder)
        self.unlabeled_preprocessed_folder = unlabeled_folder if os.path.exists(unlabeled_folder) else None
        self.dataloader_unlabeled = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def get_unlabeled_dataset(self):
        assert self.unlabeled_preprocessed_folder is not None, (
            "Set trainer.unlabeled_preprocessed_folder to the output of "
            "preprocess_unlabeled.py before calling run_training()."
        )

        unlbl_case_identifiers = self.dataset_class.get_identifiers(self.unlabeled_preprocessed_folder)

        return self.dataset_class(self.unlabeled_preprocessed_folder, unlbl_case_identifiers,
                                        folder_with_segs_from_previous_stage=None)

    def get_dataloaders(self):
        # standard labeled train/val dataloaders, unchanged from the base class
        dataloader_train, dataloader_val = super().get_dataloaders()

        unlabeled_dataset = self.get_unlabeled_dataset()
        bs = self.unlabeled_batch_size or self.batch_size
        patch_size = self.configuration_manager.patch_size

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dl_unlabeled = nnUNetDataLoader(unlabeled_dataset, bs,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=0,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=False)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            self.dataloader_unlabeled = SingleThreadedAugmenter(dl_unlabeled, None)
        else:
            self.dataloader_unlabeled = NonDetMultiThreadedAugmenter(data_loader=dl_unlabeled, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)

        # # let's get this party started
        _ = next(self.dataloader_unlabeled)
        return dataloader_train, dataloader_val

    # ------------------------------------------------------------------
    # Sigmoid ramp-up for the unsupervised loss weight (Laine & Aila /
    # Mean Teacher style) -- prevents early collapse from noisy pseudo-labels
    # ------------------------------------------------------------------
    def get_unsup_weight(self) -> float:
        if self.current_epoch >= self.rampup_epochs:
            return self.unsup_loss_max_weight
        phase = 1.0 - self.current_epoch / max(self.rampup_epochs, 1)
        return self.unsup_loss_max_weight * float(np.exp(-5.0 * phase ** 2))

    # ------------------------------------------------------------------
    # Training step: supervised loss + confidence-masked pseudo-label loss
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        target = [t.to(self.device, non_blocking=True) for t in target] \
            if isinstance(target, list) else target.to(self.device, non_blocking=True)

        unlabeled_batch = next(self.dataloader_unlabeled)
        unlabeled_data = unlabeled_batch['data'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        autocast_ctx = autocast(self.device.type, enabled=True) if self.device.type == 'cuda' \
            else dummy_context()

        with autocast_ctx:
            # ---- supervised loss on the labeled batch ----
            output = self.network(data)
            sup_loss = self.loss(output, target)

            # ---- generate pseudo-labels from the model's own prediction ----
            with torch.no_grad():
                pl_logits = self.network(unlabeled_data)
                pl_logits = pl_logits[0] if isinstance(pl_logits, (list, tuple)) else pl_logits
                probs = torch.softmax(pl_logits, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                confidence_mask = (max_probs >= self.confidence_threshold).float()

            # ---- second forward pass (with grad) supervised by pseudo-labels ----
            unsup_output = self.network(unlabeled_data)
            unsup_logits = unsup_output[0] if isinstance(unsup_output, (list, tuple)) else unsup_output

            voxelwise_ce = torch.nn.functional.cross_entropy(
                unsup_logits, pseudo_labels, reduction='none'
            )
            denom = confidence_mask.sum().clamp(min=1.0)
            unsup_loss = (voxelwise_ce * confidence_mask).sum() / denom

            weight = self.get_unsup_weight()
            total_loss = sup_loss + weight * unsup_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'sup_loss': sup_loss.detach().cpu().numpy(),
            'unsup_loss': unsup_loss.detach().cpu().numpy(),
            'unsup_weight': weight,
            'confident_fraction': confidence_mask.mean().detach().cpu().numpy(),
        }


class nnUNetTrainerPseudoLabelLonger(nnUNetTrainerPseudoLabel):
    # ----------------------- SSL hyperparameters -----------------------
    rampup_epochs: int = 30                # epochs over which the weight ramps up
    unlabeled_batch_size = 1
    # ---------------------------------------------------------------------

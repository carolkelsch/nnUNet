import torch
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.multihead_wrapper import DeepSupervisionTwinHeadNNUNet
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.multihead_boundary_loss import TwinHeadSeparateLossWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import numpy as np
import SimpleITK as sitk
from torch import autocast
import torch.nn as nn

class nnUNetTrainerMultiHeadBDLoss(nnUNetTrainer):
    """
    Trainer that combines baseline nnUNet loss (Dice and Cross-entropy) with boundary regression loss.
    """
    lambda_boundary = 0.01

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = DeepSupervisionTwinHeadNNUNet(self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            )).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            '''if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])'''

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True

            logger_config_hparas = {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.configuration_manager.batch_size
                }
            self.logger.update_config({"hparas": logger_config_hparas})
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _build_loss(self):
        """
        Builds the standard native nnUNet loss and safely pairs it 
        with our independent boundary loss wrapper.
        """
        # Call super() to let nnUNet natively build its DeepSupervisionWrapper(DC_and_CE_loss)
        # This keeps all native configurations completely intact without hardcoding them.
        native_nnunet_loss = super()._build_loss()

        # Wrap it cleanly with our standalone multi-head separator
        composite_loss = TwinHeadSeparateLossWrapper(
            native_nnunet_ds_loss=native_nnunet_loss,
            lambda_boundary=self.lambda_boundary  # Your alpha factor for the boundary head
        )

        return composite_loss
    
    def compute_sdf_target_3d(self, target_tensor: torch.Tensor, scale_factor: float = 5.0) -> torch.Tensor:
        """
        Computes a bounded, smooth 3D Signed Distance Function from a binary mask.
        Using same signal definitions as nnSAM (https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17481).
        Input target_tensor: Full-resolution target on CPU/GPU -> shape (B, 1, D, H, W)
        Output: Smoothed SDF tensor on the same device -> shape (B, 1, D, H, W)
        """
        # Move to numpy for SimpleITK processing
        mask_np = target_tensor.cpu().numpy().astype(np.uint8)
        B, C, D, H, W = mask_np.shape
        
        sdf_out = np.zeros_like(mask_np, dtype=np.float32)
        
        for b in range(B):
            for c in range(C):
                slice_3d = mask_np[b, c]
                
                # If the patch doesn't contain any airway voxels, assign a large maximum distance
                if not slice_3d.any():
                    sdf_out[b, c] = np.ones_like(slice_3d, dtype=np.float32) * scale_factor
                    continue
                    
                # Convert to SimpleITK Image
                sitk_img = sitk.GetImageFromArray(slice_3d)
                
                # Configure Maurer Distance Map Filter
                sdf_filter = sitk.SignedMaurerDistanceMapImageFilter()
                sdf_filter.SetInsideIsPositive(False)  # Inside = Negative, Outside = Positive
                sdf_filter.SetSquaredDistance(False)   # Linear distance metric
                sdf_filter.SetUseImageSpacing(True)    # Respects isotropic/anisotropic voxel spacing
                
                sdf_img = sdf_filter.Execute(sitk_img)
                arr = sitk.GetArrayFromImage(sdf_img).astype(np.float32)
                
                # Apply smooth Tanh bounding to compress distances safely into [-1.0, 1.0]
                # This prevents distant background voxels from causing giant exploding losses.
                sdf_out[b, c] = np.tanh(arr / scale_factor)
                
        return torch.from_numpy(sdf_out).to(target_tensor.device).float()
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        
        # Generate the continuous 3D target Signed Distance Function at full resolution
        # target[0] is the highest resolution segment map -> shape (B, 1, D, H, W)
        with torch.no_grad():
            gt_sdf = self.compute_sdf_target_3d(target[0], scale_factor=5.0)

        # Forward Pass through our Multi-Head Wrapper
        # seg_outputs: List of predictions matching deep supervision scales
        # pred_sdf: Continuous 3D distance field from our regression head -> shape (B, 1, D, H, W)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            model_outputs = self.network(data)
            # Compute Regional Segmentation Loss (Deeply Supervised Dice + CE)
            loss_dict = self.loss(model_outputs, target, gt_sdf)
            l = loss_dict['total_loss']
        
        # Backward and Optimize
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            
        return {'loss': l.detach().cpu().item()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        
        with torch.no_grad():
            gt_sdf = self.compute_sdf_target_3d(target[0], scale_factor=5.0)
            
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                # Forward pass
                model_outputs = self.network(data)
                del data
                loss_dict = self.loss(model_outputs, target, gt_sdf)
                l = loss_dict['total_loss']

        # only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = model_outputs[0][0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class PositiveDiceLossWrapper(nn.Module):
    """
    Translates nnUNet's native negative Dice (-1.0 to 0.0) 
    into a standard positive cost function (0.0 to 1.0).
    """
    def __init__(self, native_dice_loss):
        super().__init__()
        self.native_dice_loss = native_dice_loss

    def forward(self, net_output, target):
        # Native returns -Dice (e.g., -0.02)
        negative_dice = self.native_dice_loss(net_output, target)
        
        # Shift to positive space: 1.0 + (-Dice) -> 1.0 - Dice (e.g., 0.98)
        return 1.0 + negative_dice

class nnUNetTrainerMultiHeadDCBDLoss(nnUNetTrainer):
    """
    Trainer that combines Dice loss with boundary regression loss.
    """
    lambda_boundary = 0.01

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = DeepSupervisionTwinHeadNNUNet(self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            )).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True

            logger_config_hparas = {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.configuration_manager.batch_size
                }
            self.logger.update_config({"hparas": logger_config_hparas})
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _build_loss(self):
        """
        Builds a customized deep supervision loss targeting ONLY Dice Loss,
        and pairs it smoothly with our independent boundary loss wrapper.
        """
        # Instantiate nnUNet's native, optimized Soft Dice Loss module
        # We copy the exact parameters nnUNet uses internally from your dataset configuration profile
        raw_dice = MemoryEfficientSoftDiceLoss(
            smooth=1e-5,
            do_bg=False,             # Standard setting: ignore background channel optimization
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp
        )

        # Shift it into standard positive range (0.0 to 1.0)
        positive_dice_loss = PositiveDiceLossWrapper(raw_dice)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss

            # Encapsulate the pure Dice Loss within nnUNet's DeepSupervisionWrapper manager
            positive_dice_loss = DeepSupervisionWrapper(
                positive_dice_loss, 
                weights
            )

        # Route the resulting loss directly into your twin-head tracking layer
        composite_loss = TwinHeadSeparateLossWrapper(
            native_nnunet_ds_loss=positive_dice_loss,
            lambda_boundary=self.lambda_boundary
        )

        return composite_loss
    
    def compute_sdf_target_3d(self, target_tensor: torch.Tensor, scale_factor: float = 5.0) -> torch.Tensor:
        """
        Computes a bounded, smooth 3D Signed Distance Function from a binary mask.
        Using same signal definitions as nnSAM (https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17481).
        Input target_tensor: Full-resolution target on CPU/GPU -> shape (B, 1, D, H, W)
        Output: Smoothed SDF tensor on the same device -> shape (B, 1, D, H, W)
        """
        # Move to numpy for SimpleITK processing
        mask_np = target_tensor.cpu().numpy().astype(np.uint8)
        B, C, D, H, W = mask_np.shape
        
        sdf_out = np.zeros_like(mask_np, dtype=np.float32)
        
        for b in range(B):
            for c in range(C):
                slice_3d = mask_np[b, c]
                
                # If the patch doesn't contain any airway voxels, assign a large maximum distance
                if not slice_3d.any():
                    sdf_out[b, c] = np.ones_like(slice_3d, dtype=np.float32) * scale_factor
                    continue
                    
                # Convert to SimpleITK Image
                sitk_img = sitk.GetImageFromArray(slice_3d)
                
                # Configure Maurer Distance Map Filter
                sdf_filter = sitk.SignedMaurerDistanceMapImageFilter()
                sdf_filter.SetInsideIsPositive(False)  # Inside = Negative, Outside = Positive
                sdf_filter.SetSquaredDistance(False)   # Linear distance metric
                sdf_filter.SetUseImageSpacing(True)    # Respects isotropic/anisotropic voxel spacing
                
                sdf_img = sdf_filter.Execute(sitk_img)
                arr = sitk.GetArrayFromImage(sdf_img).astype(np.float32)
                
                # Apply smooth Tanh bounding to compress distances safely into [-1.0, 1.0]
                # This prevents distant background voxels from causing giant exploding losses.
                sdf_out[b, c] = np.tanh(arr / scale_factor)
                
        return torch.from_numpy(sdf_out).to(target_tensor.device).float()
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        
        # Generate the continuous 3D target Signed Distance Function at full resolution
        # target[0] is the highest resolution segment map -> shape (B, 1, D, H, W)
        with torch.no_grad():
            gt_sdf = self.compute_sdf_target_3d(target[0], scale_factor=5.0)

        # Forward Pass through our Multi-Head Wrapper
        # seg_outputs: List of predictions matching deep supervision scales
        # pred_sdf: Continuous 3D distance field from our regression head -> shape (B, 1, D, H, W)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            model_outputs = self.network(data)
            # Compute Regional Segmentation Loss (Deeply Supervised Dice + CE)
            loss_dict = self.loss(model_outputs, target, gt_sdf)
            l = loss_dict['total_loss']
        
        # Backward and Optimize
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0) #12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0) #12)
            self.optimizer.step()
            
        return {'loss': l.detach().cpu().item()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        
        with torch.no_grad():
            gt_sdf = self.compute_sdf_target_3d(target[0], scale_factor=5.0)
            
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                # Forward pass
                model_outputs = self.network(data)
                del data
                loss_dict = self.loss(model_outputs, target, gt_sdf)
                l = loss_dict['total_loss']

        # only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = model_outputs[0][0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

""" Variations of lambda hyperparameter for combining baseline loss with boundary regression """
class nnUNetTrainerMultiHead001BDLoss(nnUNetTrainerMultiHeadBDLoss):
    lambda_boundary = 0.01

class nnUNetTrainerMultiHead005BDLoss(nnUNetTrainerMultiHeadBDLoss):
    lambda_boundary = 0.05

class nnUNetTrainerMultiHead0005BDLoss(nnUNetTrainerMultiHeadBDLoss):
    lambda_boundary = 0.005

class nnUNetTrainerMultiHead008BDLoss(nnUNetTrainerMultiHeadBDLoss):
    lambda_boundary = 0.08

""" Variations of lambda hyperparameter for combining dice loss with boundary regression """
class nnUNetTrainerMultiHead001DCBDLoss(nnUNetTrainerMultiHeadDCBDLoss):
    lambda_boundary = 0.01

class nnUNetTrainerMultiHead005DCBDLoss(nnUNetTrainerMultiHeadDCBDLoss):
    lambda_boundary = 0.05

class nnUNetTrainerMultiHead0005DCBDLoss(nnUNetTrainerMultiHeadDCBDLoss):
    lambda_boundary = 0.005

class nnUNetTrainerMultiHead008DCBDLoss(nnUNetTrainerMultiHeadDCBDLoss):
    lambda_boundary = 0.08

class nnUNetTrainerMultiHead00005DCBDLoss(nnUNetTrainerMultiHeadDCBDLoss):
    lambda_boundary = 0.0005
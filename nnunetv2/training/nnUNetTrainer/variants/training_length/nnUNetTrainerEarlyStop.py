from typing import Union, List

import numpy as np
import torch
from torch._dynamo import OptimizedModule

from torch import distributed as dist
from nnunetv2.training.logging.nnunet_logger import MyMetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainerEarlyStop(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), early_stopping: bool = True):

        super().__init__(plans, configuration, fold, dataset_json, device)

        ### Definitions for early stopping
        self.early_stopping = early_stopping
        self.early_stop_thresh = 5e-4
        self.early_stop_patience = 30
        self.slope_window = 15
        self.min_epoch = 150
        self.prev_waiting_count = 0

        del self.logger # will create a new one

        logger_config = {"plans": plans, "configuration": configuration, "fold": fold, "dataset": dataset_json}

        self.logger = MyMetaLogger(self.output_folder, self.continue_training)
        self.logger.update_config(logger_config)

    def check_slope(self, history, wait) -> tuple[bool, float, int]:

        if len(history) < self.slope_window:
            return False, 1, 0 # Keep training

        # Get the last N points
        y = np.array(history[-self.slope_window:])
        # Create x-axis (0, 1, 2, 3, 4)
        x = np.arange(self.slope_window)
        
        # Perform linear regression (degree 1 polynomial)
        # polyfit returns [slope, intercept]
        slope, _ = np.polyfit(x, y, 1)
        
        # For Loss: If slope is flatter than positive threshold
        if slope is None:
            return False, 1, 0 # Keep training
        
        if slope < self.early_stop_thresh:
            wait += 1
        else:
            wait = 0 # Reset if we see a good upward trend

        if wait >= self.early_stop_patience:
            self.print_to_log_file(f"Early stopping triggered! Slope {slope:.6f} stayed below threshold for {self.early_stop_patience} epochs.")
            
            return True, slope, wait
        
        return False, slope, wait
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self.prev_waiting_count = self.logger.get_value('patience_waiting', step=-1)
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def on_validation_epoch_end(self, val_outputs: List[dict]) -> bool:
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # compute slope of validation mean_fg_dice
        if self.early_stopping and self.current_epoch > self.min_epoch:
            stop, slope, self.prev_waiting_count = self.check_slope(self.logger.get_last_x_values('mean_fg_dice', last=self.slope_window), self.prev_waiting_count) # if early stopping
            self.logger.log('patience_waiting', self.prev_waiting_count, self.current_epoch)
            self.print_to_log_file(f"Debug: Slope {slope:.6f} \t Patience count {self.prev_waiting_count}")
            
            return stop
        else:
            self.logger.log('patience_waiting', 0, self.current_epoch)
            return False
    
    def run_training(self):
        self.on_train_start()

        stop_training = False

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                stop_training = self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            if stop_training:
                break

        self.on_train_end()
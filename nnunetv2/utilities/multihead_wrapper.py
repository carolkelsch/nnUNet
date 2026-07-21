import torch.nn as nn
import sys

class DeepSupervisionTwinHeadNNUNet(nn.Module):
    """
    Twin head (regression) for UNet generated architecture to apply boundary loss.
    """
    def __init__(self, nnunet_model):
        super(DeepSupervisionTwinHeadNNUNet, self).__init__()
        self.backbone = nnunet_model
        
        # Target the final layer in the seg_layers list.
        # In nnUNet v2's forward loop execution pass, seg_layers[-1] is the one 
        # that processes the full spatial resolution (e.g., 96x160x160).
        primary_seg_layer = self.backbone.decoder.seg_layers[-1]
        
        # Build boundary tracking head using the correct full-res parameters
        self.boundary_head = nn.Conv3d(
            in_channels=primary_seg_layer.in_channels,
            out_channels=primary_seg_layer.out_channels,
            kernel_size=primary_seg_layer.kernel_size,
            stride=primary_seg_layer.stride,
            padding=primary_seg_layer.padding,
            bias=primary_seg_layer.bias is not None
        )
        
        self._intercepted_features = None
        
        # Register the forward hook onto the input of the full-resolution head
        primary_seg_layer.register_forward_hook(self._hook_segmentation_input)

    def _hook_segmentation_input(self, module, input_tensor, output_tensor):
        """Captures the full-resolution spatial features right before classification"""
        # input_tensor[0] contains the actual feature map tensor
        self._intercepted_features = input_tensor[0]

    @property
    def decoder(self):
        """Exposes the internal decoder to satisfy nnUNet's initialization managers"""
        return self.backbone.decoder

    def _is_called_by_validation(self) -> bool:
        """
        Inspects the Python call stack to check if the current model pass
        was initiated by the trainer's validation loop step.
        """
        # Look back up through the executing frames of code
        frame = sys._getframe()
        while frame:
            if 'validation_step' in frame.f_code.co_name:
                return True
            frame = frame.f_back
        return False

    def forward(self, x):
        # Handle Training and Validation Iterations
        # If we are either training OR inside the validation_step loop, return BOTH heads
        if self.training or self._is_called_by_validation():
            # Let nnUNet natively run its entire pipeline.
            # This triggers seg_layers[-1] internally, saving the full-res features to our hook.
            seg_outputs = self.backbone(x)
            
            # Pull the features caught by our hook
            full_res_features = self._intercepted_features
            
            # Feed those features through the boundary head
            predicted_sdf = self.boundary_head(full_res_features)
            
            # Clear memory reference to avoid VRAM tracking accumulation
            self._intercepted_features = None
            
            return seg_outputs, predicted_sdf
            
        # Handle Test Deployment Inference Passes
        # When called by nnUNetv2_predict, it skips everything else and executes natively.
        else:
            # Clean inference fallback pass
            return self.backbone(x)
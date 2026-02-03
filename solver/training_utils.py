import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ..quantization.quantized_module import QuantizedLayer, QuantizedModule
from ..quantization.fake_quant import QuantizeBase
from ..quantization.observer import ObserverBase


class FlowLoss(nn.Module):
    """Optical flow consistency loss for video matting"""
    
    def __init__(self, lambda_flow: float = 0.05, use_confidence: bool = False):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.use_confidence = use_confidence
    
    def forward(self, pred_sequence: torch.Tensor, flow_sequence: torch.Tensor, 
                confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_sequence: Predicted matte sequence [B, T, C, H, W]
            flow_sequence: Optical flow sequence [B, T-1, 2, H, W]
            confidence: Flow confidence [B, T-1, 1, H, W]
        """
        if len(pred_sequence) <= 1:
            return torch.tensor(0.0, device=pred_sequence.device)
        
        flow_loss = 0.0
        
        for t in range(len(pred_sequence) - 1):
            current_frame = pred_sequence[t]
            next_frame = pred_sequence[t + 1]
            flow = flow_sequence[t]
            
            # Warp next frame back to current frame
            warped_next = self.warp_frame(next_frame, flow)
            
            # Compute flow consistency loss
            frame_loss = F.mse_loss(warped_next, current_frame)
            
            # Apply confidence weighting if available
            if confidence is not None and self.use_confidence:
                frame_loss = (frame_loss * confidence[t]).mean()
            
            flow_loss += frame_loss
        
        return flow_loss * self.lambda_flow
    
    @staticmethod
    def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp frame using optical flow"""
        B, C, H, W = frame.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device),
            torch.arange(W, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add flow to grid
        flow_grid = grid + flow.permute(0, 2, 3, 1)
        
        # Normalize to [-1, 1]
        flow_grid[..., 0] = flow_grid[..., 0] / (W - 1) * 2 - 1
        flow_grid[..., 1] = flow_grid[..., 1] / (H - 1) * 2 - 1
        
        # Sample
        return F.grid_sample(frame, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)


class VideoMatteLoss(nn.Module):
    """Combined loss for video matting quantization"""
    
    def __init__(self, lambda_task: float = 1.0, lambda_flow: float = 0.05, 
                 use_confidence: bool = False):
        super().__init__()
        self.lambda_task = lambda_task
        self.flow_loss = FlowLoss(lambda_flow, use_confidence)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                flow_sequence: Optional[torch.Tensor] = None,
                confidence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Predicted matte [B, C, H, W] or [B, T, C, H, W]
            target: Ground truth matte [B, C, H, W] or [B, T, C, H, W]
            flow_sequence: Optical flow sequence [B, T-1, 2, H, W]
            confidence: Flow confidence [B, T-1, 1, H, W]
        """
        losses = {}
        
        # Task loss (alpha matte MSE)
        if pred.dim() == 5:  # Sequence input
            task_loss = F.mse_loss(pred, target)
        else:  # Single frame
            task_loss = F.mse_loss(pred, target)
        
        losses['task_loss'] = task_loss * self.lambda_task
        
        # Flow consistency loss
        if flow_sequence is not None:
            if pred.dim() == 4:  # Convert single frame to sequence
                pred = pred.unsqueeze(1)
                target = target.unsqueeze(1)
            
            flow_loss_val = self.flow_loss(pred, flow_sequence, confidence)
            losses['flow_loss'] = flow_loss_val
            losses['total_loss'] = losses['task_loss'] + losses['flow_loss']
        else:
            losses['total_loss'] = losses['task_loss']
        
        return losses


class QuantizationAwareTrainer:
    """Trainer for quantization-aware fine-tuning"""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, 
                 optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Move to device
        self.model.to(device)
    
    def train_epoch(self, dataloader, use_flow: bool = False, 
                   flow_model: Optional[nn.Module] = None) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_losses = {}
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            if len(batch) == 3:  # (fgr, pha, bgr)
                fgr, pha, bgr = batch
                fgr, pha, bgr = fgr.to(self.device), pha.to(self.device), bgr.to(self.device)
                
                # Composite input
                comp_input = fgr * pha + bgr * (1 - pha)
                target = comp_input
            else:
                comp_input, target = batch
                comp_input, target = comp_input.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(comp_input, (list, tuple)):
                output = self.model(comp_input)
            else:
                output = self.model(comp_input)
            
            # Compute flow if needed
            flow_sequence = None
            confidence = None
            
            if use_flow and flow_model is not None and isinstance(output, (list, tuple)):
                flow_sequence = []
                confidence = []
                
                for i in range(len(output) - 1):
                    with torch.no_grad():
                        flow, conf = self._compute_flow(output[i], output[i+1], flow_model)
                        flow_sequence.append(flow)
                        confidence.append(conf)
                
                flow_sequence = torch.stack(flow_sequence, dim=1)  # [B, T-1, 2, H, W]
                confidence = torch.stack(confidence, dim=1)  # [B, T-1, 1, H, W]
            
            # Compute loss
            if isinstance(output, (list, tuple)):
                output_tensor = torch.stack(output, dim=1)  # [B, T, C, H, W]
                if isinstance(target, torch.Tensor):
                    target = target.unsqueeze(1).repeat(1, len(output), 1, 1, 1)
            else:
                output_tensor = output
            
            losses = self.criterion(output_tensor, target, flow_sequence, confidence)
            
            # Backward pass
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    @torch.no_grad()
    def _compute_flow(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                     flow_model: nn.Module) -> tuple:
        """Compute optical flow between two frames"""
        try:
            # Assuming RAFT model interface
            if hasattr(flow_model, 'compute_flow'):
                flow, confidence = flow_model.compute_flow(frame1, frame2)
            else:
                # Generic interface
            flow = flow_model(frame1, frame2)
            confidence = torch.ones(flow.shape[0], 1, flow.shape[2], flow.shape[3], 
                                   device=flow.device)
            
            return flow, confidence
        except Exception as e:
            print(f"Flow computation failed: {e}")
            # Return zero flow as fallback
            B, C, H, W = frame1.shape
            zero_flow = torch.zeros(B, 2, H, W, device=frame1.device)
            confidence = torch.ones(B, 1, H, W, device=frame1.device)
            return zero_flow, confidence
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_losses = {}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack and move to device (same as training)
                if len(batch) == 3:
                    fgr, pha, bgr = batch
                    fgr, pha, bgr = fgr.to(self.device), pha.to(self.device), bgr.to(self.device)
                    comp_input = fgr * pha + bgr * (1 - pha)
                    target = comp_input
                else:
                    comp_input, target = batch
                    comp_input, target = comp_input.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(comp_input)
                
                # Compute loss (simplified for validation)
                if isinstance(output, (list, tuple)):
                    output_tensor = torch.stack(output, dim=1)
                    target = target.unsqueeze(1).repeat(1, len(output), 1, 1, 1)
                else:
                    output_tensor = output
                
                losses = self.criterion(output_tensor, target)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses


def create_quantization_config(bit_width: int, symmetric: bool = True) -> Dict[str, Any]:
    """Create quantization configuration"""
    return {
        'bit': bit_width,
        'symmetric': symmetric,
        'ch_axis': -1,
        'quant_min': -(2 ** (bit_width - 1)) if symmetric else 0,
        'quant_max': (2 ** (bit_width - 1) - 1) if symmetric else (2 ** bit_width - 1)
    }


def setup_quantization(model: nn.Module, w_config: Dict, a_config: Dict) -> nn.Module:
    """Setup quantization for model"""
    from ..solver.fold_bn import search_fold_and_remove_bn
    
    # Remove batch normalization
    search_fold_and_remove_bn(model)
    
    # Replace layers with quantized versions
    from ..solver.main_ptq4vm import quantize_model
    model = quantize_model(model, w_config, a_config)
    
    return model

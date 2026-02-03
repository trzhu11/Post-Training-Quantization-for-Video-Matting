# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import sys
import time
import random
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# PTQ4VM imports
from quantization.quantized_module import QuantizedLayer, QuantizedModule
from solver.fold_bn import StraightThrough
from solver.videomatte_utils import load_data, set_seed
from quantization.state import enable_quantization, disable_all
from quantization.fake_quant import QuantizeBase
from quantization.observer import ObserverBase
from solver.recon import reconstruction
from model import load_model, specials

# RAFT imports (optional)
try:
    from RAFT.core.raft import RAFT
    from RAFT.core.utils.utils import InputPadder
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("Warning: RAFT not available. Flow-based features will be disabled.")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + ')'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays training progress"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def compute_optical_flow_raft(frame1, frame2, flow_model, iters=12):
    """Compute optical flow using RAFT"""
    if not RAFT_AVAILABLE or flow_model is None:
        return None, None
    
    B, C, H, W = frame1.shape
    padder = InputPadder(frame1.shape)
    image1, image2 = padder.pad(frame1 * 255.0, frame2 * 255.0)
    
    flow_low, flow_up = flow_model(image1, image2, iters=iters, test_mode=True)
    flow_1_to_2 = padder.unpad(flow_up)
    confidence = torch.ones(B, 1, H, W, device=frame1.device)
    
    return flow_1_to_2, confidence


def apply_flow_warp(tensor_to_warp, flow_1_to_2):
    """Apply flow warping to tensor"""
    B, C, H, W = tensor_to_warp.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=tensor_to_warp.device),
        torch.arange(W, device=tensor_to_warp.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    
    flow_grid = grid + flow_1_to_2.permute(0, 2, 3, 1)
    flow_grid[..., 0] = flow_grid[..., 0] / (W - 1) * 2 - 1
    flow_grid[..., 1] = flow_grid[..., 1] / (H - 1) * 2 - 1
    
    return F.grid_sample(tensor_to_warp, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)


def quantize_model(model, w_qconfig, a_qconfig):
    """Replace model layers with quantized equivalents"""
    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        
        while st < ed:
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardswish, nn.Tanh)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1
    
    replace_module(model, w_qconfig, a_qconfig, qoutput=False)
    return model


def get_calibration_data(train_loader, num_samples):
    """Get calibration data from train loader"""
    fgr_samples, pha_samples, bgr_samples = [], [], []
    
    for batch in train_loader:
        true_fgr, true_pha, true_bgr = batch[0], batch[1], batch[2]
        fgr_samples.append(true_fgr)
        pha_samples.append(true_pha)
        bgr_samples.append(true_bgr)
        
        if len(fgr_samples) * true_fgr.size(0) >= num_samples:
            break
    
    truncate = lambda x: torch.cat(x, dim=0)[:num_samples]
    return truncate(fgr_samples), truncate(pha_samples), truncate(bgr_samples)


def calibrate_model(model, cali_data):
    """Calibrate quantization ranges"""
    model.eval()
    with torch.no_grad():
        # Calibrate activations
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                module.enable_observer()
                module.disable_fake_quant()
        
        model(cali_data[:1].cuda())
        
        # Calibrate weights
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                if 'weight' in name:
                    module.enable_observer()
                    module.disable_fake_quant()
        
        model(cali_data[:1].cuda())
        
        # Enable quantization
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                module.disable_observer()
                module.enable_fake_quant()


def reconstruct_model(model, fp_model, cali_data, recon_config):
    """Reconstruct quantized model weights"""
    def recon_module(module: nn.Module, fp_module: nn.Module):
        for name, child_module in module.named_children():
            if isinstance(child_module, (QuantizedLayer, QuantizedModule)):
                reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, recon_config)
            else:
                recon_module(child_module, getattr(fp_module, name))
    
    recon_module(model, fp_model)


def train_quantized_model(model, train_loader, args):
    """Fine-tune quantized model with flow guidance"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Load RAFT model if flow is enabled
    flow_model = None
    if args.use_flow and RAFT_AVAILABLE:
        try:
            flow_model = RAFT(args)
            flow_model.load_state_dict(torch.load(args.raft_checkpoint))
            flow_model = flow_model.cuda()
            flow_model.eval()
        except Exception as e:
            print(f"Failed to load RAFT model: {e}")
            args.use_flow = False
    
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(train_loader), [losses], prefix='Epoch: [{}]'.format(0))
    
    for epoch in range(args.epochs):
        losses.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            fgr, pha, bgr = batch
            fgr, pha, bgr = fgr.cuda(), pha.cuda(), bgr.cuda()
            
            # Composite input
            comp = fgr * pha + bgr * (1 - pha)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(comp)
            
            # Basic reconstruction loss
            loss = F.mse_loss(output, comp)
            
            # Add flow consistency loss if enabled
            if args.use_flow and flow_model is not None and len(output) > 1:
                flow_loss = 0.0
                for i in range(len(output) - 1):
                    flow, confidence = compute_optical_flow_raft(output[i], output[i+1], flow_model, args.raft_iters)
                    if flow is not None:
                        warped_next = apply_flow_warp(output[i], flow)
                        flow_loss += F.mse_loss(warped_next, output[i+1]) * args.flow_lambda
                
                loss += flow_loss
            
            loss.backward()
            optimizer.step()
            
            losses.update(loss.item(), fgr.size(0))
            
            if batch_idx % args.log_freq == 0:
                progress.display(batch_idx)
        
        print(f'Epoch {epoch}: Loss {losses.avg:.4f}')


def main(args):
    """Main training function"""
    # Set random seed
    set_seed(args.seed)
    
    # Device setup
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)
    
    # Load data
    print("Loading calibration data...")
    train_loader = load_data(
        videomatte_dir_train=args.videomatte_dir_train,
        background_video_dir_train=args.background_video_dir_train,
        size=args.resolution,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    cali_fgr, cali_pha, cali_bgr = get_calibration_data(train_loader, args.num_cali_sequences * args.seq_length)
    cali_data = cali_fgr * cali_pha + cali_bgr * (1 - cali_pha)
    
    # Load and quantize model
    print("Loading and quantizing model...")
    model = load_model(args.model_config)
    model = quantize_model(model, args.w_qconfig, args.a_qconfig)
    model.cuda()
    
    # Create FP model for reconstruction
    fp_model = load_model(args.model_config)
    fp_model.cuda()
    fp_model.eval()
    
    # Set observer names
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)
    
    # Calibration
    print("Calibrating model...")
    calibrate_model(model, cali_data)
    
    # Reconstruction
    if hasattr(args, 'recon_config'):
        print("Reconstructing model...")
        reconstruct_model(model, fp_model, cali_data, args.recon_config)
    
    # Fine-tuning
    if args.finetune:
        print("Fine-tuning quantized model...")
        train_quantized_model(model, train_loader, args)
    
    # Save model
    save_path = os.path.join(args.save_dir, f"ptq4vm_model_{args.w_qconfig['bit']}bit.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTQ4VM: Post-Training Quantization for Video Matting')
    
    # Model and data
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration file')
    parser.add_argument('--videomatte_dir_train', type=str, required=True, help='VideoMatte training data directory')
    parser.add_argument('--background_video_dir_train', type=str, required=True, help='Background video training data directory')
    
    # Quantization config
    parser.add_argument('--w_bit', type=int, default=4, help='Weight quantization bit width')
    parser.add_argument('--a_bit', type=int, default=4, help='Activation quantization bit width')
    
    # Training config
    parser.add_argument('--resolution', type=int, default=512, help='Input resolution')
    parser.add_argument('--seq_length', type=int, default=4, help='Sequence length')
    parser.add_argument('--num_cali_sequences', type=int, default=64, help='Number of calibration sequences')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    
    # Fine-tuning config
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning')
    parser.add_argument('--epochs', type=int, default=10, help='Fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_freq', type=int, default=50, help='Log frequency')
    
    # Flow config
    parser.add_argument('--use_flow', action='store_true', help='Use optical flow guidance')
    parser.add_argument('--flow_lambda', type=float, default=0.05, help='Flow loss weight')
    parser.add_argument('--raft_checkpoint', type=str, help='RAFT model checkpoint path')
    parser.add_argument('--raft_iters', type=int, default=12, help='RAFT iterations')
    
    # Output config
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Model save directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create quantization configs
    args.w_qconfig = {'bit': args.w_bit, 'symmetric': True}
    args.a_qconfig = {'bit': args.a_bit, 'symmetric': True}
    
    main(args)

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import os


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics for video matting"""
    # Ensure tensors are on CPU and in numpy format
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Squeeze batch dimension if present
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
    
    # Convert to [0, 255] range for some metrics
    pred_255 = (pred * 255).astype(np.uint8)
    target_255 = (target * 255).astype(np.uint8)
    
    metrics = {}
    
    # MSE
    metrics['mse'] = np.mean((pred - target) ** 2)
    
    # PSNR
    if metrics['mse'] > 0:
        metrics['psnr'] = 20 * np.log10(1.0 / np.sqrt(metrics['mse']))
    else:
        metrics['psnr'] = 100.0
    
    # SAD (Sum of Absolute Differences)
    metrics['sad'] = np.sum(np.abs(pred - target))
    
    # Gradient error
    pred_grad = np.gradient(pred)
    target_grad = np.gradient(target)
    grad_error = np.sqrt((pred_grad[0] - target_grad[0])**2 + (pred_grad[1] - target_grad[1])**2)
    metrics['gradient_error'] = np.mean(grad_error)
    
    # Connectivity error (simplified)
    metrics['connectivity_error'] = compute_connectivity_error(pred_255, target_255)
    
    return metrics


def compute_connectivity_error(pred: np.ndarray, target: np.ndarray, radius: int = 4) -> float:
    """Compute connectivity error for matting evaluation"""
    h, w = pred.shape
    
    # Create binary masks
    pred_bin = (pred > 128).astype(np.uint8)
    target_bin = (target > 128).astype(np.uint8)
    
    # Compute connectivity errors
    errors = []
    
    for y in range(h):
        for x in range(w):
            # Check 4-connectivity
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbors.append((ny, nx))
            
            if neighbors:
                # Compute connectivity difference
                pred_conn = sum(pred_bin[ny, nx] for ny, nx in neighbors) / len(neighbors)
                target_conn = sum(target_bin[ny, nx] for ny, nx in neighbors) / len(neighbors)
                errors.append(abs(pred_conn - target_conn))
    
    return np.mean(errors) if errors else 0.0


def evaluate_video_sequence(model: torch.nn.Module, dataloader, device: torch.device, 
                          save_dir: Optional[str] = None) -> Dict[str, float]:
    """Evaluate model on video sequence"""
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            if len(batch) == 3:
                fgr, pha, bgr = batch
                fgr, pha, bgr = fgr.to(device), pha.to(device), bgr.to(device)
                comp_input = fgr * pha + bgr * (1 - pha)
                target = comp_input
            else:
                comp_input, target = batch
                comp_input, target = comp_input.to(device), target.to(device)
            
            # Forward pass
            output = model(comp_input)
            
            # Handle sequence output
            if isinstance(output, (list, tuple)):
                output = output[-1]  # Use last frame
            
            # Compute metrics for each sample in batch
            for i in range(comp_input.shape[0]):
                sample_metrics = compute_metrics(output[i], target[i])
                all_metrics.append(sample_metrics)
                
                # Save results if requested
                if save_dir:
                    save_sample_results(comp_input[i], output[i], target[i], 
                                      batch_idx, i, save_dir)
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        return avg_metrics
    else:
        return {}


def save_sample_results(input_img: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                       batch_idx: int, sample_idx: int, save_dir: str):
    """Save sample results for visual inspection"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and denormalize
    input_np = input_img.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Handle different channel arrangements
    if input_np.shape[0] == 3:  # RGB
        input_np = np.transpose(input_np, (1, 2, 0))
    if pred_np.shape[0] == 3:
        pred_np = np.transpose(pred_np, (1, 2, 0))
    if target_np.shape[0] == 3:
        target_np = np.transpose(target_np, (1, 2, 0))
    
    # Clip to valid range
    input_np = np.clip(input_np, 0, 1)
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # Convert to uint8
    input_uint8 = (input_np * 255).astype(np.uint8)
    pred_uint8 = (pred_np * 255).astype(np.uint8)
    target_uint8 = (target_np * 255).astype(np.uint8)
    
    # Save images
    filename = f"batch{batch_idx}_sample{sample_idx}"
    
    Image.fromarray(input_uint8).save(os.path.join(save_dir, f"{filename}_input.png"))
    Image.fromarray(pred_uint8).save(os.path.join(save_dir, f"{filename}_pred.png"))
    Image.fromarray(target_uint8).save(os.path.join(save_dir, f"{filename}_target.png"))
    
    # Save side-by-side comparison
    if input_uint8.shape[-1] == 3:  # RGB images
        comparison = np.hstack([input_uint8, pred_uint8, target_uint8])
    else:  # Grayscale (alpha matte)
        comparison = np.hstack([input_uint8, pred_uint8, target_uint8])
    
    Image.fromarray(comparison).save(os.path.join(save_dir, f"{filename}_comparison.png"))


def compute_temporal_consistency(sequence: torch.Tensor) -> Dict[str, float]:
    """Compute temporal consistency metrics for video sequence"""
    if sequence.dim() != 5:  # [B, T, C, H, W]
        return {}
    
    B, T, C, H, W = sequence.shape
    
    metrics = {}
    
    # Compute temporal differences
    temporal_diffs = []
    for t in range(T - 1):
        diff = torch.abs(sequence[:, t+1] - sequence[:, t])
        temporal_diffs.append(diff.mean().item())
    
    metrics['avg_temporal_diff'] = np.mean(temporal_diffs)
    metrics['max_temporal_diff'] = np.max(temporal_diffs)
    metrics['temporal_std'] = np.std(temporal_diffs)
    
    return metrics


def profile_model(model: torch.nn.Module, input_shape: Tuple[int, ...], device: torch.device) -> Dict[str, float]:
    """Profile model performance"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    import time
    num_runs = 100
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    # Compute FPS
    batch_size = input_shape[0]
    fps = batch_size / avg_time
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_inference_time': avg_time,
        'fps': fps,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved
    }


def analyze_quantization_error(fp_model: torch.nn.Module, quant_model: torch.nn.Module,
                              dataloader, device: torch.device) -> Dict[str, float]:
    """Analyze quantization error between FP and quantized models"""
    fp_model.eval()
    quant_model.eval()
    
    total_error = 0
    max_error = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Prepare input
            if len(batch) == 3:
                fgr, pha, bgr = batch
                fgr, pha, bgr = fgr.to(device), pha.to(device), bgr.to(device)
                comp_input = fgr * pha + bgr * (1 - pha)
            else:
                comp_input, _ = batch
                comp_input = comp_input.to(device)
            
            # Forward pass
            fp_output = fp_model(comp_input)
            quant_output = quant_model(comp_input)
            
            # Handle sequence output
            if isinstance(fp_output, (list, tuple)):
                fp_output = fp_output[-1]
            if isinstance(quant_output, (list, tuple)):
                quant_output = quant_output[-1]
            
            # Compute error
            error = torch.abs(fp_output - quant_output).mean().item()
            total_error += error
            max_error = max(max_error, error)
            num_samples += comp_input.shape[0]
    
    return {
        'avg_quantization_error': total_error / num_samples,
        'max_quantization_error': max_error,
        'num_samples': num_samples
    }

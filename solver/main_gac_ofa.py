# -*- coding: utf-8 -*-
"""
Stage 2: BN Affine Fine-tuning + OFA Optical Flow Loss
=======================================================
Loads Stage 1 quantized model, adds learnable affine params to
QuantizedLayers, and fine-tunes with combined Alpha MSE + Flow L1 loss.

Usage:
    python -m solver.main_bn_flow --config configs/stage2_bn_flow_w4a4.yaml
"""

import sys
import os
import time
import datetime
import argparse
import types
import random
import traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from typing import List, Tuple, Optional, Callable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from easydict import EasyDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from solver.videomatte_utils import load_data, set_seed
from solver.fold_bn import StraightThrough
from quantization.quantized_module import QuantizedLayer
from quantization.state import enable_quantization
from quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize
from inference_utils import ImageSequenceReader, ImageSequenceWriter

# RAFT (optional)
RAFT_INSTALLED = False
try:
    raft_path = os.environ.get('RAFT_PATH', '')
    if raft_path and raft_path not in sys.path:
        sys.path.insert(0, raft_path)
    from RAFT.core.raft import RAFT
    from RAFT.core.utils.utils import InputPadder
    RAFT_INSTALLED = True
except ImportError:
    pass


# ====================== Utilities ======================

class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ====================== Optical Flow ======================

@torch.no_grad()
def compute_optical_flow(frame1, frame2, flow_model, raft_iters=12):
    padder = InputPadder(frame1.shape)
    image1, image2 = padder.pad(frame1 * 255.0, frame2 * 255.0)
    _, flow_up = flow_model(image1, image2, iters=raft_iters, test_mode=True)
    flow = padder.unpad(flow_up)
    B, C, H, W = frame1.shape
    confidence = torch.ones(B, 1, H, W, device=frame1.device)
    return flow, confidence


def apply_flow_warp(tensor, flow):
    B, C, H, W = tensor.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    flow_uv = flow.permute(0, 2, 3, 1)
    scale = torch.tensor([W, H], device=flow.device).view(1, 1, 1, 2)
    flow_normalized = flow_uv * (2.0 / scale)
    sample_coords = grid + flow_normalized
    return F.grid_sample(
        tensor, sample_coords, mode="bilinear", padding_mode="border", align_corners=False
    )


# ====================== Data Loading ======================

def get_cali_data_sequences(train_loader, num_sequences, seq_length):
    fgr_seqs, pha_seqs, bgr_seqs = [], [], []
    collected = 0
    pbar = tqdm(total=num_sequences, desc="Collecting calibration sequences")
    for batch in train_loader:
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 3):
            continue
        true_fgr, true_pha, true_bgr = batch[0], batch[1], batch[2]
        if true_fgr.dim() != 5 or true_fgr.shape[1] != seq_length:
            continue
        bs = true_fgr.size(0)
        take = min(bs, num_sequences - collected)
        if take <= 0:
            break
        fgr_seqs.append(true_fgr[:take].cpu())
        pha_seqs.append(true_pha[:take].cpu())
        bgr_seqs.append(true_bgr[:take].cpu())
        collected += take
        pbar.update(take)
        if collected >= num_sequences:
            break
    pbar.close()
    if not fgr_seqs:
        raise RuntimeError("Failed to collect any calibration sequences.")
    fgr = torch.cat(fgr_seqs, dim=0)
    pha = torch.cat(pha_seqs, dim=0)
    bgr = torch.cat(bgr_seqs, dim=0)
    src = fgr * pha + bgr * (1 - pha)
    print(f"Collected {src.size(0)} sequences. src={src.shape}, pha={pha.shape}")
    return src, pha


# ====================== Model Patching ======================

def patch_model_and_collect_params(model, target_class=QuantizedLayer):
    """
    Add learnable scalar affine params (gamma', beta') to bn-folded
    QuantizedLayers and collect activation quantizer scale params.
    """
    affine_params = []
    act_scale_params = []
    device = next(iter(model.parameters())).device
    patched = 0

    for name, module in model.named_modules():
        if isinstance(module, target_class):
            if getattr(module, "bn_folded", False) and not hasattr(module, "affine_enabled_runtime"):
                gamma = nn.Parameter(torch.tensor(1.0, device=device))
                beta = nn.Parameter(torch.tensor(0.0, device=device))
                module.gamma_prime_runtime = gamma
                module.beta_prime_runtime = beta
                module.affine_enabled_runtime = True
                affine_params.extend([gamma, beta])

                def new_forward(self, x, *args, **kwargs):
                    x_orig = self.module(x)
                    if getattr(self, "affine_enabled_runtime", False):
                        x_orig = x_orig * self.gamma_prime_runtime + self.beta_prime_runtime
                    if self.activation is not None and not isinstance(self.activation, StraightThrough):
                        x_orig = self.activation(x_orig)
                    if self.qoutput and hasattr(self, "layer_post_act_fake_quantize") and self.layer_post_act_fake_quantize is not None:
                        x_orig = self.layer_post_act_fake_quantize(x_orig)
                    return x_orig

                module.forward = types.MethodType(new_forward, module)
                patched += 1

        # Collect activation scale params
        if "post_act_fake_quantize" in name and hasattr(module, "scale") and isinstance(module.scale, nn.Parameter):
            if module.scale not in act_scale_params:
                act_scale_params.append(module.scale)
            if isinstance(module, LSQPlusFakeQuantize) and hasattr(module, "zero_point") and isinstance(module.zero_point, nn.Parameter):
                if module.zero_point not in act_scale_params:
                    act_scale_params.append(module.zero_point)
        elif isinstance(module, target_class):
            for attr_name in ["layer_post_act_fake_quantize", "act_quantizer"]:
                quant = getattr(module, attr_name, None)
                if quant is not None and hasattr(quant, "scale") and isinstance(quant.scale, nn.Parameter):
                    if quant.scale not in act_scale_params:
                        act_scale_params.append(quant.scale)

    print(f"Patched {patched} QuantizedLayers with affine params.")
    print(f"Found {len(affine_params)} affine params, {len(act_scale_params)} act scale params.")
    return affine_params, act_scale_params


def freeze_model_except(model, trainable_params):
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for p in trainable_params:
        p.requires_grad = True
        count += p.numel()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False
    print(f"Total trainable elements: {count}")


# ====================== Loss Function ======================

def get_combined_loss_fn(device, flow_lambda, task_mse_lambda):
    mse_loss = nn.MSELoss(reduction="mean").to(device)

    def combined_loss(pred_pha_seq, true_pha_seq, flows, confs):
        task_loss = mse_loss(pred_pha_seq, true_pha_seq)
        flow_loss = torch.tensor(0.0, device=device)

        if flows and pred_pha_seq.shape[1] > 1:
            T = pred_pha_seq.shape[1]
            flow_sum = 0
            for t in range(T - 1):
                warped = apply_flow_warp(pred_pha_seq[:, t + 1], flows[t])
                diff = torch.abs(pred_pha_seq[:, t] - warped)
                flow_sum += torch.mean(diff)
            flow_loss = flow_sum / (T - 1)

        total = task_mse_lambda * task_loss + flow_lambda * flow_loss
        return total, task_loss, flow_loss

    return combined_loss


# ====================== Evaluation ======================

def auto_downsample_ratio(h, w):
    return min(512 / max(h, w), 1)


def evaluate_model(model, eval_input_root, eval_input_resize, output_dir, device):
    print(f"Running evaluation inference -> {output_dir}")
    model.eval()
    enable_quantization(model)
    transform = transforms.Compose([transforms.Resize(eval_input_resize), transforms.ToTensor()])
    processed = 0
    for subset in sorted(os.listdir(eval_input_root)):
        subset_path = os.path.join(eval_input_root, subset)
        if not os.path.isdir(subset_path):
            continue
        for clip in sorted(os.listdir(subset_path)):
            clip_path = os.path.join(subset_path, clip)
            input_source = os.path.join(clip_path, "com")
            if not os.path.isdir(input_source):
                continue
            out_pha = os.path.join(output_dir, subset, clip, "pha")
            out_fgr = os.path.join(output_dir, subset, clip, "fgr")
            os.makedirs(out_pha, exist_ok=True)
            os.makedirs(out_fgr, exist_ok=True)
            try:
                source = ImageSequenceReader(input_source, transform)
                reader = DataLoader(source, batch_size=4, pin_memory=True, num_workers=0)
                writer_pha = ImageSequenceWriter(out_pha, "png")
                writer_fgr = ImageSequenceWriter(out_fgr, "png")
                rec = [None] * 4
                with torch.no_grad():
                    for src_batch in reader:
                        for i in range(src_batch.shape[0]):
                            frame = src_batch[i:i + 1].to(device, dtype=torch.float32)
                            h, w = frame.shape[2:]
                            dr = auto_downsample_ratio(h, w)
                            out = model(frame, *rec, downsample_ratio=dr)
                            fgr_t, pha_t = out[0], out[1]
                            rec = list(out[2:]) if len(out) > 2 else [None] * 4
                            writer_pha.write(pha_t.cpu())
                            writer_fgr.write(fgr_t.cpu())
                writer_pha.close()
                writer_fgr.close()
                processed += 1
            except Exception as e:
                print(f"  Error processing {subset}/{clip}: {e}")
    print(f"Evaluation done. Processed {processed} clips.")


# ====================== Metrics ======================

class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred - true).mean() * 1e3

class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3

class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self._gauss_filter(sigma)
    def __call__(self, pred, true):
        pred_n = np.zeros_like(pred); true_n = np.zeros_like(true)
        cv2.normalize(pred, pred_n, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_n, 1., 0., cv2.NORM_MINMAX)
        tg = self._gauss_gradient(true_n).astype(np.float32)
        pg = self._gauss_gradient(pred_n).astype(np.float32)
        return ((tg - pg) ** 2).sum() / 1000
    def _gauss_gradient(self, img):
        gx = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(gx**2 + gy**2)
    @staticmethod
    def _gauss_filter(sigma, eps=1e-2):
        hs = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * eps)))
        size = int(2 * hs + 1)
        fx = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                g = np.exp(-(i - hs)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
                dg = -(j - hs) * np.exp(-(j - hs)**2 / (2 * sigma**2)) / (sigma**3 * np.sqrt(2 * np.pi))
                fx[i, j] = g * dg
        norm = np.sqrt((fx**2).sum())
        fx /= norm
        return fx, fx.T

class MetricCONN:
    def __call__(self, pred, true):
        step = 0.1; ts = np.arange(0, 1 + step, step)
        rdm = -np.ones_like(true)
        for i in range(1, len(ts)):
            tt = true >= ts[i]; pt = pred >= ts[i]
            inter = (tt & pt).astype(np.uint8)
            _, out, stats, _ = cv2.connectedComponentsWithStats(inter, connectivity=4)
            sz = stats[1:, -1]; omega = np.zeros_like(true)
            if len(sz): omega[out == np.argmax(sz) + 1] = 1
            rdm[(rdm == -1) & (omega == 0)] = ts[i - 1]
        rdm[rdm == -1] = 1
        tp = 1 - (true - rdm) * ((true - rdm) >= 0.15)
        pp = 1 - (pred - rdm) * ((pred - rdm) >= 0.15)
        return np.sum(np.abs(tp - pp)) / 1000

class MetricDTSSD:
    def __call__(self, pt, ptm1, tt, ttm1):
        if ptm1 is None or ttm1 is None:
            return 0.0
        d = ((pt - ptm1) - (tt - ttm1)) ** 2
        return np.sqrt(np.sum(d) / tt.size) * 1e2


def compute_metrics(pred_dir, true_dir):
    metrics_list = ["pha_mad", "pha_mse", "pha_grad", "pha_conn", "pha_dtssd"]
    calcs = {"mad": MetricMAD(), "mse": MetricMSE(), "grad": MetricGRAD(),
             "conn": MetricCONN(), "dtssd": MetricDTSSD()}
    all_metrics = {m: [] for m in metrics_list}

    for dataset in sorted(os.listdir(pred_dir)):
        dp = os.path.join(pred_dir, dataset)
        dt = os.path.join(true_dir, dataset)
        if not os.path.isdir(dp) or not os.path.isdir(dt):
            continue
        for clip in sorted(os.listdir(dp)):
            pp = os.path.join(dp, clip, "pha")
            tp = os.path.join(dt, clip, "pha")
            if not os.path.isdir(pp) or not os.path.isdir(tp):
                continue
            frames = sorted(os.listdir(pp))
            pred_tm1 = true_tm1 = None
            for fn in frames:
                pf = os.path.join(pp, fn); tf = os.path.join(tp, fn)
                if not os.path.exists(tf):
                    continue
                pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                true = cv2.imread(tf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                all_metrics["pha_mad"].append(calcs["mad"](pred, true))
                all_metrics["pha_mse"].append(calcs["mse"](pred, true))
                all_metrics["pha_grad"].append(calcs["grad"](pred, true))
                all_metrics["pha_conn"].append(calcs["conn"](pred, true))
                all_metrics["pha_dtssd"].append(calcs["dtssd"](pred, pred_tm1, true, true_tm1))
                pred_tm1 = pred; true_tm1 = true

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for m in metrics_list:
        vals = all_metrics[m]
        if vals:
            print(f"  {m}: {np.mean(vals):.4f}")
    print("=" * 50)


# ====================== Main ======================

def main(config_path, gpu_id=None):
    with open(config_path) as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if gpu_id is not None:
        cfg.gpu_id = gpu_id
    device = torch.device(f"cuda:{cfg.gpu_id}")
    torch.cuda.set_device(device)
    set_seed(cfg.get('seed', 42))

    print("=" * 60)
    print("Stage 2: BN Affine Fine-tuning + OFA Flow Loss")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {cfg.quantized_model_path}")
    print(f"Epochs: {cfg.epochs}, LR: {cfg.lr}, Flow lambda: {cfg.flow_lambda}")

    # ---- 1. Load calibration sequences ----
    print("[1/7] Loading calibration sequences...")
    loader = load_data(
        videomatte_dir_train=cfg.videomatte_dir_train,
        background_video_dir_train=cfg.background_video_dir_train,
        size=cfg.resolution, seq_length=cfg.seq_length,
        batch_size=cfg.batch_size * 2, num_workers=cfg.get('workers', 4),
        pin_memory=False,
    )
    cali_src, cali_pha = get_cali_data_sequences(loader, cfg.num_cali_sequences, cfg.seq_length)
    del loader
    num_seqs = cali_src.size(0)

    # ---- 2. Precompute optical flow ----
    should_use_flow = not cfg.get('disable_flow', False) and RAFT_INSTALLED and cfg.seq_length >= 2
    cali_flows, cali_confs = [], []

    if should_use_flow:
        print("[2/7] Loading RAFT and precomputing optical flow...")
        raft_args = argparse.Namespace(
            small=cfg.get('raft_small', False),
            mixed_precision=cfg.get('raft_mixed_precision', False),
            alternate_corr=False
        )
        flow_model = RAFT(raft_args)
        state_dict = torch.load(cfg.flow_model_path, map_location=device)
        if any(k.startswith("module.") for k in state_dict):
            from collections import OrderedDict
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        flow_model.load_state_dict(state_dict)
        flow_model.to(device).eval()

        flow_bs = max(cfg.batch_size, 8)
        with torch.no_grad():
            for t in tqdm(range(cfg.seq_length - 1), desc="Precomputing flow"):
                flows_t, confs_t = [], []
                for i in range(0, num_seqs, flow_bs):
                    end = min(i + flow_bs, num_seqs)
                    f1 = cali_src[i:end, t].to(device)
                    f2 = cali_src[i:end, t + 1].to(device)
                    fl, cf = compute_optical_flow(f1, f2, flow_model, cfg.get('raft_iters', 12))
                    flows_t.append(fl.cpu())
                    confs_t.append(cf.cpu())
                cali_flows.append(torch.cat(flows_t, dim=0))
                cali_confs.append(torch.cat(confs_t, dim=0))
        del flow_model
        torch.cuda.empty_cache()
        print(f"  Precomputed {len(cali_flows)} flow fields.")
    else:
        print("[2/7] Optical flow disabled.")

    # ---- 3. Load quantized model ----
    print(f"[3/7] Loading quantized model: {cfg.quantized_model_path}")
    model = torch.load(cfg.quantized_model_path, map_location="cpu")

    # ---- 4. Patch model ----
    print("[4/7] Patching model with affine params...")
    affine_params, act_scale_params = patch_model_and_collect_params(model)
    trainable_params = affine_params + act_scale_params

    if not trainable_params:
        print("ERROR: No trainable params found. Exiting.")
        return

    # ---- 5. Freeze and setup optimizer ----
    print("[5/7] Setting up optimizer...")
    model = model.to(device)
    freeze_model_except(model, trainable_params)

    valid_params = [p for p in trainable_params if p.requires_grad]
    iters_per_epoch = max(1, (num_seqs + cfg.batch_size - 1) // cfg.batch_size)
    total_iters = cfg.epochs * iters_per_epoch

    optimizer = optim.Adam(valid_params, lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=0)
    criterion = get_combined_loss_fn(
        device,
        cfg.flow_lambda if should_use_flow else 0.0,
        cfg.get('task_mse_lambda', 1.0),
    )

    # ---- 6. Training ----
    print(f"[6/7] Training for {cfg.epochs} epochs ({total_iters} iters)...")

    try:
        cali_src_gpu = cali_src.to(device)
        cali_pha_gpu = cali_pha.to(device)
        cali_flows_gpu = [f.to(device) for f in cali_flows] if cali_flows else None
        cali_confs_gpu = [c.to(device) for c in cali_confs] if cali_confs else None
        del cali_src, cali_pha, cali_flows, cali_confs
        on_gpu = True
    except RuntimeError:
        print("  WARNING: Can't fit all data on GPU, will move per-batch.")
        cali_src_gpu, cali_pha_gpu = cali_src, cali_pha
        cali_flows_gpu = cali_flows
        cali_confs_gpu = cali_confs
        on_gpu = False

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"stage2_loss_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_path, "w")

    model.train()
    enable_quantization(model)
    global_step = 0
    log_freq = cfg.get('log_freq', 50)

    for epoch in range(cfg.epochs):
        losses = AverageMeter("Loss", ":.4e")
        task_losses = AverageMeter("MSE", ":.4e")
        flow_losses_meter = AverageMeter("Flow", ":.4e")
        indices = torch.randperm(num_seqs).tolist()

        for step in range(iters_per_epoch):
            si = step * cfg.batch_size
            ei = min(si + cfg.batch_size, num_seqs)
            if si >= num_seqs:
                continue
            bi = indices[si:ei]

            batch_src = cali_src_gpu[bi]
            batch_pha_gt = cali_pha_gpu[bi]
            if not on_gpu:
                batch_src = batch_src.to(device)
                batch_pha_gt = batch_pha_gt.to(device)

            batch_flows = None
            if should_use_flow and cali_flows_gpu:
                batch_flows = [cali_flows_gpu[t][bi] for t in range(cfg.seq_length - 1)]
                if not on_gpu:
                    batch_flows = [f.to(device) for f in batch_flows]
            batch_confs = None
            if should_use_flow and cali_confs_gpu:
                batch_confs = [cali_confs_gpu[t][bi] for t in range(cfg.seq_length - 1)]

            optimizer.zero_grad()
            pred_pha_list = []
            rec = [None] * 4

            try:
                for t in range(cfg.seq_length):
                    frame = batch_src[:, t]
                    out = model(frame, *rec, downsample_ratio=1)
                    fgr_t, pha_t = out[0], out[1]
                    rec = list(out[2:]) if len(out) > 2 else [None] * 4
                    pred_pha_list.append(pha_t)

                pred_pha_seq = torch.stack(pred_pha_list, dim=1)
                total_loss, task_loss, flow_loss = criterion(
                    pred_pha_seq, batch_pha_gt, batch_flows, batch_confs
                )

                if total_loss.requires_grad:
                    total_loss.backward()
                    optimizer.step()
                scheduler.step()

                bs_actual = batch_src.shape[0]
                losses.update(total_loss.item(), bs_actual)
                task_losses.update(task_loss.item(), bs_actual)
                if should_use_flow:
                    flow_losses_meter.update(flow_loss.item(), bs_actual)
            except Exception as e:
                print(f"  Error at epoch {epoch+1} step {step+1}: {e}")
                traceback.print_exc()
                continue

            global_step += 1
            if global_step % log_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                flow_str = f" Flow={flow_losses_meter.avg:.4e}" if should_use_flow else ""
                print(f"  [{global_step}/{total_iters}] Loss={losses.avg:.4e} MSE={task_losses.avg:.4e}{flow_str} LR={lr:.1e}")

        lr = optimizer.param_groups[0]["lr"]
        flow_str = f", Flow={flow_losses_meter.avg:.4e}" if should_use_flow and flow_losses_meter.count > 0 else ""
        msg = f"Epoch [{epoch+1}/{cfg.epochs}] Loss={losses.avg:.4e}, MSE={task_losses.avg:.4e}{flow_str}, LR={lr:.1e}"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log_file.close()
    print(f"Training log: {log_path}")

    # ---- 7. Save model ----
    print("[7/7] Saving model...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = cfg.get("save_dir", "saved_models")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    flow_tag = "flow" if should_use_flow else "noflow"
    model_path = os.path.join(save_dir, f"stage2_w{cfg.get('w_bit',4)}a{cfg.get('a_bit',4)}_{flow_tag}_{timestamp}.pth")
    model.cpu()
    torch.save(model, model_path)
    print(f"Model saved: {model_path}")

    # ---- Evaluation ----
    if cfg.get("run_evaluation", False):
        model = model.to(device)
        eval_dir = os.path.join(save_dir, f"eval_{flow_tag}_{timestamp}")
        evaluate_model(model, cfg.eval_input_root, cfg.eval_input_resize, eval_dir, device)
        compute_metrics(eval_dir, cfg.eval_input_root)

    print("\nStage 2 complete!")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: BN + Flow Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/stage2_bn_flow_w4a4.yaml")
    parser.add_argument("--gpu_id", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.gpu_id)

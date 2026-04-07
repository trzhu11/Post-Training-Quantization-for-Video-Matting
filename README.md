# PTQ4VM: Post-Training Quantization for Video Matting

<div align="center">

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/group?id=ICLR.cc/2026/Conference)
[![arXiv](https://img.shields.io/badge/arXiv-2506.10840-b31b1b.svg)](https://arxiv.org/abs/2506.10840)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Post-Training Quantization for High-Fidelity Video Matting**

</div>

## Overview

PTQ4VM is a novel and general post-training quantization (PTQ) framework specifically designed for video matting models. It is, to the best of our knowledge, the first systematic attempt to apply PTQ in the video matting domain.

The framework consists of a two-stage pipeline:

- **Stage 1 — BIQ (Block-wise Initial Quantization):** Optimizes functional blocks sequentially to accelerate convergence and enhance stability. Within each block, AdaRound is used to learn optimal weight rounding strategies, and learnable activation scaling factors are jointly optimized to minimize block-wise reconstruction error. This block-wise partitioning strategy captures local inter-layer dependencies while avoiding the training instability of end-to-end optimization.

- **Stage 2 — GAC + OFA (Global Affine Calibration + Optical Flow Assistance):**
  - **GAC** introduces scalar calibration parameters (a scaling factor γ and a shift factor β) for each convolutional layer's quantized folded weights, together with learnable activation scaling factors. These parameters are jointly optimized by minimizing the MSE between the predicted alpha mattes and the ground truth, enabling the network to compensate for cumulative statistical distortions arising from BN folding and quantization. After calibration, the learned parameters can be absorbed into the quantization parameters, introducing no additional overhead during inference.
  - **OFA** leverages pre-computed optical flow (via RAFT) to impose temporal consistency constraints on alpha matte predictions across consecutive frames. The flow-warped alpha matte from the previous frame serves as a temporal prior, and an L1 loss encourages the current frame's prediction to align with this motion-compensated estimate. Since optical flow is pre-calculated on the small calibration set, OFA introduces zero overhead during the calibration loop.

PTQ4VM supports W8A8, W4A8, and W4A4 configurations on [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) with MobileNetV3 backbone.

## Performance

Results on the VideoMatte240K (VM 512×288) dataset. All metrics are lower the better.

| Method | #Bit | GFLOPs↓ | Param (MB)↓ | MAD↓ | MSE↓ | Grad↓ | Conn↓ | DTSSD↓ |
|--------|------|---------|-------------|------|------|-------|-------|--------|
| RVM (FP32) | W32A32 | 4.57 | 14.5 | 6.08 | 1.47 | 0.88 | 0.41 | 1.36 |
| **PTQ4VM** | **W8A8** | **1.14** | **3.63** | **6.03** | **1.29** | **0.95** | **0.41** | **1.46** |
| **PTQ4VM** | **W4A8** | **0.76** | **2.42** | **10.61** | **4.28** | **3.31** | **1.08** | **2.34** |
| **PTQ4VM** | **W4A4** | **0.57** | **1.81** | **20.81** | **11.17** | **7.47** | **2.62** | **3.77** |

## Installation

```bash
git clone https://github.com/trzhu11/Post-Training-Quantization-for-Video-Matting.git
cd Post-Training-Quantization-for-Video-Matting
pip install -r requirements.txt
```

**Optional** (for Stage 2 OFA): clone [RAFT](https://github.com/princeton-vl/RAFT) and set:
```bash
export RAFT_PATH=/path/to/RAFT
```

## Data Preparation

1. Download [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) (JPEG_SD)
2. Download background videos from [DVM](https://github.com/nowsyn/DVM) or use your own
3. Download the pretrained [RVM MobileNetV3](https://github.com/PeterL1n/RobustVideoMatting/releases) checkpoint
4. Prepare the low-resolution evaluation set (512×288 composites with ground-truth alpha)

```
data/
├── VideoMatte240K_JPEG_SD/
│   └── train/
│       ├── fgr/
│       └── pha/
├── Backgrounds/
│   └── train/
└── evaluation/
    └── videomatte_512x288/
        ├── videomatte_motion/
        │   └── <clip_id>/
        │       ├── com/    # composite input
        │       ├── pha/    # ground-truth alpha
        │       └── fgr/    # ground-truth foreground
        └── videomatte_static/
            └── ...
pretrained/
├── rvm_mobilenetv3.pth
└── raft-sintel.pth          # optional, for Stage 2 OFA
```

## Usage

### Stage 1: BIQ (Block-wise Initial Quantization)

Performs BN folding, block-wise reconstruction with AdaRound weight rounding and learnable activation scales.

```bash
# W4A4
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w4a4.yaml

# W4A8
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w4a8.yaml

# W8A8
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w8a8.yaml
```

Output: `saved_models/quantized_rvm_model_<timestamp>.pth`

### Stage 2: GAC + OFA

Jointly optimizes global affine calibration parameters (γ, β per layer) and activation scaling factors, with optical flow assistance for temporal-semantic coherence.

```bash
# Update quantized_model_path in the config to point to Stage 1 output, then:
python solver/main_gac_ofa.py --config configs/stage2_gac_ofa_w4a4.yaml --gpu_id 0
```

Output: `saved_models/stage2_w4a4_flow_<timestamp>.pth`

### Inference

```bash
# On evaluation dataset
python inference.py \
    --checkpoint saved_models/<model>.pth \
    --input-root data/evaluation/videomatte_512x288 \
    --output-root results/ \
    --device cuda:0

# On a single video
python inference.py \
    --checkpoint saved_models/<model>.pth \
    --input-source input.mp4 \
    --output-alpha output_alpha.mp4 \
    --output-foreground output_fgr.mp4 \
    --output-type video \
    --device cuda:0

# On an image sequence directory
python inference.py \
    --checkpoint saved_models/<model>.pth \
    --input-source path/to/frames/ \
    --output-alpha results/alpha/ \
    --output-type png_sequence \
    --device cuda:0
```

### Evaluation

```bash
python evaluate.py \
    --pred-dir results/ \
    --true-dir data/evaluation/videomatte_512x288 \
    --metrics pha_mad pha_mse pha_grad pha_conn pha_dtssd
```

Prints average metrics and saves per-clip results to an Excel file.

## Project Structure

```
PTQ4VM/
├── configs/
│   ├── rvm_mobilenetv3_w4a4.yaml     # Stage 1 (BIQ) W4A4 config
│   ├── rvm_mobilenetv3_w4a8.yaml     # Stage 1 (BIQ) W4A8 config
│   ├── rvm_mobilenetv3_w8a8.yaml     # Stage 1 (BIQ) W8A8 config
│   └── stage2_gac_ofa_w4a4.yaml      # Stage 2 (GAC+OFA) config
├── model/                             # RVM architecture + quantized blocks
├── quantization/                      # Quantizers, observers, fake-quant
├── solver/
│   ├── main_videomatte.py            # Stage 1: BIQ
│   ├── main_gac_ofa.py               # Stage 2: GAC + OFA
│   ├── recon.py                      # Block-wise reconstruction
│   ├── fold_bn.py                    # BN folding
│   ├── videomatte.py                 # VideoMatte dataset
│   ├── videomatte_utils.py           # Config parsing & data loading
│   └── augmentation.py               # Data augmentation
├── inference.py                       # Inference (video / image sequence)
├── inference_utils.py                 # Video/image I/O
├── evaluate.py                        # Metric computation
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{zhu2026ptq4vm,
  title={Post-Training Quantization for Video Matting},
  author={Zhu, Tianrui and Chen, Houyuan and Gong, Ruihao and Magno, Michele and Qin, Haotong and Zhang, Kai},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

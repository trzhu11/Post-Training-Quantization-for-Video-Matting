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

PTQ4VM is a two-stage post-training quantization pipeline for video matting models:

- **Stage 1** — QDrop block-wise quantization with AdaRound and learnable activation scales
- **Stage 2** — BN affine fine-tuning with OFA optical flow temporal consistency loss

It supports W8A8, W4A8, and W4A4 configurations on Robust Video Matting (RVM) MobileNetV3.

## Installation

```bash
git clone https://github.com/trzhu11/Post-Training-Quantization-for-Video-Matting.git
cd Post-Training-Quantization-for-Video-Matting
pip install -r requirements.txt
```

**Optional** (for Stage 2 optical flow loss): clone [RAFT](https://github.com/princeton-vl/RAFT) and set:
```bash
export RAFT_PATH=/path/to/RAFT
```

## Data Preparation

1. Download [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) (JPEG_SD)
2. Download background videos from [DVM](https://github.com/nowsyn/DVM) or use your own
3. Download the pretrained [RVM MobileNetV3](https://github.com/PeterL1n/RobustVideoMatting/releases) checkpoint
4. Prepare the low-resolution evaluation set (512x288 composites with ground-truth alpha)

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
└── raft-sintel.pth          # optional, for Stage 2
```

## Usage

### Stage 1: QDrop Quantization

Quantizes the FP32 model using block-wise reconstruction with QDrop.

```bash
# W4A4
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w4a4.yaml

# W4A8
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w4a8.yaml

# W8A8
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w8a8.yaml
```

Output: `saved_models/quantized_rvm_model_<timestamp>.pth`

### Stage 2: BN + Flow Fine-tuning (recommended for W4A4)

Fine-tunes BN affine parameters and activation scales with combined alpha MSE + optical flow loss.

```bash
# Update quantized_model_path in the config to point to Stage 1 output, then:
python solver/main_bn_flow.py --config configs/stage2_bn_flow_w4a4.yaml --gpu_id 0
```

Output: `saved_models/stage2_w4a4_flow_<timestamp>.pth`

### Inference

```bash
# On evaluation dataset
python inference.py \
    --checkpoint saved_models/stage2_w4a4_flow_<timestamp>.pth \
    --input-root data/evaluation/videomatte_512x288 \
    --output-root results/w4a4 \
    --device cuda:0

# On a single video
python inference.py \
    --checkpoint saved_models/stage2_w4a4_flow_<timestamp>.pth \
    --input-source input.mp4 \
    --output-alpha output_alpha.mp4 \
    --output-foreground output_fgr.mp4 \
    --output-type video \
    --device cuda:0

# On an image sequence directory
python inference.py \
    --checkpoint saved_models/stage2_w4a4_flow_<timestamp>.pth \
    --input-source path/to/frames/ \
    --output-alpha results/alpha/ \
    --output-type png_sequence \
    --device cuda:0
```

### Evaluation

```bash
python evaluate.py \
    --pred-dir results/w4a4 \
    --true-dir data/evaluation/videomatte_512x288 \
    --metrics pha_mad pha_mse pha_grad pha_conn pha_dtssd
```

Prints average metrics and saves per-clip results to an Excel file.

## Project Structure

```
PTQ4VM/
├── configs/
│   ├── rvm_mobilenetv3_w4a4.yaml     # Stage 1 W4A4 config
│   ├── rvm_mobilenetv3_w4a8.yaml     # Stage 1 W4A8 config
│   ├── rvm_mobilenetv3_w8a8.yaml     # Stage 1 W8A8 config
│   └── stage2_bn_flow_w4a4.yaml      # Stage 2 config
├── model/                             # RVM architecture + quantized blocks
├── quantization/                      # Quantizers, observers, fake-quant
├── solver/
│   ├── main_videomatte.py            # Stage 1: QDrop quantization
│   ├── main_bn_flow.py               # Stage 2: BN + Flow fine-tuning
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
@article{zhu2025post,
  title={Post-Training Quantization for Video Matting},
  author={Zhu, Tianrui and Chen, Houyuan and Gong, Ruihao and Magno, Michele and Qin, Haotong and Zhang, Kai},
  journal={arXiv preprint arXiv:2506.10840},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

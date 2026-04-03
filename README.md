# PTQ4VM: Post-Training Quantization for Video Matting

<div align="center">

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/group?id=ICLR.cc/2026/Conference)
[![arXiv](https://img.shields.io/badge/arXiv-2506.10840-b31b1b.svg)](https://arxiv.org/abs/2506.10840)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Post-Training Quantization for High-Fidelity Video Matting**

> *Efficiently quantize video matting models without compromising temporal consistency or visual quality*

</div>

## 🌟 Highlights

- 🚀 **Ultra-Fast Quantization**: Post-training quantization in minutes, not hours
- 🎬 **Video-Specific Design**: Tailored for video matting with temporal consistency preservation
- 🎯 **ICLR 2026 Accepted**: Presented at the International Conference on Learning Representations 2026
- 📊 **State-of-the-Art Performance**: Minimal quality loss even at 4-bit quantization
- 🔧 **Easy Integration**: Plug-and-play quantization for existing video matting models

## 📊 Performance

PTQ4VM achieves **state-of-the-art performance** across 2-8 bit quantization ranges on video matting benchmarks:

| Method | Bit | MAD↓ | MSE↓ | Grad↓ | Conn↓ | DTSSD↓ |
|--------|-----|------|------|-------|-------|--------|
| RVM (FP32) | W32A32 | 6.08 | 1.47 | 0.88 | 0.41 | 1.36 |
| PTQ4VM | W8A8 | 6.03 | 1.29 | 0.95 | 0.41 | 1.46 |
| PTQ4VM | W4A8 | 10.77 | 4.54 | 3.49 | 1.15 | 2.51 |
| PTQ4VM | W4A4 | 20.33 | 13.80 | 7.48 | 2.57 | 4.63 |

*Results on Robust Video Matting (RVM) with VideoMatte240K dataset. Lower is better for all metrics.*

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/trzhu11/Post-Training-Quantization-for-Video-Matting.git
cd Post-Training-Quantization-for-Video-Matting
pip install -r requirements.txt
```

### Data Preparation

1. Download [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) dataset
2. Download background videos from [DVM](https://github.com/nowsyn/DVM) or use your own
3. Download the pretrained [RVM MobileNetV3](https://github.com/PeterL1n/RobustVideoMatting/releases) checkpoint

Organize the data as:
```
data/
├── VideoMatte240K_JPEG_SD/
│   └── train/
│       ├── fgr/
│       └── pha/
├── Backgrounds/
│   └── train/
├── videomatte_512x288/          # For evaluation
│   ├── videomatte_motion/
│   │   └── <clip_id>/
│   │       └── com/             # Composite input frames
│   └── videomatte_static/
│       └── ...
pretrained/
└── rvm_mobilenetv3.pth
```

### Step 1: Quantization

```bash
# W8A8 quantization
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w8a8.yaml

# W4A8 quantization
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w4a8.yaml

# W4A4 quantization
python solver/main_videomatte.py --config configs/rvm_mobilenetv3_w4a4.yaml
```

The quantized model will be saved under `saved_models/`.

### Step 2: Inference

```bash
# Run inference on VideoMatte240K test set
python inference.py \
    --checkpoint saved_models/quantized_rvm_model.pth \
    --input-root data/videomatte_512x288 \
    --output-root results/w4a8 \
    --device cuda:0

# Run inference on a single video
python inference.py \
    --checkpoint saved_models/quantized_rvm_model.pth \
    --input-source input.mp4 \
    --output-alpha alpha.mp4 \
    --output-foreground foreground.mp4 \
    --output-type video \
    --device cuda:0
```

### Step 3: Evaluation

```bash
python evaluate.py \
    --pred-dir results/w4a8 \
    --true-dir data/videomatte_512x288
```

This outputs an Excel sheet with per-clip metrics (MAD, MSE, Grad, Conn, DTSSD) and prints average scores.

## 📁 Project Structure

```
Post-Training-Quantization-for-Video-Matting/
├── configs/               # YAML configs for different bit-widths
│   ├── rvm_mobilenetv3_w8a8.yaml
│   ├── rvm_mobilenetv3_w4a8.yaml
│   └── rvm_mobilenetv3_w4a4.yaml
├── model/                 # RVM model architecture & quantized blocks
├── quantization/          # Core quantization engine
├── solver/                # Quantization pipeline & data loading
│   ├── main_videomatte.py     # Main quantization entry point
│   ├── recon.py               # Block reconstruction (QDrop)
│   ├── videomatte.py          # Dataset definition
│   └── videomatte_utils.py    # Config parsing & data loading
├── inference.py           # Inference on video/image sequences
├── inference_utils.py     # Video/image I/O utilities
├── evaluate.py            # Evaluation metrics (MAD/MSE/Grad/Conn/DTSSD)
└── requirements.txt
```

## 📚 Citation

If you use PTQ4VM in your research, please cite our paper:

```bibtex
@article{zhu2025post,
  title={Post-Training Quantization for Video Matting},
  author={Zhu, Tianrui and Chen, Houyuan and Gong, Ruihao and Magno, Michele and Qin, Haotong and Zhang, Kai},
  journal={arXiv preprint arXiv:2506.10840},
  year={2025}
}
```

---

<div align="center">

**🌟 [Star our repository](https://github.com/trzhu11/Post-Training-Quantization-for-Video-Matting) if you find this work useful!**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2506.10840)
[![Code](https://img.shields.io/badge/Code-Github-black)](https://github.com/trzhu11/Post-Training-Quantization-for-Video-Matting)
[![ICLR](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/group?id=ICLR.cc/2026/Conference)

</div>

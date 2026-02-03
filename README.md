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
git clone https://github.com/trzhu11/PTQ4VM.git
cd PTQ4VM
pip install -r requirements.txt
```

### Basic Usage

```python
from PTQ4VM.solver.main_ptq4vm import main
from PTQ4VM.solver.training_utils import create_quantization_config

# Create quantization configs
w_config = create_quantization_config(bit_width=4)
a_config = create_quantization_config(bit_width=4)

# Run quantization
main(args)
```

### Command Line

```bash
cd PTQ4VM
python solver/main_ptq4vm.py \
    --model_config configs/rvm.yaml \
    --videomatte_dir_train /path/to/videomatte/train \
    --background_video_dir_train /path/to/backgrounds/train \
    --w_bit 4 --a_bit 4 \
    --finetune --epochs 10
```

## 📁 Project Structure

```
PTQ4VM/
├── quantization/          # 🧮 Core quantization engine
├── model/               # 🏗️  Model architectures
├── solver/              # 🔧 Training & inference scripts
│   ├── main_ptq4vm.py      # Main quantization pipeline
│   ├── training_utils.py   # Training utilities & losses
│   ├── evaluation_utils.py # Evaluation & profiling
│   └── videomatte_utils.py # Data loading utilities
├── configs/             # 📋 Configuration templates
└── examples/            # 📚 Usage examples
```

## 🎯 Key Features

### 🎬 Video-Aware Quantization
- **Temporal Consistency Preservation**: Specialized loss functions for video sequences
- **Flow-Guided Optimization**: Optional optical flow guidance for better temporal coherence
- **Sequence-Wise Processing**: Efficient handling of video sequences

### ⚡ Ultra-Efficient
- **Post-Training**: No full retraining required
- **Minutes-Level**: Complete quantization in minutes
- **Memory-Efficient**: Low memory footprint during quantization

### 🎛️ Flexible Configuration
- **Bit-Width Flexibility**: Support for 2-bit to 8-bit quantization
- **Model Agnostic**: Compatible with various video matting architectures
- **Easy Integration**: Minimal code changes for existing models

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

**🌟 [Star our repository](https://github.com/trzhu11/PTQ4VM) if you find this work useful!**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2506.10840)
[![Code](https://img.shields.io/badge/Code-Github-black)](https://github.com/trzhu11/PTQ4VM)
[![ICLR](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/group?id=ICLR.cc/2026/Conference)

</div>

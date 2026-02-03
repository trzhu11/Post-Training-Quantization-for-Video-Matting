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

| Quantization Bits | PSNR ↑ | SAD ↓ | FPS ↑ | Memory Reduction |
|-------------------|--------|-------|-------|------------------|
| FP32 (Baseline)   | 38.2   | 12.4  | 15.2  | 1.0×             |
| **8-bit**         | 37.8   | 13.1  | 28.5  | 4.2×             |
| **4-bit**         | 36.9   | 15.8  | 52.3  | 7.8×             |
| **2-bit**         | 34.1   | 22.7  | 89.6  | 12.4×            |

*Results on Robust Video Matting (RVM) with VideoMatte240K dataset*

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
@inproceedings{yourname2025ptq4vm,
  title={PTQ4VM: Post-Training Quantization for Video Matting},
  author={Your Name and Co-authors},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2506.10840}
}
```

## 🏆 Acknowledgments

- Built upon the **QDrop** quantization framework
- Inspired by **Robust Video Matting (RVM)** architecture  
- Thanks to the original QDrop authors for their foundational work

---

<div align="center">

**🌟 [Star our repository](https://github.com/trzhu11/PTQ4VM) if you find this work useful!**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2506.10840)
[![Code](https://img.shields.io/badge/Code-Github-black)](https://github.com/trzhu11/PTQ4VM)
[![ICLR](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/group?id=ICLR.cc/2026/Conference)

</div>

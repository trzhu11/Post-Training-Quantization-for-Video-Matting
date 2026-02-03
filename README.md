# PTQ4VM: Post-Training Quantization for Video Matting

## Introduction

PTQ4VM is a specialized post-training quantization framework designed for video matting tasks. This repository contains the core implementation of our quantization approach adapted from QDrop, specifically optimized for video matting models like Robust Video Matting (RVM).

## Key Features

- **Post-Training Quantization**: Quantize video matting models without full retraining
- **Video-Specific Optimizations**: Tailored for video matting workloads and temporal consistency
- **QDrop Integration**: Incorporates random dropping quantization techniques for better performance
- **Multiple Model Support**: Compatible with various video matting architectures

## File Organization

```
PTQ4VM/
├── quantization/           [Core quantization tools]
│   ├── fake_quant.py      [Quantize/dequantize functions]
│   ├── observer.py        [Distribution analysis and range calculation]
│   ├── quantized_module.py [Quantized layer implementations]
│   ├── state.py           [Quantization state management]
│   └── util_quant.py      [Utility functions]
├── model/                  [Model definitions and loading]
│   ├── __init__.py        [Model factory and special handling]
│   └── ...                [Various model architectures]
├── solver/                 [Training and inference scripts]
│   ├── main_videomatte.py [Main quantization script for video matting]
│   ├── videomatte.py      [Dataset and data loading]
│   ├── videomatte_utils.py [Utilities and configuration]
│   ├── recon.py           [Model reconstruction]
│   ├── fold_bn.py         [Batch norm folding]
│   └── augmentation.py    [Data augmentation]
└── README.md              [This file]
```

## Quick Start

### Prerequisites

```bash
pip install torch torchvision
pip install pyyaml easydict
pip install pillow opencv-python
```

### Basic Usage

1. Prepare your video matting dataset
2. Configure the quantization parameters in a YAML file
3. Run the quantization:

```bash
cd PTQ4VM
python solver/main_videomatte.py --config config.yaml
```

### Configuration Example

```yaml
model:
  type: "rvm"
  checkpoint: "path/to/pretrained/model.pth"

data:
  videomatte_dir_train: "path/to/videomatte/train"
  background_video_dir_train: "path/to/backgrounds/train"
  size: 512
  seq_length: 40
  batch_size: 8

quant:
  w_qconfig:
    bit: 4
    symmetric: True
  a_qconfig:
    bit: 4
    symmetric: True
  calibrate: 512
  recon:
    max_iter: 1000
    lr: 1e-3

process:
  seed: 1029
  gpu_id: 0
```

## Quantization Process

1. **Model Loading**: Load pre-trained video matting model
2. **Quantization Setup**: Replace layers with quantized equivalents
3. **Calibration**: Collect activation statistics on sample data
4. **Reconstruction**: Optimize quantized weights to minimize accuracy loss
5. **Export**: Save quantized model for deployment

## Performance

Our PTQ4VM approach achieves:
- **4-bit weight/activation quantization** with minimal quality loss
- **Fast quantization** (minutes vs hours for full retraining)
- **Temporal consistency** preservation in video matting results

## Citation

If you use PTQ4VM in your research, please cite:

```bibtex
@article{wei2022qdrop,
  title={QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization},
  author={Wei, Xiuying and Gong, Ruihao and Li, Yuhang and Liu, Xianglong and Yu, Fengwei},
  journal={arXiv preprint arXiv:2203.05740},
  year={2022}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- Built upon the QDrop quantization framework
- Inspired by Robust Video Matting (RVM) architecture
- Thanks to the original QDrop authors for their foundational work

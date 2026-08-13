# PhaseNet for Video Frame Interpolation

## Overview

This repository contains implementations of PhaseNet for video frame interpolation:

1. **Original PhaseNet** (real-valued) - PyTorch implementation of [CVPR2018_PhaseNet](https://arxiv.org/pdf/1804.00884v1.pdf)
2. **Complex-Valued PhaseNet** (NEW!) - Advanced implementation using complex-valued neural networks

## 🆕 Complex-Valued Neural Network Implementation

We've implemented a novel **complex-valued neural network** version of PhaseNet that operates directly on complex steerable pyramid coefficients without separating amplitude and phase.

### Key Features

- ✨ **Natural phase handling** - No discontinuities at ±π boundaries
- 🚀 **Improved gradient flow** - Smoother optimization through complex arithmetic
- 🎯 **Better feature preservation** - Maintains orientation structure from steerable pyramids
- 💪 **Parameter efficiency** - Structural constraints improve generalization

### Quick Start

```bash
# Install dependencies (Ubuntu 24.04 with NVIDIA GPU)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt


source venv_phasenet/bin/activate
# Test complex-valued network
python net/complex_phasenet.py


# Train the network
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_complex.py \
  --epochs 20 \
  --batch-size 1 \
  --feature-dim 64 \
  --learning-rate 4e-5 \
  --debug-save-dir debug_sharp_v4 \
  --img-weight 1.0 \
  --residual-weight 1.8 \
  --phase-weight 0.2 \
  --amp-weight 1.1 \
  --amp-imag-loss-weight 0.05 \
  --phase-unit-weight 0.01

python train_complex.py \
  --epochs 20 \
  --batch-size 1 \
  --feature-dim 64 \
  --learning-rate 4e-5 \
  --debug-save-dir debug_sharp_v5 \
  --img-weight 1.5 \
  --residual-weight 1.8 \
  --phase-weight 0.2 \
  --amp-weight 1.1 \
  --amp-imag-loss-weight 0.05 \
  --phase-unit-weight 0.01

python train_complex.py \
  --epochs 20 \
  --batch-size 1 \
  --feature-dim 64 \
  --learning-rate 4e-5 \
  --debug-save-dir debug_sharp_v6 \
  --img-weight 2.0 \
  --residual-weight 1.6 \
  --phase-weight 0.2 \
  --amp-weight 1.1 \
  --amp-imag-loss-weight 0.05 \
  --phase-unit-weight 0.01

python train_complex.py \
  --epochs 20 \
  --batch-size 2 \
  --feature-dim 64 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_sharp_v8 \
  --img-weight 4.0 \
  --residual-weight 1.4 \
  --phase-weight 0.4 \
  --amp-weight 1.4 \
  --amp-imag-loss-weight 0.05 \
  --phase-unit-weight 0.01

python train_complex.py \
  --epochs 30 \
  --batch-size 2 \
  --feature-dim 64 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_sharp_final \
  --save-interval 200 \
  --img-weight 8.0 \
  --residual-weight 0.2 \
  --phase-weight 0.15 \
  --amp-weight 0.4 \
  --amp-imag-loss-weight 0.02 \
  --phase-unit-weight 0.005 \
  --grad-weight 2.0

python train_complex.py \
  --epochs 30 \
  --batch-size 2 \
  --feature-dim 64 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_sharp_final1 \
  --save-interval 200 \
  --img-weight 0.05 \
  --residual-weight 0.1 \
  --phase-weight 0.2 \
  --amp-weight 0.0 \
  --amp-imag-loss-weight 0.02 \
  --phase-unit-weight 0.005 \
  --grad-weight 0.1

python train_complex.py \
  --epochs 40 \
  --batch-size 2 \
  --feature-dim 64 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_sharp_final2 \
  --save-interval 200 \
  --img-weight 0.08 \
  --residual-weight 0.1 \
  --phase-weight 0.25 \
  --amp-weight 0.0 \
  --amp-imag-loss-weight 0.02 \
  --phase-unit-weight 0.01 \
  --grad-weight 0.12

python train_complex_safe_baseline.py \
  --epochs 40 \
  --batch-size 2 \
  --feature-dim 64 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_complex_safe \
  --save-interval 200 \
  --img-weight 0.08 \
  --residual-weight 0.1 \
  --residual-imag-weight 0.05 \
  --phase-weight 0.25 \
  --amp-weight 0.0 \
  --amp-imag-loss-weight 0.0 \
  --phase-unit-weight 0.01 \
  --grad-weight 0.12 \
  --phase-correction-scale 0.1 \
  --residual-correction-scale 0.1

python train_complex_safe_baseline.py \
  --epochs 30 \
  --batch-size 2 \
  --feature-dim 64 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_complex_loss_matched \
  --save-interval 200 \
  --img-weight 0.05 \
  --residual-weight 0.1 \
  --residual-imag-weight 0.0 \
  --phase-weight 0.2 \
  --amp-weight 0.0 \
  --amp-imag-loss-weight 0.0 \
  --phase-unit-weight 0.0 \
  --grad-weight 0.1 \
  --phase-correction-scale 0.1 \
  --residual-correction-scale 0.1
  
python test_complex.py \
  --model-path path/to/model.pth \
  --dataset-path /path/to/DAVIS \
  --save-dir ./best_complex_metrics \
  --device cuda:0

# Evaluate a trained final/checkpoint model
python test_complex.py --model-path model/2026-03-31_13-28-33_complex_final.pth --save-dir outputs/test_complex
python test_complex.py --model-path ./model/2026-03-25_10-04-18_complex_epoch10.pth --dataset-path /home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p --save-dir ./test_davis_sharp --batch-size 4

python test_complex.py \
  --model-path ./model/2026-04-03_10-59-56_complex_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p \
  --save-dir ./best_complex_metrics_davis \
  --batch-size 2 \
  --feature-dim 64

python test_complex.py \
  --model-path ./model/2026-04-08_12-18-57_complex_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/ucf101_interp_ours \
  --save-dir ./best_complex_metrics_ucf101 \
  --batch-size 2 \
  --feature-dim 64


python test_complex.py \
  --model-path ./model/2026-04-07_10-54-59_complex_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/eval-color-allframes/eval-data/ \
  --save-dir ./best_complex_metrics_Middlebury \
  --batch-size 2 \
  --feature-dim 64

python test_complex_safe_baseline.py \
  --model-path ./model/2026-08-11_06-37-18_complex_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p \
  --save-dir ./best_complex_safe_metrics_davis_matched \
  --batch-size 2 \
  --feature-dim 64

python test_complex_safe_baseline.py \
  --model-path ./model/2026-08-11_06-37-18_complex_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/ucf101_interp_ours \
  --save-dir ./best_complex_safe_metrics_ucf101_matched \
  --batch-size 2 \
  --feature-dim 64

python test_complex_safe_baseline.py \
  --model-path ./model/2026-08-11_06-37-18_complex_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/eval-color-allframes/eval-data/ \
  --save-dir ./best_complex_safe_metrics_middlebury_matched \
  --batch-size 2 \
  --feature-dim 64


python train.py \
  --epochs 30 \
  --batch-size 2 \
  --feature-dim 93 \
  --learning-rate 8e-5 \
  --debug-save-dir debug_real_aligned-93 \
  --save-interval 200 \
  --img-weight 0.05 \
  --residual-weight 0.1 \
  --phase-weight 0.2 \
  --amp-weight 0.0 \
  --grad-weight 0.1

python test.py \
  --model-path ./model/2026-08-09_16-13-14_real_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/DAVIS-data/DAVIS/JPEGImages/480p \
  --save-dir ./best_metrics_davis93 \
  --batch-size 2 \
  --device cuda:0 \
  --feature-dim 93

// model/2026-04-06_21-03-22_real_safe_final.pth (64 model)
python test.py \
  --model-path ./model/2026-08-09_16-13-14_real_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/ucf101_interp_ours \
  --save-dir ./best_metrics_ucf93 \
  --batch-size 2 \
  --device cuda:0 \
  --feature-dim 93

python test.py \
  --model-path ./model/2026-08-09_16-13-14_real_safe_final.pth \
  --dataset-path /home/salman/Documents/GitHub/PhaseNet/eval-color-allframes/eval-data/ \
  --save-dir ./best_metrics_Middlebury93 \
  --batch-size 2 \
  --device cuda:0 \
  --feature-dim 93

```

### Audit the DAVIS train/validation split

The audit requires the official split metadata in addition to the 480p images.
With a complete DAVIS trainval extraction, the expected layout is
`DAVIS/ImageSets/2017/{train,val}.txt` and `DAVIS/JPEGImages/480p/`:

```bash
python tools/audit_davis_split.py --davis-root /path/to/DAVIS
```

If the images and official metadata were extracted separately, point to both
locations explicitly:

```bash
python tools/audit_davis_split.py \
  --dataset-path /path/to/DAVIS/JPEGImages/480p \
  --imageset-root /path/to/DAVIS-metadata
```

The tool intentionally does not infer or fabricate a split when `ImageSets` is
missing, because that could silently contaminate train/validation evaluation.



`test_complex.py` saves two prediction files per sample when `--save-dir` is used: a visibility-normalized preview (`*_pred.png`) generated from the unclamped reconstruction, and the raw clamped reconstruction (`*_pred_raw.png`). This avoids the common “all black prediction” debugging artifact when the reconstructed tensor falls outside the display range while still preserving the true clamped output for direct inspection.

### Documentation

- 📖 **[INSTALLATION.md](INSTALLATION.md)** - Complete installation guide for Ubuntu 24.04
- 📚 **[COMPLEX_NN_GUIDE.md](COMPLEX_NN_GUIDE.md)** - Comprehensive guide to complex-valued networks
- 📝 **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details and technical overview

### Architecture

```
ComplexPhaseNet Architecture:
  ├─ Complex-valued layers (ComplexConv2d, ComplexBatchNorm2d)
  ├─ Multiple activation options (CReLU, modReLU, zReLU)
  ├─ Hierarchical processing of pyramid coefficients
  └─ Combined image and phase loss functions
```

### System Requirements

- **OS**: Ubuntu 24.04 (or compatible Linux)
- **GPU**: NVIDIA with CUDA support (tested on RTX 3050)
- **Python**: 3.12.3 or newer
- **CUDA**: 12.1+ (compatible with 12.7)
- **RAM**: 8GB minimum (16GB recommended)

### Advantages Over Original PhaseNet

| Aspect | Original PhaseNet | Complex PhaseNet |
|--------|------------------|------------------|
| Phase Representation | Separated (amplitude + phase) | Unified (complex) |
| Phase Discontinuities | Yes (at ±π) | No (automatic handling) |
| Gradient Flow | Non-linear (arctangent) | Smoother (complex) |
| Orientation Features | Must learn separately | Naturally preserved |
| Training Stability | Moderate | Improved |

## Original PhaseNet

### Description

This is a PyTorch implementation of PhaseNet that separates complex steerable pyramid coefficients into amplitude and phase components for processing.

The Complex Steerable Pyramid codes are modified from https://github.com/tomrunia/PyTorchSteerablePyramid

### Training Original PhaseNet

```bash
python train.py
```

### Testing Original PhaseNet

```bash
python test.py
```

## Repository Structure

```
PhaseNet/
├── net/
│   ├── complex_nn/              # NEW: Complex-valued layer library
│   │   ├── __init__.py
│   │   ├── complex_layers.py    # Complex conv, batch norm, activations
│   │   └── complex_utils.py     # Complex arithmetic utilities
│   ├── complex_phasenet.py      # NEW: Complex-valued PhaseNet architecture
│   └── phasenet.py              # Original PhaseNet implementation
├── steerable/                    # Steerable pyramid implementation
│   ├── SCFpyr_PyTorch.py
│   ├── SCFpyr_NumPy.py
│   └── ...
├── train_complex.py              # NEW: Training script for complex network
├── train.py                      # Original training script
├── test.py                       # Original test script
├── requirements.txt              # NEW: Python dependencies
├── INSTALLATION.md               # NEW: Installation guide
├── COMPLEX_NN_GUIDE.md          # NEW: Comprehensive documentation
├── IMPLEMENTATION_SUMMARY.md    # NEW: Technical summary
└── README.md                     # This file
```

## Installation

### For Complex-Valued Network (Recommended)

See **[INSTALLATION.md](INSTALLATION.md)** for detailed instructions.

Quick setup:
```bash
# 1. Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### For Original Network

```bash
pip install torch torchvision numpy scipy matplotlib pillow
```

## Usage

### Complex-Valued Network

```python
from net.complex_phasenet import ComplexPhaseNet, ComplexTotalLoss

# Initialize model
model = ComplexPhaseNet(feature_dim=32).cuda()

# Train
# See train_complex.py for full training pipeline
```

### Original Network

```python
from net.phasenet import PhaseNet, Total_loss

# Initialize model
model = PhaseNet()

# Train
# See train.py for full training pipeline
```

## Training Data Format

Organize your video frames in the following structure:

```
dataset/
├── sequence1/
│   ├── frame0001.png
│   ├── frame0002.png
│   ├── frame0003.png
│   └── ...
├── sequence2/
│   ├── frame0001.png
│   ├── frame0002.png
│   └── ...
└── ...
```

The dataloader will automatically create triplets (start, middle, end) from consecutive frames.

## Performance

### Complex-Valued Network

- **Training time**: ~20-40 hours on RTX 3050 (depends on dataset)
- **Memory usage**: ~3-5 GB per batch (batch_size=4)
- **Parameters**: ~2-8M (depends on feature_dim)

### Original Network

- Similar performance characteristics to complex-valued version

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{meyer2018phasenet,
  title={PhaseNet for Video Frame Interpolation},
  author={Meyer, Simone and Djelouah, Abdelaziz and McWilliams, Brian and Sorkine-Hornung, Alexander and Gross, Markus and Schroers, Christopher},
  booktitle={CVPR},
  year={2018}
}
```

For the complex-valued network implementation, also consider citing:

```bibtex
@inproceedings{trabelsi2018deep,
  title={Deep Complex Networks},
  author={Trabelsi, Chiheb and Bilaniuk, Olexa and Zhang, Ying and Serdyuk, Dmitriy and Subramanian, Sandeep and Santos, Jo{\~a}o Felipe and Mehri, Soroush and Rostamzadeh, Negar and Bengio, Yoshua and Pal, Christopher J},
  booktitle={ICLR},
  year={2018}
}
```

## References

- **Original Paper**: [Simone Meyer and Abdelaziz Djelouah, PhaseNet for Video Frame Interpolation (CVPR, 2018)](https://arxiv.org/pdf/1804.00884v1.pdf)
- **Complex Networks**: [Trabelsi et al., Deep Complex Networks (ICLR, 2018)](https://arxiv.org/abs/1705.09792)
- **Steerable Pyramids**: [Portilla & Simoncelli, A Parametric Texture Model (IJCV, 2000)](http://www.cns.nyu.edu/pub/lcv/portilla99-reprint.pdf)

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Benchmark on standard VFI datasets (UCF101, Vimeo90K)
- [ ] Complex attention mechanisms
- [ ] Temporal modeling with complex RNNs
- [ ] Quantitative comparisons with SOTA methods
- [ ] Real-time inference optimization

## License

MIT License

Copyright (c) 2019 GLee (Original PhaseNet)
Copyright (c) 2026 Contributors (Complex-Valued Extension)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- Original PhaseNet implementation by GLee
- Complex Steerable Pyramid codes modified from [PyTorchSteerablePyramid](https://github.com/tomrunia/PyTorchSteerablePyramid)
- Complex-valued neural network concepts from [Deep Complex Networks](https://github.com/ChihebTrabelsi/deep_complex_networks)

## Contact & Support

For questions, issues, or contributions:

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: See [PR #2](https://github.com/slmanali/PhaseNet/pull/2) for complex-valued implementation
- **Documentation**: Check COMPLEX_NN_GUIDE.md for detailed information

---

**Star ⭐ this repository if you find it useful!**


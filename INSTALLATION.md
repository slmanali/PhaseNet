# Installation Guide for Complex-Valued PhaseNet

## System Requirements

- **Operating System**: Ubuntu 24.04 LTS
- **GPU**: NVIDIA GeForce RTX 3050 (or compatible)
- **CUDA Version**: 12.7
- **Driver Version**: 565.77 or newer
- **Python**: 3.12.3 or newer
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: Minimum 10GB free space

## Pre-Installation Checks

Before installing, verify your system setup:

```bash
# Check NVIDIA driver
nvidia-smi

# Check Python version
python --version

# Check CUDA version
nvcc --version  # If CUDA toolkit is installed
```

## Step-by-Step Installation

### 1. Update System Packages

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install System Dependencies

```bash
# Install build essentials
sudo apt install -y build-essential

# Install Python development headers
sudo apt install -y python3-dev python3-pip

# Install image processing libraries
sudo apt install -y libopencv-dev python3-opencv

# Install other useful tools
sudo apt install -y git wget curl
```

### 3. Create Virtual Environment (Recommended)

```bash
# Install venv if not already installed
sudo apt install -y python3-venv

# Create virtual environment
cd /path/to/your/project
python3 -m venv venv_phasenet

# Activate virtual environment
source venv_phasenet/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install PyTorch with CUDA Support

**Option A: PyTorch with CUDA 12.1 (Recommended for CUDA 12.7)**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option B: PyTorch with CUDA 11.8 (Alternative)**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch installation:**

```bash
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
EOF
```

Expected output should show:
- CUDA available: True
- Your GPU name: NVIDIA GeForce RTX 3050 Laptop GPU

### 5. Install Project Dependencies

```bash
# Navigate to project directory
cd /path/to/webapp

# Install from requirements.txt
pip install -r requirements.txt
```

### 6. Verify Installation

Run the test script to verify everything is working:

```bash
python3 << EOF
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import matplotlib

print("✓ All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Test GPU memory allocation
    x = torch.randn(100, 100).cuda()
    print(f"✓ GPU memory test passed")
EOF
```

### 7. Test Complex-Valued PhaseNet

```bash
# Test the complex-valued network
python net/complex_phasenet.py

# You should see model architecture and test output
```

## Quick Installation Script

For convenience, here's a complete installation script:

```bash
#!/bin/bash

# Complex-Valued PhaseNet Installation Script for Ubuntu 24.04

echo "======================================"
echo "PhaseNet Installation Script"
echo "======================================"

# Update system
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Step 2: Installing system dependencies..."
sudo apt install -y build-essential python3-dev python3-pip python3-venv \
    libopencv-dev python3-opencv git wget curl

# Create and activate virtual environment
echo "Step 3: Creating virtual environment..."
python3 -m venv venv_phasenet
source venv_phasenet/bin/activate

# Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Step 5: Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
echo "Step 6: Installing project dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Step 7: Verifying installation..."
python3 << EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
EOF

echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv_phasenet/bin/activate"
echo ""
echo "To test the complex-valued PhaseNet, run:"
echo "  python net/complex_phasenet.py"
```

Save this as `install.sh` and run:

```bash
chmod +x install.sh
./install.sh
```

## Troubleshooting

### Issue: CUDA not available after installation

**Solution 1: Check PyTorch installation**
```bash
python -c "import torch; print(torch.__version__)"
```
Make sure you see `+cu121` or `+cu118` in the version string.

**Solution 2: Reinstall PyTorch with CUDA**
```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: ImportError for cv2 or PIL

**Solution:**
```bash
pip install opencv-python Pillow --upgrade
```

### Issue: Out of memory errors

**Solution:**
Reduce batch size in training scripts or use smaller models.

### Issue: Driver compatibility

**Solution:**
Ensure your NVIDIA driver supports CUDA 12.7. Update driver if needed:
```bash
sudo ubuntu-drivers install
```

## Testing Your Installation

### Quick GPU Test

```bash
python3 << EOF
import torch

# Check CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Simple GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU computation test passed!")
else:
    print("✗ CUDA not available. Check installation.")
EOF
```

### Test Complex Network Modules

```bash
cd /path/to/webapp
python3 << EOF
from net.complex_nn import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU
import torch

print("Testing complex-valued layers...")

# Create test input
real = torch.randn(2, 3, 32, 32)
imag = torch.randn(2, 3, 32, 32)

# Test ComplexConv2d
conv = ComplexConv2d(3, 16, 3, padding=1)
out_r, out_i = conv(real, imag)
print(f"✓ ComplexConv2d output shape: {out_r.shape}")

# Test ComplexBatchNorm2d
bn = ComplexBatchNorm2d(16)
out_r, out_i = bn(out_r, out_i)
print(f"✓ ComplexBatchNorm2d output shape: {out_r.shape}")

# Test ComplexReLU
relu = ComplexReLU()
out_r, out_i = relu(out_r, out_i)
print(f"✓ ComplexReLU output shape: {out_r.shape}")

print("All tests passed!")
EOF
```

## Next Steps

After successful installation:

1. **Prepare Training Data**: Organize your video frames in the required format
2. **Configure Training**: Modify `train_complex.py` for your dataset
3. **Start Training**: Run the training script
4. **Monitor Progress**: Use TensorBoard to track training metrics
5. **Evaluate Results**: Test on validation set

For more information, see `COMPLEX_NN_GUIDE.md` in the documentation.

## Support

If you encounter issues:

1. Check the GitHub issues page
2. Verify all system requirements
3. Make sure NVIDIA drivers are up to date
4. Check CUDA compatibility with your GPU

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **CUDA Installation Guide**: https://docs.nvidia.com/cuda/
- **Original PhaseNet Paper**: https://arxiv.org/abs/1804.00884
- **Complex Networks Paper**: https://arxiv.org/abs/1705.09792

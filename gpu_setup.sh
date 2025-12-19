#!/bin/bash
# =============================================================================
# GPU Setup Script for Emotion Classification
# =============================================================================
# This script sets up the environment for GPU-accelerated training.
# Tested on Ubuntu 20.04/22.04 with NVIDIA GPUs.
#
# Usage:
#   chmod +x gpu_setup.sh
#   ./gpu_setup.sh
# =============================================================================

set -e  # Exit on error

echo "========================================"
echo "GPU Training Environment Setup"
echo "========================================"
echo ""

# -----------------------------------------------------------------------------
# Check for NVIDIA GPU
# -----------------------------------------------------------------------------
echo "1. Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "     $line"
    done
else
    echo "   ✗ No NVIDIA GPU found or drivers not installed"
    echo ""
    echo "   To install NVIDIA drivers on Ubuntu:"
    echo "     sudo apt update"
    echo "     sudo apt install nvidia-driver-535"  # or latest version
    echo "     sudo reboot"
    echo ""
    echo "   Continuing with CPU-only setup..."
fi
echo ""

# -----------------------------------------------------------------------------
# Check CUDA installation
# -----------------------------------------------------------------------------
echo "2. Checking CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "   ✓ CUDA Toolkit found: $CUDA_VERSION"
else
    echo "   ! CUDA Toolkit not found"
    echo ""
    echo "   To install CUDA on Ubuntu 22.04:"
    echo "     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "     sudo dpkg -i cuda-keyring_1.1-1_all.deb"
    echo "     sudo apt-get update"
    echo "     sudo apt-get install cuda-toolkit-12-2"
    echo ""
    echo "   After installation, add to ~/.bashrc:"
    echo "     export PATH=/usr/local/cuda/bin:\$PATH"
    echo "     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi
echo ""

# -----------------------------------------------------------------------------
# Create Python virtual environment
# -----------------------------------------------------------------------------
echo "3. Setting up Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv .venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# -----------------------------------------------------------------------------
# Upgrade pip
# -----------------------------------------------------------------------------
echo "4. Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "   ✓ pip upgraded"
echo ""

# -----------------------------------------------------------------------------
# Install PyTorch with CUDA support
# -----------------------------------------------------------------------------
echo "5. Installing PyTorch..."

# Detect CUDA version and install appropriate PyTorch
if command -v nvcc &> /dev/null; then
    CUDA_MAJOR=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d'.' -f1 | cut -c2-)
    
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "   Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    elif [ "$CUDA_MAJOR" -ge 11 ]; then
        echo "   Installing PyTorch for CUDA 11.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    else
        echo "   Installing PyTorch (CPU version)..."
        pip install torch torchvision torchaudio -q
    fi
else
    echo "   Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio -q
fi
echo "   ✓ PyTorch installed"
echo ""

# -----------------------------------------------------------------------------
# Verify PyTorch installation
# -----------------------------------------------------------------------------
echo "6. Verifying PyTorch installation..."
python3 -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
echo ""

# -----------------------------------------------------------------------------
# Install project dependencies
# -----------------------------------------------------------------------------
echo "7. Installing project dependencies..."
pip install -r requirements.txt -q
echo "   ✓ Dependencies installed"
echo ""

# -----------------------------------------------------------------------------
# Download NLTK data
# -----------------------------------------------------------------------------
echo "8. Downloading NLTK resources..."
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"
echo "   ✓ NLTK resources downloaded"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run training:"
echo "  python main.py                  # Run all approaches"
echo "  python main.py --approach fe    # Feature extraction only"
echo "  python main.py --approach ft    # Fine-tuning only"
echo "  python main.py --approach lora  # LoRA fine-tuning only"
echo ""
echo "For GPU memory issues, try reducing batch size in config.py"
echo ""

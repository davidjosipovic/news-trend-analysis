#!/bin/bash
# Setup script for AMD GPU (Radeon RX 5700 XT) support with PyTorch

echo "=========================================="
echo "AMD GPU Setup for PyTorch - Arch Linux"
echo "=========================================="
echo ""

# Step 1: Install ROCm
echo "Step 1: Installing ROCm packages..."
echo "This requires sudo privileges."
echo ""
echo "Run: sudo pacman -S rocm-hip-sdk rocm-opencl-sdk rocm-smi-lib"
echo ""
read -p "Press Enter after installing ROCm packages..."

# Step 2: Add user to groups
echo ""
echo "Step 2: Adding user to video and render groups..."
sudo usermod -a -G video,render $USER
echo "✓ User added to groups (logout and login required)"

# Step 3: Verify ROCm
echo ""
echo "Step 3: Verifying ROCm installation..."
if [ -f /opt/rocm/bin/rocminfo ]; then
    echo "✓ ROCm found!"
    /opt/rocm/bin/rocminfo | grep -A 5 "Agent"
else
    echo "✗ ROCm not found. Please install it first."
    exit 1
fi

# Step 4: Install PyTorch with ROCm
echo ""
echo "Step 4: Installing PyTorch with ROCm support..."
cd /home/David/Documents/GithubRepos/news-trend-analysis
source .venv/bin/activate

# Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with ROCm 6.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

# Step 5: Verify GPU is detected
echo ""
echo "Step 5: Testing PyTorch GPU detection..."
python3 << 'PYEOF'
import torch
print("="*50)
print("PyTorch GPU Test")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("✓ GPU is ready to use!")
else:
    print("✗ GPU not detected")
print("="*50)
PYEOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Note: You may need to logout and login again for group changes to take effect."
echo ""
echo "To verify GPU is working, run:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"

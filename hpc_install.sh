#!/bin/bash
# Quick installation script for mmfuse on HPC machines
# Usage: bash hpc_install.sh [cu125|cu124|cu121|cu118|cpu]

set -e

CUDA_VERSION=${1:-cu125}
PYTHON_VERSION=3.10

echo "=========================================="
echo "mmfuse HPC Installation"
echo "=========================================="
echo "CUDA Version: $CUDA_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "[1/4] Creating Python environment..."
if command -v conda &> /dev/null; then
    echo "Using conda..."
    conda create -n mmfuse python=$PYTHON_VERSION -y
    conda activate mmfuse
else
    echo "Using venv..."
    python$PYTHON_VERSION -m venv mmfuse-env
    source mmfuse-env/bin/activate
fi

# Install PyTorch
echo "[2/4] Installing PyTorch with CUDA $CUDA_VERSION..."
case $CUDA_VERSION in
    cu125)
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu125
        ;;
    cu124)
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
        ;;
    cu121)
        pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
    cu118)
        pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    cpu)
        pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0
        ;;
    *)
        echo "Unknown CUDA version: $CUDA_VERSION"
        echo "Use: cu125, cu124, cu121, cu118, or cpu"
        exit 1
        ;;
esac

# Install dependencies
echo "[3/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo "[4/4] Verifying installation..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import open_clip; print(f'✓ Open-CLIP installed')"
python -c "import transformers; print(f'✓ Transformers installed')"
python -c "import librosa; print(f'✓ Librosa installed')"

if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✓ CUDA Available: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "⚠ CUDA not available (CPU mode)"
fi

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "Run tests:"
echo "  python tests/sanity/test_encoders_simple.py"
echo "  python tests/sanity/test_fusion.py"
echo "  python tests/sanity/test_end_to_end.py"
echo ""
echo "Run demo:"
echo "  python demo_multimodal_fusion.py"
echo ""

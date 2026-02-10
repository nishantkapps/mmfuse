# Installation Guide - PyTorch & Dependencies

## Quick Summary

The project works with multiple PyTorch versions depending on your CUDA setup:

| Setup | PyTorch Version | CUDA | Recommendation |
|-------|-----------------|------|-----------------|
| **CPU Only** (Current) | 2.3.1 | None | `pip install torch torchvision` |
| **HPC with CUDA 11.8** | 2.3.1 | cu118 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| **HPC with CUDA 12.4** | 2.4.1+ | cu124 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| **HPC with CUDA 12.5** | 2.4.1+ | cu124 | Use cu124 (cu125 support still rolling out) |

---

## Installation Steps

### Option 1: CPU-Only (Current Device - No CUDA)

```bash
cd /home/nishant/projects/mmfuse

# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install transformers==4.35.2 librosa==0.10.0 numpy==1.24.3 Pillow==10.0.0 soundfile==0.12.1

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
# OR
pip install clip-by-openai==1.3.0
```

**Verification:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Option 2: For HPC with CUDA 11.8 (cu118)

```bash
# Install PyTorch 2.3.1 with cu118 support
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install transformers==4.35.2 librosa==0.10.0 numpy==1.24.3 Pillow==10.0.0 soundfile==0.12.1

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

**Verification:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

### Option 3: For HPC with CUDA 12.4/12.5 (cu124)

```bash
# Install PyTorch 2.4.1 with cu124 support
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install transformers==4.35.2 librosa==0.10.0 numpy==1.24.3 Pillow==10.0.0 soundfile==0.12.1

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

**Verification:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

## Version Compatibility Matrix

### PyTorch 2.3.1
- ✅ CUDA 11.8 (cu118)
- ✅ CUDA 12.1 (cu121)
- ❌ CUDA 12.5 (use 2.4.1 instead)
- ✅ CPU
- **Best for:** cu118 HPC systems

### PyTorch 2.4.1
- ✅ CUDA 11.8 (cu118)
- ✅ CUDA 12.1 (cu121)
- ✅ CUDA 12.4 (cu124)
- ⚠️ CUDA 12.5 (use cu124 wheels, works with cu125)
- ✅ CPU
- **Best for:** cu125 HPC systems

### PyTorch 2.5.x
- ✅ CUDA 12.1+
- ❌ CUDA 11.8
- ✅ CPU
- **Not recommended:** For cu118 systems

---

## Recommended Setup Path

### Phase 1: Development (Current Device - CPU)

```bash
# Install CPU version for development
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.35.2 librosa==0.10.0 numpy==1.24.3 Pillow==10.0.0 soundfile==0.12.1
pip install git+https://github.com/openai/CLIP.git
```

### Phase 2: HPC Deployment

**If HPC has CUDA 11.8:**
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

**If HPC has CUDA 12.5:**
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

---

## Dependency Breakdown

| Package | Version | Purpose | Required? |
|---------|---------|---------|-----------|
| torch | 2.3.1+ | Deep learning framework | ✅ Core |
| torchvision | 0.18.1+ | Image processing | ✅ Core |
| transformers | 4.35.2 | HuggingFace models (Wav2Vec) | ✅ Core |
| librosa | 0.10.0 | Audio processing | ✅ Core |
| numpy | 1.24.3 | Numerical computing | ✅ Core |
| Pillow | 10.0.0 | Image I/O | ✅ Core |
| soundfile | 0.12.1 | Audio I/O | ✅ Core |
| CLIP | Latest | Vision-language model | ✅ Core |

---

## Testing Installation

After installation, verify everything works:

```python
# test_installation.py
import torch
import torchvision
import transformers
import librosa
import numpy as np
from PIL import Image
import clip

print("✓ torch:", torch.__version__)
print("✓ torchvision:", torchvision.__version__)
print("✓ transformers:", transformers.__version__)
print("✓ librosa:", librosa.__version__)
print("✓ numpy:", np.__version__)
print("✓ Pillow version:", Image.__version__)
print("✓ CLIP available")

# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
else:
    print("✓ CPU mode (CUDA not available)")

print("\n✅ All dependencies installed correctly!")
```

---

## Common Issues & Solutions

### Issue: ModuleNotFoundError: No module named 'clip'

**Solution:** Install CLIP from source:
```bash
pip install git+https://github.com/openai/CLIP.git
```

Or via PyPI:
```bash
pip install clip-by-openai==1.3.0
```

### Issue: RuntimeError: CUDA out of memory

**Solution (CPU mode):**
```bash
python -c "import torch; torch.cuda.is_available = lambda: False"
```

Or use CPU explicitly:
```python
system = RoboticFeedbackSystem(device="cpu")
```

### Issue: Version mismatch between torch and torchvision

**Solution:** Always install matching versions:
- torch 2.3.1 → torchvision 0.18.1
- torch 2.4.1 → torchvision 0.19.1

### Issue: CUDA not detected even after installation

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If False:
1. Verify CUDA installation: `nvcc --version`
2. Reinstall torch with correct CUDA version
3. Check environment variables: `echo $CUDA_HOME`

---

## Installation Script

For convenience, here's an automated installation script:

```bash
#!/bin/bash
# install.sh

echo "Robotic Multimodal Fusion System - Installation"
echo "==============================================="
echo ""

# Detect setup
if [ "$1" == "cu118" ]; then
    echo "Installing for CUDA 11.8..."
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
elif [ "$1" == "cu124" ]; then
    echo "Installing for CUDA 12.4/12.5..."
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing for CPU..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing remaining dependencies..."
pip install transformers==4.35.2 librosa==0.10.0 numpy==1.24.3 Pillow==10.0.0 soundfile==0.12.1

echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

echo ""
echo "✅ Installation complete!"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Usage:**
```bash
chmod +x install.sh

# For CPU
./install.sh

# For CUDA 11.8
./install.sh cu118

# For CUDA 12.5
./install.sh cu124
```

---

## My Recommendation for Your Workflow

1. **Now (CPU):** Install CPU-only PyTorch 2.3.1
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install transformers==4.35.2 librosa==0.10.0 numpy==1.24.3 Pillow==10.0.0 soundfile==0.12.1
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **On HPC (cu118):** Reinstall with
   ```bash
   pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **On HPC (cu125):** Use cu124 wheels
   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
   ```

The code is **100% compatible** across all these setups - only the PyTorch version changes!

---

## Next Steps

Run the test suite:
```bash
python test_encoders.py
python demo.py
```

All tests will work on CPU - just slower!

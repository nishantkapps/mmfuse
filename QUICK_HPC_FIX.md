# HPC Installation - Quick Steps

## Problem
You got: `ModuleNotFoundError: No module named 'open_clip'`

This means the Python environment on the HPC doesn't have the required dependencies installed.

## Solution - Follow These Steps

### Step 1: Load HPC Modules
```bash
module load conda  # or module load python
module load cuda/12.5  # or cu124, cu121, cu118 based on your HPC
```

Check available CUDA versions:
```bash
module avail cuda
```

### Step 2: Create Virtual Environment
```bash
cd /home/theta/nishant/mmfuse

# Option A: Using conda (recommended)
conda create -n mmfuse python=3.10 -y
conda activate mmfuse

# Option B: Using venv
python3 -m venv mmfuse-env
source mmfuse-env/bin/activate
```

### Step 3: Install PyTorch (Choose ONE based on your CUDA)

**For CUDA 12.5 (cu125) - RECOMMENDED:**
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu125
```

**For CUDA 12.4:**
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 12.1:**
```bash
pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**
```bash
pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0
```

### Step 4: Install All Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- open-clip-torch (fixes your import error)
- transformers
- librosa
- numpy, Pillow, scipy, etc.

### Step 5: Verify Everything Works
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import open_clip; print('✓ open_clip working')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 6: Run Tests
```bash
cd /home/theta/nishant/mmfuse
python tests/sanity/test_encoders_simple.py
```

Should now work! If still issues, try:
```bash
pip install --upgrade -r requirements.txt
```

## Automated Installation (Alternative)

Or use the provided script:
```bash
cd /home/theta/nishant/mmfuse
bash hpc_install.sh cu125  # or cu124, cu121, cu118, cpu
```

This automatically:
1. Creates environment
2. Installs PyTorch with correct CUDA
3. Installs all dependencies
4. Verifies installation

## If You Get Other Errors

**Error: "CUDA out of memory"**
- Edit test to use `device = "cpu"` 
- Reduce batch size

**Error: "No module named X"**
- Run: `pip install --upgrade -r requirements.txt`

**Error: PyTorch version mismatch**
- Check CUDA: `nvidia-smi`
- Reinstall matching PyTorch version

## What Was Fixed

Your `requirements.txt` now explicitly lists `open-clip-torch==2.24.0` which was missing before. This is what caused your error.

## Next Steps

After installation works:
1. Run all tests: `python tests/sanity/test_*.py`
2. Run demo: `python demo_multimodal_fusion.py`
3. Check performance on HPC GPUs
4. Proceed with training/deployment

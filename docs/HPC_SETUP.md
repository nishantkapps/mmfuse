# HPC Setup Guide - mmfuse on THETA

## Quick Start

### 1. Load Required Modules
```bash
# On THETA HPC, load Python and CUDA modules
module load conda
module load cuda/12.5  # or your available CUDA version (cu124, cu121, cu118)

# Check available CUDA versions:
module avail cuda
```

### 2. Create Virtual Environment
```bash
cd /home/theta/nishant/mmfuse

# Create conda environment
conda create -n mmfuse python=3.10 -y
conda activate mmfuse

# Or with venv
python3 -m venv mmfuse-env
source mmfuse-env/bin/activate
```

### 3. Install Dependencies

**For CUDA 12.5 (cu125) - RECOMMENDED:**
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu125
pip install -r requirements.txt
```

**For CUDA 12.4 (cu124):**
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**For CUDA 12.1 (cu121):**
```bash
pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For CUDA 11.8 (cu118):**
```bash
pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CPU-only (if no GPU available):**
```bash
pip install torch==2.10.0 torchvision==0.11.0 torchaudio==0.10.0
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import open_clip; print(f'Open-CLIP: {open_clip.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Running Tests

### From Root Directory
```bash
cd /home/theta/nishant/mmfuse
python tests/sanity/test_encoders_simple.py
python tests/sanity/test_fusion.py
python tests/sanity/test_end_to_end.py
```

### Run Demo
```bash
python demo_multimodal_fusion.py
```

## Troubleshooting

### Missing Modules
If you get `ModuleNotFoundError`, reinstall requirements:
```bash
pip install --upgrade -r requirements.txt
```

### CUDA Version Mismatch
Check available CUDA on HPC:
```bash
nvidia-smi  # Shows CUDA version
```

Update torch installation URL accordingly (cu121, cu118, cu124, etc.)

### Memory Issues
If running out of memory:
1. Reduce batch size in test scripts
2. Use CPU for testing: `device = "cpu"` in code
3. Enable gradient checkpointing for large batches

### Import Errors with open_clip
Sometimes needs reinstall:
```bash
pip uninstall open-clip-torch -y
pip install open-clip-torch==2.24.0
```

## HPC Job Script Example

Create `submit_job.sh`:
```bash
#!/bin/bash
#COBALT -n 4           # 4 nodes
#COBALT -t 30          # 30 minutes
#COBALT -q debug
#COBALT -A <project>

# Load modules
module load conda
module load cuda/12.1

# Activate environment
source activate mmfuse

# Run test
cd /home/theta/nishant/mmfuse
python tests/sanity/test_encoders_simple.py

# Or run demo
# python demo_multimodal_fusion.py
```

Submit with:
```bash
qsub submit_job.sh
```

## Performance on HPC

Expected performance with GPU:
- Vision encoding: ~5-10ms per batch (CLIP inference)
- Audio encoding: ~20-50ms per batch (1D CNN)
- Fusion: ~2-5ms per batch
- Total pipeline: ~50-100ms per batch

Expected performance with CPU:
- ~5-10x slower than GPU
- Good for debugging, not production

## Path Notes

On THETA:
- Code location: `/home/theta/nishant/mmfuse/`
- Virtual env: `/home/theta/nishant/mmfuse/mmfuse-env/`
- Tests: `/home/theta/nishant/mmfuse/tests/sanity/`

## Next Steps

1. Verify all tests pass
2. Profile performance on available GPUs
3. Adjust batch sizes for optimal throughput
4. Run training if using real robot data
5. Deploy with torch.jit.script for inference optimization

## Support

For issues, check:
1. `requirements.txt` - All dependencies listed
2. `docs/INSTALLATION.md` - Detailed installation
3. `docs/USAGE_GUIDE.md` - Usage examples
4. Run individual test files for debugging

# 🚀 GPU Setup Guide for AI Festival Project

## Why Models Cannot Use GPU

The main reason your models cannot use GPU is that you have **CPU-only versions** of the deep learning frameworks installed instead of GPU-enabled versions.

## Current Issues

### 1. 🔴 PaddlePaddle CPU Version
- You have `paddlepaddle` (CPU only) instead of `paddlepaddle-gpu`
- PaddleOCR cannot use GPU acceleration

### 2. 🔴 PyTorch CPU Version (possibly)
- May have CPU-only PyTorch installed
- Affects YOLO, YOLIC, and SmolVLM models

## 🔧 Complete GPU Setup Solution

### Step 1: Check Current Installation
```bash
# Check PyTorch GPU support
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"

# Check PaddlePaddle GPU support  
python -c "import paddle; print(f'Paddle CUDA: {paddle.device.is_compiled_with_cuda()}')"
```

### Step 2: Uninstall CPU Versions
```bash
# Uninstall CPU versions
pip uninstall torch torchvision paddlepaddle
```

### Step 3: Install GPU Versions
```bash
# Install PyTorch with CUDA support (for YOLO, YOLIC, SmolVLM)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PaddlePaddle with GPU support (for PaddleOCR)
pip install paddlepaddle-gpu

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Verify GPU Installation
```bash
# Run our diagnostic script
python diagnose_gpu.py

# Test models
python test_pytorch_models.py
```

## 🎯 Expected Performance Improvements

Once GPU is properly set up:

| Model | CPU Speed | GPU Speed | Improvement |
|-------|-----------|-----------|-------------|
| YOLIC | ~2-3s | ~0.5-1s | 3-5x faster |
| YOLO | ~1-2s | ~0.3-0.5s | 3-4x faster |
| SmolVLM | ~10-15s | ~2-3s | 5-8x faster |
| PaddleOCR | ~1-2s | ~0.5s | 2-3x faster |

## 🔍 Quick GPU Status Check

```python
# Run this to check all frameworks
python -c "
import torch
print(f'PyTorch CUDA: {torch.cuda.is_available()}')

import paddle
print(f'Paddle CUDA: {paddle.device.is_compiled_with_cuda()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

## 🐛 Troubleshooting

### Issue: CUDA Version Mismatch
If you get CUDA version errors:
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory
If you get GPU memory errors:
```python
# Add to your model initialization
torch.cuda.empty_cache()  # Clear GPU cache

# Use mixed precision (already implemented in SmolVLM)
# Models will automatically use float16 on GPU
```

### Issue: PaddlePaddle Installation Problems
```bash
# Clean installation
pip uninstall paddlepaddle paddlepaddle-gpu
pip install paddlepaddle-gpu

# If still issues, try specific version
pip install paddlepaddle-gpu==2.5.2
```

## ✅ Verification Commands

After installing GPU versions:

```bash
# 1. Test PyTorch models
python -c "
from app.utils.infer_yolic import YOLICPyTorchInference
from app.utils.infer_yolo import YOLOv8SignDetection
print('PyTorch models can use GPU:', torch.cuda.is_available())
"

# 2. Test PaddleOCR
python -c "
from app.utils.infer_paddleocr import PaddleOCRInference
ocr = PaddleOCRInference()
print('PaddleOCR GPU status:', ocr.use_gpu)
"

# 3. Run comprehensive test
python test_pytorch_models.py
```

## 📋 Updated Requirements

Your `requirements.txt` has been updated to use GPU versions:
- `paddlepaddle-gpu` instead of `paddlepaddle`
- Added comment explaining GPU vs CPU versions

## 🎉 Expected Results

Once properly configured:
- ✅ All models will automatically detect and use GPU
- ✅ 3-8x faster inference speeds
- ✅ Better memory management
- ✅ Real-time processing capabilities

The key is installing the **GPU-enabled versions** of the frameworks, not just the CPU versions!

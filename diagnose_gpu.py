#!/usr/bin/env python3
"""
GPU Installation Diagnostic Script
Check why models cannot use GPU
"""

import sys

def check_pytorch_gpu():
    """Check PyTorch GPU support"""
    print("🔍 PYTORCH GPU CHECK")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ PyTorch CUDA not available")
            print("💡 Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False

def check_paddlepaddle_gpu():
    """Check PaddlePaddle GPU support"""
    print("\n🔍 PADDLEPADDLE GPU CHECK")
    print("=" * 50)
    
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
        print(f"Compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")
        
        if paddle.device.is_compiled_with_cuda():
            print(f"GPU count: {paddle.device.cuda.device_count()}")
            print("✅ PaddlePaddle has GPU support")
            return True
        else:
            print("❌ PaddlePaddle CPU version installed")
            print("💡 Install GPU version with:")
            print("   pip uninstall paddlepaddle")
            print("   pip install paddlepaddle-gpu")
            return False
            
    except Exception as e:
        print(f"❌ PaddlePaddle check failed: {e}")
        return False

def check_ultralytics_gpu():
    """Check Ultralytics GPU support"""
    print("\n🔍 ULTRALYTICS GPU CHECK")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        import torch
        
        print("Ultralytics installed ✅")
        print(f"PyTorch backend CUDA: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("✅ Ultralytics can use GPU via PyTorch")
            return True
        else:
            print("❌ Ultralytics will use CPU only")
            return False
            
    except Exception as e:
        print(f"❌ Ultralytics check failed: {e}")
        return False

def check_transformers_gpu():
    """Check Transformers GPU support"""
    print("\n🔍 TRANSFORMERS GPU CHECK")
    print("=" * 50)
    
    try:
        import transformers
        import torch
        
        print(f"Transformers version: {transformers.__version__}")
        print(f"PyTorch backend CUDA: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("✅ Transformers can use GPU via PyTorch")
            return True
        else:
            print("❌ Transformers will use CPU only")
            return False
            
    except Exception as e:
        print(f"❌ Transformers check failed: {e}")
        return False

def get_installation_commands():
    """Provide installation commands for GPU support"""
    print("\n💡 GPU INSTALLATION GUIDE")
    print("=" * 50)
    
    print("To enable GPU support for all models:")
    print()
    print("1. 🔥 PyTorch GPU (for YOLO, YOLIC, SmolVLM):")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("2. 🚀 PaddlePaddle GPU (for PaddleOCR):")
    print("   pip uninstall paddlepaddle")
    print("   pip install paddlepaddle-gpu")
    print()
    print("3. 📦 Or install both together:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("   pip uninstall paddlepaddle")
    print("   pip install paddlepaddle-gpu")
    print()
    print("Note: Make sure you have CUDA 11.8+ installed on your system")
    print("Check CUDA: nvidia-smi")

def main():
    """Run all GPU checks"""
    print("🚀 GPU INSTALLATION DIAGNOSTIC")
    print("=" * 60)
    
    results = {
        "PyTorch": check_pytorch_gpu(),
        "PaddlePaddle": check_paddlepaddle_gpu(), 
        "Ultralytics": check_ultralytics_gpu(),
        "Transformers": check_transformers_gpu()
    }
    
    print("\n📊 SUMMARY")
    print("=" * 50)
    
    for framework, has_gpu in results.items():
        status = "🟢 GPU Ready" if has_gpu else "🔴 CPU Only"
        print(f"{framework:12} : {status}")
    
    gpu_ready = sum(results.values())
    total = len(results)
    
    print(f"\nGPU-ready frameworks: {gpu_ready}/{total}")
    
    if gpu_ready == 0:
        print("\n❌ No frameworks have GPU support!")
        print("This means all models will run on CPU (much slower)")
    elif gpu_ready < total:
        print(f"\n⚠️ Only {gpu_ready} frameworks have GPU support")
        print("Some models will be slower on CPU")
    else:
        print("\n🎉 All frameworks have GPU support!")
        print("Models will run at optimal speed")
    
    if gpu_ready < total:
        get_installation_commands()
    
    return gpu_ready > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

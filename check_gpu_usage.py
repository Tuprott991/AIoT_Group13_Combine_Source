#!/usr/bin/env python3
"""
GPU Utilization Check for PyTorch Models
Check if all models are properly loading to GPU and utilizing CUDA
"""

import os
import sys
import torch
import traceback

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    print("=" * 60)
    print("🔍 GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {gpu_props.major}.{gpu_props.minor}")
        
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        return True
    else:
        print("❌ CUDA not available - models will run on CPU")
        return False

def check_yolic_gpu():
    """Check YOLIC model GPU utilization"""
    print("\n" + "=" * 60)
    print("🔍 YOLIC MODEL GPU CHECK")
    print("=" * 60)
    
    try:
        from utils.infer_yolic import YOLICPyTorchInference
        
        model_path = "app/ai_models/yolic_m2/yolic_m2.pth.tar"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"Loading YOLIC model...")
        yolic_model = YOLICPyTorchInference(model_path)
        
        print(f"✅ Model device: {yolic_model.device}")
        print(f"✅ Model parameters device: {next(yolic_model.model.parameters()).device}")
        
        # Check if model is actually on GPU
        model_on_gpu = next(yolic_model.model.parameters()).is_cuda
        print(f"✅ Model on GPU: {model_on_gpu}")
        
        # Test memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        
        return model_on_gpu
        
    except Exception as e:
        print(f"❌ YOLIC GPU check failed: {e}")
        traceback.print_exc()
        return False

def check_yolo_gpu():
    """Check YOLO model GPU utilization"""
    print("\n" + "=" * 60)
    print("🔍 YOLO MODEL GPU CHECK")
    print("=" * 60)
    
    try:
        from utils.infer_yolo import YOLOv8SignDetection
        
        model_path = "app/ai_models/yolov8_sign_detection/yolov8_traffic_sign.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"Loading YOLO model...")
        yolo_model = YOLOv8SignDetection(model_path)
        
        print(f"✅ Model device: {yolo_model.device}")
        
        # Check if ultralytics model is on GPU
        if hasattr(yolo_model.model, 'model'):
            model_device = next(yolo_model.model.model.parameters()).device
            print(f"✅ YOLO model parameters device: {model_device}")
            model_on_gpu = str(model_device) != 'cpu'
        else:
            # Fallback check
            model_on_gpu = torch.cuda.is_available()
            print(f"✅ YOLO using CUDA: {model_on_gpu}")
        
        print(f"✅ Model on GPU: {model_on_gpu}")
        
        # Test memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        
        return model_on_gpu
        
    except Exception as e:
        print(f"❌ YOLO GPU check failed: {e}")
        traceback.print_exc()
        return False

def check_smolvlm_gpu():
    """Check SmolVLM model GPU utilization"""
    print("\n" + "=" * 60)
    print("🔍 SMOLVLM MODEL GPU CHECK")
    print("=" * 60)
    
    try:
        from utils.infer_smolvlm import SmolVLMInference
        
        print(f"Loading SmolVLM model...")
        vlm_model = SmolVLMInference()
        
        print(f"✅ Model device: {vlm_model.device}")
        
        # Check if model parameters are on GPU
        model_device = next(vlm_model.model.parameters()).device
        print(f"✅ Model parameters device: {model_device}")
        
        model_on_gpu = str(model_device) != 'cpu'
        print(f"✅ Model on GPU: {model_on_gpu}")
        
        # Test memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        
        return model_on_gpu
        
    except Exception as e:
        print(f"❌ SmolVLM GPU check failed: {e}")
        traceback.print_exc()
        return False

def check_paddleocr_gpu():
    """Check PaddleOCR GPU utilization"""
    print("\n" + "=" * 60)
    print("🔍 PADDLEOCR GPU CHECK")
    print("=" * 60)
    
    try:
        from utils.infer_paddleocr import PaddleOCRInference
        
        print(f"Loading PaddleOCR model...")
        ocr_model = PaddleOCRInference()
        
        print(f"✅ PaddleOCR using GPU: {ocr_model.use_gpu}")
        print(f"✅ Device: {ocr_model.device}")
        
        return ocr_model.use_gpu
        
    except Exception as e:
        print(f"❌ PaddleOCR GPU check failed: {e}")
        traceback.print_exc()
        return False

def run_inference_timing_test():
    """Run timing tests to compare CPU vs GPU performance"""
    print("\n" + "=" * 60)
    print("⏱️ INFERENCE TIMING TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping timing test")
        return
    
    # This would require actual model loading and inference
    # For now, just show the concept
    print("🚀 GPU timing tests would go here...")
    print("   - Load models on CPU and GPU")
    print("   - Run same inference on both")
    print("   - Compare timing results")

def main():
    """Run all GPU checks"""
    print("🚀 Starting GPU Utilization Check...")
    
    # Check basic GPU availability
    gpu_available = check_gpu_availability()
    
    if not gpu_available:
        print("\n⚠️ No GPU available - all models will run on CPU")
        print("To use GPU:")
        print("1. Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("2. Install PaddlePaddle GPU: pip install paddlepaddle-gpu")
        return False
    
    # Check individual models
    results = {
        "YOLIC": check_yolic_gpu(),
        "YOLO": check_yolo_gpu(),
        "SmolVLM": check_smolvlm_gpu(),
        "PaddleOCR": check_paddleocr_gpu()
    }
    
    # Run timing test
    run_inference_timing_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 GPU UTILIZATION SUMMARY")
    print("=" * 60)
    
    for model_name, using_gpu in results.items():
        status = "🟢 GPU" if using_gpu else "🔴 CPU"
        print(f"{model_name:12} : {status}")
    
    gpu_models = sum(results.values())
    total_models = len(results)
    
    print(f"\nModels using GPU: {gpu_models}/{total_models}")
    
    if gpu_models == total_models:
        print("🎉 All models are using GPU! Optimal performance achieved.")
    elif gpu_models > 0:
        print("⚠️ Some models are using GPU. Check configuration for CPU models.")
    else:
        print("❌ No models are using GPU. Check CUDA installation.")
    
    return gpu_models > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

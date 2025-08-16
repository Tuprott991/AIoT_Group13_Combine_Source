#!/usr/bin/env python3
"""
Quick test to verify PyTorch-PaddlePaddle conflict resolution
Tests import order and basic functionality
"""

import sys
import traceback

def test_import_order():
    """Test import order fix"""
    print("🧪 Testing import order fix...")
    
    try:
        from app.utils.import_fix import check_cuda_conflicts, get_safe_paddle_ocr
        print("✅ Import fix module loaded successfully")
        
        # Check for conflicts
        conflicts = check_cuda_conflicts()
        print(f"CUDA status: {conflicts}")
        
        # Test safe PaddleOCR
        ocr = get_safe_paddle_ocr()
        if ocr:
            print("✅ PaddleOCR initialized successfully via import fix")
            return True
        else:
            print("⚠️ PaddleOCR failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ Import fix test failed: {e}")
        traceback.print_exc()
        return False

def test_pytorch_first():
    """Test PyTorch functionality"""
    print("\n🧪 Testing PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: {torch.cuda.device_count()} devices")
            # Test tensor creation
            x = torch.randn(3, 3).cuda()
            y = x + 1
            print("✅ PyTorch GPU operations working")
        else:
            print("⚠️ PyTorch CUDA not available")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_paddleocr_direct():
    """Test PaddleOCR direct import"""
    print("\n🧪 Testing PaddleOCR direct import...")
    
    try:
        from paddleocr import PaddleOCR
        
        # Try basic initialization
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("✅ PaddleOCR direct import successful")
        
        # Test with a simple image (if available)
        import numpy as np
        test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
        result = ocr.ocr(test_img, cls=True)
        print("✅ PaddleOCR basic inference working")
        
        return True
        
    except Exception as e:
        print(f"❌ PaddleOCR direct test failed: {e}")
        return False

def test_model_loader():
    """Test the main model loader"""
    print("\n🧪 Testing main model loader...")
    
    try:
        from app.model_loader import ModelLoader
        
        loader = ModelLoader()
        loader.load_all_models()
        
        # Check which models loaded
        models_status = {
            'yolo': hasattr(loader, 'yolo_model') and loader.yolo_model is not None,
            'yolic': hasattr(loader, 'yolic_model') and loader.yolic_model is not None,
            'paddleocr': hasattr(loader, 'paddleocr_model') and loader.paddleocr_model is not None,
            'smolvlm': hasattr(loader, 'smolvlm_model') and loader.smolvlm_model is not None,
        }
        
        print(f"Models loaded: {models_status}")
        
        if models_status['paddleocr']:
            print("✅ PaddleOCR loaded via model loader")
            return True
        else:
            print("⚠️ PaddleOCR not loaded via model loader")
            return False
            
    except Exception as e:
        print(f"❌ Model loader test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PYTORCH-PADDLEOCR CONFLICT RESOLUTION TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Import order
    results.append(("Import Order Fix", test_import_order()))
    
    # Test 2: PyTorch
    results.append(("PyTorch", test_pytorch_first()))
    
    # Test 3: PaddleOCR direct
    results.append(("PaddleOCR Direct", test_paddleocr_direct()))
    
    # Test 4: Model loader
    results.append(("Model Loader", test_model_loader()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} | {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Conflict resolution successful!")
    elif passed >= 2:
        print(f"\n⚠️ Partial success. {passed} tests passed.")
        print("💡 Run fix_pytorch_paddle_conflict.py to resolve remaining issues")
    else:
        print("\n❌ Most tests failed. Manual intervention required.")
        print("💡 Check PYTORCH_PADDLE_SOLUTION.md for troubleshooting")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test cancelled")
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        traceback.print_exc()

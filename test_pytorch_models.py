#!/usr/bin/env python3
"""
Test script to verify PyTorch model loading and inference
"""

import os
import sys
import cv2
import traceback

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_yolic_model():
    """Test YOLIC PyTorch model loading and inference"""
    print("=" * 50)
    print("Testing YOLIC PyTorch Model")
    print("=" * 50)
    
    try:
        from utils.infer_yolic import YOLICPyTorchInference
        
        model_path = "ai_models/yolic_m2/yolic_m2.pth.tar"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"Loading model from: {model_path}")
        yolic_model = YOLICPyTorchInference(model_path)
        print("✅ YOLIC model loaded successfully!")
        
        # Test with a sample image if available
        test_images = ["app/test_img.jpg", "app/test_img2.png", "app/test_img3.jpg"]
        test_image = None
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if test_image:
            print(f"Testing inference with: {test_image}")
            image = cv2.imread(test_image)
            if image is not None:
                detections = yolic_model.inference(image)
                print(f"✅ Inference successful! Found {len(detections)} detections")
                for detection in detections:
                    print(f"  - {detection['object']} at {detection['position']} (confidence: {detection['confidence']:.3f})")
            else:
                print(f"❌ Could not read test image: {test_image}")
        else:
            print("⚠️ No test images found, skipping inference test")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLIC model test failed: {e}")
        traceback.print_exc()
        return False

def test_yolo_model():
    """Test YOLO PyTorch model loading and inference"""
    print("\n" + "=" * 50)
    print("Testing YOLO PyTorch Model")
    print("=" * 50)
    
    try:
        from utils.infer_yolo import YOLOv8SignDetection
        
        model_path = "ai_models/yolov8_sign_detection/yolov8_traffic_sign.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"Loading model from: {model_path}")
        yolo_model = YOLOv8SignDetection(model_path)
        print("✅ YOLO model loaded successfully!")
        
        # Test with a sample image if available
        test_images = ["app/test_sign.jpg", "app/test_sign2.jpg", "app/test_sign3.jpg"]
        test_image = None
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if test_image:
            print(f"Testing inference with: {test_image}")
            image = cv2.imread(test_image)
            if image is not None:
                detections = yolo_model.inference(image)
                print(f"✅ Inference successful! Found {len(detections)} detections")
                for detection in detections:
                    print(f"  - {detection['sign']} at {detection['position']} (confidence: {detection['confidence']:.3f})")
            else:
                print(f"❌ Could not read test image: {test_image}")
        else:
            print("⚠️ No test images found, skipping inference test")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        traceback.print_exc()
        return False

def test_paddleocr_model():
    """Test PaddleOCR model loading and inference"""
    print("\n" + "=" * 50)
    print("Testing PaddleOCR Model")
    print("=" * 50)
    
    try:
        from utils.infer_paddleocr import PaddleOCRInference
        
        print("Loading PaddleOCR model...")
        ocr_model = PaddleOCRInference()
        print("✅ PaddleOCR model loaded successfully!")
        
        # Test with a sample image if available
        test_images = ["app/test_img.jpg", "app/test_img2.png", "app/test_img3.jpg"]
        test_image = None
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if test_image:
            print(f"Testing OCR with: {test_image}")
            result = ocr_model.extract_text(test_image)
            print(f"✅ OCR successful! Found {result['total_texts']} text regions")
            for i, text_info in enumerate(result['texts'][:3]):  # Show first 3 results
                print(f"  {i+1}: '{text_info['text']}' (confidence: {text_info['confidence']:.3f})")
        else:
            print("⚠️ No test images found, skipping OCR test")
        
        return True
        
    except Exception as e:
        print(f"❌ PaddleOCR model test failed: {e}")
        traceback.print_exc()
        return False

def test_smolvlm_model():
    """Test SmolVLM PyTorch model loading and inference"""
    print("\n" + "=" * 50)
    print("Testing SmolVLM PyTorch Model")
    print("=" * 50)
    
    try:
        from utils.infer_smolvlm import SmolVLMInference
        
        print("Loading SmolVLM model...")
        vlm_model = SmolVLMInference()
        print("✅ SmolVLM model loaded successfully!")
        
        # Test with a sample image if available
        test_images = ["app/test_img.jpg", "app/test_img2.png", "app/test_img3.jpg"]
        test_image = None
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if test_image:
            print(f"Testing VLM with: {test_image}")
            description = vlm_model.describe_image(test_image, "What do you see in this image?")
            print(f"✅ VLM inference successful!")
            print(f"Description: {description}")
        else:
            print("⚠️ No test images found, skipping VLM test")
        
        return True
        
    except Exception as e:
        print(f"❌ SmolVLM model test failed: {e}")
        traceback.print_exc()
        return False
    """Test the model loader with PyTorch models"""
    print("\n" + "=" * 50)
    print("Testing Model Loader")
    print("=" * 50)
    
    try:
        from model_loader import load_all_models, get_loaded_models, get_model
        
        print("Loading all models...")
        load_all_models()
        
        loaded_models = get_loaded_models()
        print(f"✅ Model loader completed! Loaded models: {loaded_models}")
        
        # Test getting individual models
        yolo_model = get_model("yolo_sign_detection")
        yolic_model = get_model("yolic_hazard_detection")
        
        if yolo_model:
            print("✅ YOLO model retrieved from loader")
        else:
            print("❌ YOLO model not found in loader")
        
        if yolic_model:
            print("✅ YOLIC model retrieved from loader")
        else:
            print("❌ YOLIC model not found in loader")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loader test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loader():
    """Test the model loader with all PyTorch models"""
    print("\n" + "=" * 50)
    print("Testing Model Loader")
    print("=" * 50)
    
    try:
        from model_loader import load_all_models, get_loaded_models, get_model
        
        print("Loading all models...")
        load_all_models()
        
        loaded_models = get_loaded_models()
        print(f"✅ Model loader completed! Loaded models: {loaded_models}")
        
        # Test getting individual models
        yolo_model = get_model("yolo_sign_detection")
        yolic_model = get_model("yolic_hazard_detection")
        smolvlm_model = get_model("smolvlm")
        paddleocr_model = get_model("paddleocr")
        
        results = {}
        results["YOLO"] = yolo_model is not None
        results["YOLIC"] = yolic_model is not None
        results["SmolVLM"] = smolvlm_model is not None
        results["PaddleOCR"] = paddleocr_model is not None
        
        for model_name, loaded in results.items():
            status = "✅" if loaded else "❌"
            print(f"{status} {model_name} model retrieved from loader")
        
        return all(results.values())
        
    except Exception as e:
        print(f"❌ Model loader test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting PyTorch Model Tests...")
    
    results = {
        "YOLIC": test_yolic_model(),
        "YOLO": test_yolo_model(),
        "PaddleOCR": test_paddleocr_model(),
        "SmolVLM": test_smolvlm_model(),
        "Model Loader": test_model_loader()
    }
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! PyTorch models are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

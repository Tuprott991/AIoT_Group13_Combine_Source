"""
Simple API Tester for AI Festival Project
Run individual tests or quick checks
"""

import requests
import json
import os
import sys
from pathlib import Path

# Configuration
API_BASE = "http://127.0.0.1:8000"

def print_header(title):
    print(f"\n{'='*50}")
    print(f"🧪 {title}")
    print(f"{'='*50}")

def print_result(success, message):
    status = "✅" if success else "❌"
    print(f"{status} {message}")

def test_server_running():
    """Quick test to see if server is running"""
    print_header("Server Health Check")
    
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=5)
        if response.status_code == 200:
            print_result(True, "Server is running")
            print(f"📖 API Documentation: {API_BASE}/docs")
            return True
        else:
            print_result(False, f"Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_result(False, "Cannot connect to server")
        print("💡 Start server with: python app/main.py")
        return False
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False

def test_model_health():
    """Test model loading status"""
    print_header("Model Health Check")
    
    try:
        response = requests.get(f"{API_BASE}/health/models", timeout=30)
        if response.status_code == 200:
            data = response.json()
            total_models = data.get("total_models", 0)
            loaded_models = data.get("loaded_models", [])
            
            print_result(True, f"Models loaded: {total_models}")
            print(f"📋 Loaded models: {', '.join(loaded_models)}")
            
            # Show detailed model status
            model_details = data.get("model_details", {})
            for model_name, details in model_details.items():
                status = details.get("status", "unknown")
                model_type = details.get("type", "unknown")
                print(f"   🤖 {model_name}: {status} ({model_type})")
            
            return total_models > 0
        else:
            print_result(False, f"Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Error checking model health: {str(e)}")
        return False

def find_test_image():
    """Find a test image to use"""
    # Look for test images in common locations
    possible_images = [
        "test_img.jpg", "test_img2.png", "test_img3.jpg", "test_img4.jpg",
        "test_sign.jpg", "test_sign2.jpg", "test_sign3.jpg", "test_sign4.jpg",
        "test_sign_detected.jpg"
    ]
    
    search_paths = [".", "app", "app/"]
    
    for search_path in search_paths:
        for img in possible_images:
            img_path = Path(search_path) / img
            if img_path.exists():
                return str(img_path)
    
    return None

def test_endpoint(endpoint, image_path, description="", params=None):
    """Test a specific endpoint with an image"""
    print(f"\n🔍 Testing {description or endpoint}...")
    
    if not os.path.exists(image_path):
        print_result(False, f"Test image not found: {image_path}")
        return False
    
    try:
        # Read image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Prepare URL
        url = f"{API_BASE}/api{endpoint}"
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url += f"?{param_str}"
        
        # Make request
        headers = {"Content-Type": "application/octet-stream"}
        print(f"📤 Sending request to: {url}")
        print(f"📦 Image size: {len(image_data)} bytes")
        
        response = requests.post(url, data=image_data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print_result(True, f"Request successful")
            
            # Print response details based on endpoint type
            if "detect" in endpoint:
                detections = data.get("detections", [])
                print(f"🎯 Found {len(detections)} detections")
                for i, detection in enumerate(detections[:3]):  # Show first 3
                    label = detection.get("label", "unknown")
                    confidence = detection.get("confidence", 0)
                    print(f"   {i+1}. {label} (confidence: {confidence:.2f})")
                if len(detections) > 3:
                    print(f"   ... and {len(detections) - 3} more")
                    
            elif "analyze" in endpoint:
                description = data.get("description", "")
                prompt = data.get("prompt", "")
                print(f"💭 Prompt: {prompt}")
                print(f"📝 Description: {description[:200]}{'...' if len(description) > 200 else ''}")
                
            elif "extract" in endpoint:
                texts = data.get("texts", [])
                print(f"📄 Found {len(texts)} text regions")
                for i, text_info in enumerate(texts[:3]):  # Show first 3
                    text = text_info.get("text", "")
                    confidence = text_info.get("confidence", 0)
                    print(f"   {i+1}. \"{text}\" (confidence: {confidence:.2f})")
            
            return True
            
        else:
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", response.text)
            except:
                error_detail = response.text
            
            print_result(False, f"Request failed (Status: {response.status_code})")
            print(f"❌ Error: {error_detail}")
            return False
            
    except requests.exceptions.Timeout:
        print_result(False, "Request timed out (60 seconds)")
        return False
    except Exception as e:
        print_result(False, f"Error: {str(e)}")
        return False

def run_quick_test():
    """Run a quick test of all endpoints"""
    print("🚀 Quick API Test")
    
    # Check server
    if not test_server_running():
        return
    
    # Check models
    if not test_model_health():
        print("⚠️  Models not loaded properly. Tests may fail.")
    
    # Find test image
    test_image = find_test_image()
    if not test_image:
        print_result(False, "No test image found")
        print("💡 Please add a test image (jpg/png) to the current directory")
        return
    
    print(f"\n📸 Using test image: {test_image}")
    
    # Test endpoints
    tests = [
        ("/hazzard_detect/detect", "Hazard Detection"),
        ("/sign_detect/detect", "Sign Detection"),
        ("/smolvlm/analyze", "SmolVLM Analysis"),
        ("/paddleocr/extract", "PaddleOCR Text Extraction")
    ]
    
    results = []
    for endpoint, description in tests:
        success = test_endpoint(endpoint, test_image, description)
        results.append((description, success))
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        print_result(success, description)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "server":
            test_server_running()
        elif command == "models":
            test_model_health()
        elif command == "hazard" and len(sys.argv) > 2:
            test_endpoint("/hazzard_detect/detect", sys.argv[2], "Hazard Detection")
        elif command == "sign" and len(sys.argv) > 2:
            test_endpoint("/sign_detect/detect", sys.argv[2], "Sign Detection")
        elif command == "analyze" and len(sys.argv) > 2:
            prompt = sys.argv[3] if len(sys.argv) > 3 else "Describe what you see"
            test_endpoint("/smolvlm/analyze", sys.argv[2], "SmolVLM Analysis", {"prompt": prompt})
        elif command == "ocr" and len(sys.argv) > 2:
            test_endpoint("/paddleocr/extract", sys.argv[2], "PaddleOCR")
        else:
            print("Usage:")
            print("  python simple_test.py                    # Run all tests")
            print("  python simple_test.py server             # Test server only")
            print("  python simple_test.py models             # Test models only")
            print("  python simple_test.py hazard image.jpg   # Test hazard detection")
            print("  python simple_test.py sign image.jpg     # Test sign detection")
            print("  python simple_test.py analyze image.jpg [prompt]  # Test SmolVLM")
            print("  python simple_test.py ocr image.jpg      # Test OCR")
    else:
        run_quick_test()

if __name__ == "__main__":
    main()

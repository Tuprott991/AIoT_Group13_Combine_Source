"""
API Testing Script for AI Festival Project
This script provides comprehensive testing for all API endpoints
"""

import requests
import json
import os
import sys
from typing import Dict, Any
import time

# API Configuration
BASE_URL = "http://127.0.0.1:8000"
API_BASE = f"{BASE_URL}/api"

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, response_time: float, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({response_time:.2f}s)")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "response_time": response_time,
            "details": details
        })
    
    def test_server_health(self) -> bool:
        """Test if server is running"""
        print("\n🏥 Testing Server Health...")
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/docs", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test("Server Health Check", True, response_time, "FastAPI docs accessible")
                return True
            else:
                self.log_test("Server Health Check", False, response_time, f"Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Server Health Check", False, 0, f"Connection error: {str(e)}")
            return False
    
    def test_model_health(self) -> bool:
        """Test model health endpoint"""
        print("\n🤖 Testing Model Health...")
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health/models", timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                total_models = data.get("total_models", 0)
                loaded_models = data.get("loaded_models", [])
                self.log_test("Model Health Check", True, response_time, 
                            f"Models loaded: {total_models} ({', '.join(loaded_models)})")
                return True
            else:
                self.log_test("Model Health Check", False, response_time, f"Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Model Health Check", False, 0, f"Error: {str(e)}")
            return False
    
    def test_model_management_endpoints(self):
        """Test model management endpoints"""
        print("\n⚙️ Testing Model Management Endpoints...")
        
        # Test model status
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base}/models/status", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Model Status Endpoint", True, response_time, 
                            f"Response: {len(data.get('model_details', {}))} models")
            else:
                self.log_test("Model Status Endpoint", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Model Status Endpoint", False, 0, f"Error: {str(e)}")
        
        # Test model list
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base}/models/list", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                loaded = list(data.get("loaded_models", {}).keys())
                self.log_test("Model List Endpoint", True, response_time, 
                            f"Loaded models: {loaded}")
            else:
                self.log_test("Model List Endpoint", False, response_time, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Model List Endpoint", False, 0, f"Error: {str(e)}")
    
    def test_endpoint_with_image(self, endpoint: str, image_path: str, 
                                additional_params: Dict = None) -> bool:
        """Test an endpoint that requires image data"""
        endpoint_name = endpoint.split('/')[-1]
        
        if not os.path.exists(image_path):
            self.log_test(f"{endpoint_name} (No Test Image)", False, 0, 
                         f"Test image not found: {image_path}")
            return False
        
        try:
            # Read image data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Prepare request
            url = f"{self.api_base}{endpoint}"
            if additional_params:
                url += "?" + "&".join([f"{k}={v}" for k, v in additional_params.items()])
            
            headers = {"Content-Type": "application/octet-stream"}
            
            start_time = time.time()
            response = requests.post(url, data=image_data, headers=headers, timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                # Extract relevant info based on endpoint
                if "detect" in endpoint:
                    detections = len(data.get("detections", []))
                    self.log_test(f"{endpoint_name}", True, response_time, 
                                f"Found {detections} detections")
                elif "analyze" in endpoint:
                    description = data.get("description", "")[:100] + "..."
                    self.log_test(f"{endpoint_name}", True, response_time, 
                                f"Description: {description}")
                elif "extract" in endpoint:
                    texts = len(data.get("texts", []))
                    self.log_test(f"{endpoint_name}", True, response_time, 
                                f"Found {texts} text regions")
                else:
                    self.log_test(f"{endpoint_name}", True, response_time, "Response received")
                return True
            else:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", "")
                except:
                    error_detail = response.text[:100]
                
                self.log_test(f"{endpoint_name}", False, response_time, 
                            f"Status: {response.status_code}, Error: {error_detail}")
                return False
                
        except requests.exceptions.Timeout:
            self.log_test(f"{endpoint_name}", False, 60, "Request timeout (60s)")
            return False
        except Exception as e:
            self.log_test(f"{endpoint_name}", False, 0, f"Error: {str(e)}")
            return False
    
    def find_test_images(self) -> list:
        """Find available test images"""
        possible_images = [
            "test_img.jpg",
            "test_img2.png", 
            "test_img3.jpg",
            "test_img4.jpg",
            "test_sign.jpg",
            "test_sign2.jpg",
            "test_sign3.jpg",
            "test_sign4.jpg",
            "test_sign_detected.jpg"
        ]
        
        found_images = []
        
        # Check in current directory and app directory
        search_paths = [".", "app", "app/", "../app"]
        
        for search_path in search_paths:
            for img in possible_images:
                full_path = os.path.join(search_path, img)
                if os.path.exists(full_path):
                    found_images.append(full_path)
                    break
        
        return found_images
    
    def test_detection_endpoints(self):
        """Test all detection endpoints"""
        print("\n🔍 Testing Detection Endpoints...")
        
        test_images = self.find_test_images()
        if not test_images:
            print("⚠️  No test images found! Please ensure you have test images in the project directory.")
            return
        
        test_image = test_images[0]
        print(f"📸 Using test image: {test_image}")
        
        # Test hazard detection
        self.test_endpoint_with_image("/hazzard_detect/detect", test_image)
        
        # Test sign detection
        self.test_endpoint_with_image("/sign_detect/detect", test_image)
        
        # Test SmolVLM analysis
        self.test_endpoint_with_image("/smolvlm/analyze", test_image)
        
        # Test SmolVLM with custom prompt
        self.test_endpoint_with_image("/smolvlm/analyze", test_image, 
                                     {"prompt": "What traffic signs and hazards do you see?"})
        
        # Test PaddleOCR
        self.test_endpoint_with_image("/paddleocr/extract", test_image)
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting AI Festival API Test Suite")
        print("=" * 50)
        
        # Test server health first
        if not self.test_server_health():
            print("\n❌ Server is not running! Please start the server first:")
            print("   cd app")
            print("   python main.py")
            return
        
        # Test model health
        self.test_model_health()
        
        # Test management endpoints
        self.test_model_management_endpoints()
        
        # Test detection endpoints
        self.test_detection_endpoints()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test']}: {result['details']}")
        
        print("\n🎯 Next Steps:")
        if failed_tests == 0:
            print("   All tests passed! Your API is working correctly.")
        else:
            print("   1. Check the failed tests above")
            print("   2. Ensure all models are loaded properly")
            print("   3. Verify test images are available")
            print("   4. Check server logs for detailed error messages")

def create_test_image_if_needed():
    """Create a simple test image if none exists"""
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save("test_image_generated.jpg")
        print("📸 Created test image: test_image_generated.jpg")
        return "test_image_generated.jpg"
    except ImportError:
        print("⚠️  Cannot create test image (PIL not available)")
        return None

if __name__ == "__main__":
    print("🧪 AI Festival API Tester")
    print("This script will test all your API endpoints")
    print()
    
    # Check if server is specified
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
        print(f"🌐 Testing server: {base_url}")
    else:
        base_url = BASE_URL
        print(f"🌐 Testing default server: {base_url}")
    
    # Create tester instance
    tester = APITester(base_url)
    
    # Run tests
    tester.run_all_tests()

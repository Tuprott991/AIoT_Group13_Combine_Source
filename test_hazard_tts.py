"""
Quick test for hazard detection with TTS
"""

import requests
import os

def test_hazard_detection():
    """Test hazard detection endpoint with TTS"""
    
    # Find test image
    test_images = ["app/test_hazzard_2.jpg"]
    test_image = None
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found!")
        print("Please make sure you have test images in the app/ directory")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    # Test server connection first
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not running!")
            print("Please start server: cd app && python main.py")
            return False
    except:
        print("❌ Server is not running!")
        print("Please start server: cd app && python main.py")
        return False
    
    print("✅ Server is running")
    
    # Test hazard detection
    try:
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        url = "http://127.0.0.1:8000/api/hazzard_detect/detect"
        headers = {"Content-Type": "application/octet-stream"}
        
        print("🔍 Testing hazard detection with TTS...")
        response = requests.post(url, data=image_data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Hazard detection successful!")
            print(f"📊 Response: {result}")
            print("🔊 Check if audio played on server!")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick Hazard Detection TTS Test")
    print("=" * 40)
    
    success = test_hazard_detection()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("💡 What should happen:")
        print("   1. Server receives image")
        print("   2. Detects hazards")
        print("   3. Plays audio warning on server")
        print("   4. Returns JSON to client")
    else:
        print("\n❌ Test failed!")
        print("💡 Troubleshooting:")
        print("   1. Make sure server is running")
        print("   2. Check if pygame is installed: pip install pygame")
        print("   3. Ensure test images exist in app/ directory")

"""
Test script for unified detection API with mode switching
"""

import requests
import os
import time

def test_unified_api():
    """Test unified detection API with different modes"""
    
    base_url = "http://127.0.0.1:8000/api/unified"
    test_image = "app/test_img.jpg"
    
    if not os.path.exists(test_image):
        print("❌ Test image not found!")
        return False
    
    # Test server connection
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running!")
        return False
    
    print("🚀 Testing Unified Detection API")
    print("=" * 50)
    
    # Test all modes
    modes = [
        (1, "Hazard Detection"),
        (2, "Sign Detection"),
        (3, "PaddleOCR"),
        (4, "SmolVLM")
    ]
    
    for mode_num, mode_name in modes:
        print(f"\n🔄 Testing Mode {mode_num}: {mode_name}")
        
        # Change mode
        try:
            mode_response = requests.post(f"{base_url}/mode", params={"mode": mode_num})
            if mode_response.status_code == 200:
                mode_data = mode_response.json()
                print(f"✅ Mode changed to: {mode_data['mode_name']}")
            else:
                print(f"❌ Failed to change mode: {mode_response.text}")
                continue
        except Exception as e:
            print(f"❌ Mode change failed: {e}")
            continue
        
        # Test detection
        try:
            with open(test_image, 'rb') as f:
                image_data = f.read()
            
            headers = {"Content-Type": "application/octet-stream"}
            detect_response = requests.post(f"{base_url}/detect", 
                                          data=image_data, 
                                          headers=headers, 
                                          timeout=30)
            
            if detect_response.status_code == 200:
                result = detect_response.json()
                print(f"✅ Detection successful!")
                print(f"📊 Mode: {result['mode']} ({result['mode_name']})")
                
                # Display results based on mode
                if mode_num in [1, 2]:  # Hazard or Sign detection
                    detections = result['result'].get('detections', [])
                    print(f"🔍 Found {len(detections)} detections")
                elif mode_num == 3:  # OCR
                    texts = result['result'].get('texts', [])
                    print(f"📝 Found {len(texts)} text items")
                elif mode_num == 4:  # SmolVLM
                    description = result['result'].get('description', '')
                    print(f"🖼️ Description: {description[:100]}...")
                
                print("🔊 Check server console for TTS audio!")
                
            else:
                print(f"❌ Detection failed: {detect_response.status_code}")
                print(f"Error: {detect_response.text}")
                
        except Exception as e:
            print(f"❌ Detection test failed: {e}")
        
        time.sleep(2)  # Wait between tests
    
    # Test getting current mode
    print(f"\n📋 Testing mode status...")
    try:
        status_response = requests.get(f"{base_url}/mode")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"✅ Current mode: {status['current_mode']} ({status['mode_name']})")
        else:
            print(f"❌ Failed to get mode status")
    except Exception as e:
        print(f"❌ Mode status failed: {e}")
    
    return True

def test_mode_switching():
    """Test just mode switching functionality"""
    
    base_url = "http://127.0.0.1:8000/api/unified"
    
    print("🔄 Testing Mode Switching Only")
    print("=" * 40)
    
    # Test invalid mode
    try:
        response = requests.post(f"{base_url}/mode", params={"mode": 5})
        if response.status_code == 400:
            print("✅ Invalid mode (5) correctly rejected")
        else:
            print(f"❌ Invalid mode should be rejected: {response.status_code}")
    except Exception as e:
        print(f"❌ Mode test failed: {e}")
    
    # Test valid modes
    for mode in [1, 2, 3, 4, 1]:  # Test switching back to mode 1
        try:
            response = requests.post(f"{base_url}/mode", params={"mode": mode})
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Mode {mode}: {data['mode_name']}")
            else:
                print(f"❌ Mode {mode} failed: {response.text}")
        except Exception as e:
            print(f"❌ Mode {mode} error: {e}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    print("🧪 Unified Detection API Test")
    print("=" * 60)
    
    # Test mode switching first
    test_mode_switching()
    
    print("\n" + "=" * 60)
    
    # Test full unified API
    success = test_unified_api()
    
    if success:
        print("\n🎉 Unified API Test Completed!")
        print("\n💡 How to use from ESP32:")
        print("1. POST /api/unified/mode?mode=1  # Change to hazard detection")
        print("2. POST /api/unified/detect       # Send image for detection")
        print("3. GET  /api/unified/mode         # Check current mode")
        print("\n🔊 TTS Audio will play on server based on mode:")
        print("   Mode 1: English hazard warnings")
        print("   Mode 2: Vietnamese sign positions") 
        print("   Mode 3: English text extraction")
        print("   Mode 4: English image descriptions")
    else:
        print("\n❌ Test failed!")
        print("💡 Make sure server is running: cd app && python main.py")

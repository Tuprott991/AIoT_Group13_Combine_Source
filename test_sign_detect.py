"""
Quick test for sign detection with TTS and confidence filter
"""

import requests
import os
import time

def test_sign_detection_tts():
    """Test sign detection endpoint focusing on TTS functionality"""
    
    test_image = "app/test_sign4.jpg"
    
    if not os.path.exists(test_image):
        print("❌ Test image not found!")
        print(f"Please make sure {test_image} exists")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    # Test server connection first
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running!")
        print("Please start server: cd app && python main.py")
        return False
    
    # Test sign detection with TTS
    print("\n🚦 Testing Sign Detection with TTS...")
    
    try:
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        url = "http://127.0.0.1:8000/api/sign_detect/detect"
        headers = {"Content-Type": "application/octet-stream"}
        
        print("🔍 Sending request to sign detection endpoint...")
        response = requests.post(url, data=image_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', [])
            
            print(f"✅ Sign detection successful!")
            print(f"📊 Total detections: {len(detections)}")
            
            # Analyze confidence levels for TTS
            high_confidence_signs = []
            all_detections = []
            
            for detection in detections:
                sign_name = detection.get('label', detection.get('sign', 'unknown'))
                confidence = detection.get('confidence', 0)
                position = detection.get('position', 'center')
                
                all_detections.append(f"{sign_name} ({confidence:.2f})")
                
                if confidence > 0.3:
                    high_confidence_signs.append(sign_name)
            
            print(f"\n📝 All detections: {', '.join(all_detections)}")
            
            # Remove duplicates from high confidence signs
            unique_high_confidence = list(dict.fromkeys(high_confidence_signs))
            print(f"🔊 TTS will announce: {len(unique_high_confidence)} unique signs")
            
            if unique_high_confidence:
                expected_audio = f"Detected {len(unique_high_confidence)} traffic signs: " + ", ".join(unique_high_confidence)
                print(f"🎵 Expected audio: '{expected_audio}'")
            else:
                print("🎵 Expected audio: 'No high confidence traffic signs detected'")
            
            print("\n⏳ Waiting for server TTS to play...")
            time.sleep(3)  # Give time for audio to play
            
            return True
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚦 Sign Detection TTS Test")
    print("=" * 40)
    
    success = test_sign_detection_tts()
    
    if success:
        print("\n🎉 Test completed!")
        print("\n💡 TTS Logic:")
        print("   1. Filter signs with confidence > 0.3")
        print("   2. Remove duplicate sign names")
        print("   3. Play audio on server only")
        print("   4. Return full JSON to client")
        print("\n🔊 Audio Examples:")
        print("   - High confidence: 'Detected 2 traffic signs: Stop Sign, Speed Limit'")
        print("   - Low confidence: 'No high confidence traffic signs detected'")
        print("   - No detections: 'No traffic signs detected'")
    else:
        print("\n❌ Test failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Start server: cd app && python main.py")
        print("   2. Check test image exists: app/test_sign.jpg") 
        print("   3. Verify gtts and pygame installed")
        print("   4. Check server console for errors")

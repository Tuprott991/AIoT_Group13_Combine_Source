"""
Test script for SmolVLM TTS functionality
"""

import requests
import os
import time

def test_smolvlm_tts():
    """Test SmolVLM endpoint with TTS functionality"""

    test_image = "app/test_hazzard.jpg"

    if not os.path.exists(test_image):
        print("❌ Test image not found!")
        print(f"Please make sure {test_image} exists")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    # Test server connection
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running!")
        print("Please start server: cd app && python main.py")
        return False
    
    # Test SmolVLM with TTS
    print("\n🖼️ Testing SmolVLM Image Analysis with TTS...")
    
    test_prompts = [
        "Describe what you see in this image.",
        "What objects are in this image?",
        "Describe the scene in detail.",
        "What is happening in this picture?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {prompt} ---")
        
        try:
            with open(test_image, 'rb') as f:
                image_data = f.read()
            
            url = "http://127.0.0.1:8000/api/smolvlm/analyze"
            headers = {"Content-Type": "application/octet-stream"}
            params = {"prompt": prompt}
            
            print("🔍 Sending request to SmolVLM endpoint...")
            response = requests.post(url, data=image_data, headers=headers, params=params, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ SmolVLM analysis successful!")
                print(f"📝 Prompt: {result.get('prompt', 'N/A')}")
                
                description = result.get('description', '')
                print(f"🖼️ Description: {description}")
                
                # Show TTS logic
                if description:
                    if len(description) > 200:
                        expected_audio = f"Image analysis: {description[:200]}... and more details"
                    else:
                        expected_audio = f"Image analysis: {description}"
                else:
                    expected_audio = "No description generated for this image"
                
                print(f"🎵 Expected TTS: '{expected_audio[:100]}...'")
                print("🔊 Listen for audio on server!")
                
                print("⏳ Waiting for server TTS to finish...")
                time.sleep(5)  # Give time for longer descriptions
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        
        time.sleep(2)  # Wait between tests
    
    return True

if __name__ == "__main__":
    print("🖼️ SmolVLM TTS Test")
    print("=" * 50)
    
    success = test_smolvlm_tts()
    
    if success:
        print("\n🎉 SmolVLM TTS Test Completed!")
        print("\n💡 TTS Logic:")
        print("   1. Analyze image with custom prompt")
        print("   2. Generate description using SmolVLM")
        print("   3. Truncate long descriptions (>200 chars)")
        print("   4. Play English audio on server")
        print("   5. Return JSON with full description to client")
        print("\n🔊 Audio Examples:")
        print("   - With description: 'Image analysis: A busy street with cars and people...'")
        print("   - Long description: 'Image analysis: [200 chars]... and more details'")
        print("   - No description: 'No description generated for this image'")
        print("   - Analysis failed: 'Unable to analyze image'")
    else:
        print("\n❌ Test failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Start server: cd app && python main.py")
        print("   2. Check test image exists: app/test_img.jpg") 
        print("   3. Verify SmolVLM model is loaded")
        print("   4. Check server console for errors")
        print("   5. SmolVLM may take longer to respond (60s timeout)")

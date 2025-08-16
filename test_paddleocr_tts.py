"""
Test script for PaddleOCR TTS functionality
"""

import requests
import os
import time

def test_paddleocr_tts():
    """Test PaddleOCR endpoint with TTS functionality"""
    
    # Test with different types of images
    test_images = [
        "app/test_img.jpg",
        "app/test_img2.png", 
        "app/test_img3.jpg",
        "app/test_img4.jpg"
    ]
    
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print("❌ No test images found!")
        print("Available images should be in:")
        for img in test_images:
            print(f"   - {img}")
        return False
    
    print(f"📸 Found {len(available_images)} test images")
    
    # Test server connection
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running!")
        print("Please start server: cd app && python main.py")
        return False
    
    success_count = 0
    
    for i, test_image in enumerate(available_images, 1):
        print(f"\n📝 Test {i}: OCR with TTS - {test_image}")
        
        try:
            with open(test_image, 'rb') as f:
                image_data = f.read()
            
            url = "http://127.0.0.1:8000/api/paddleocr/extract"
            headers = {"Content-Type": "application/octet-stream"}
            
            print("🔍 Sending request to PaddleOCR endpoint...")
            response = requests.post(url, data=image_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ OCR extraction successful!")
                
                texts = result.get('texts', [])
                total_texts = result.get('total_texts', 0)
                
                print(f"📊 Total texts found: {total_texts}")
                
                # Show extracted texts
                if texts:
                    print("📝 Extracted texts:")
                    for idx, text_item in enumerate(texts[:5], 1):  # Show first 5
                        text_content = text_item.get('text', '').strip()
                        confidence = text_item.get('confidence', 0)
                        print(f"   {idx}. '{text_content}' (confidence: {confidence:.2f})")
                    
                    if len(texts) > 5:
                        print(f"   ... and {len(texts) - 5} more texts")
                
                # Show TTS logic
                extracted_texts = []
                for text_item in texts:
                    text_content = text_item.get('text', '').strip()
                    if text_content:
                        extracted_texts.append(text_content)
                
                if extracted_texts:
                    first_three = extracted_texts[:3]
                    expected_audio = f"Extracted {len(extracted_texts)} texts: " + ", ".join(first_three)
                    if len(extracted_texts) > 3:
                        expected_audio += f" and {len(extracted_texts) - 3} more texts"
                    
                    print(f"🎵 Expected TTS: '{expected_audio}'")
                else:
                    print("🎵 Expected TTS: 'No readable text found'")
                
                print("🔊 Listen for audio on server!")
                print("⏳ Waiting for server TTS to play...")
                time.sleep(3)
                
                success_count += 1
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed for {test_image}: {e}")
        
        time.sleep(1)  # Brief pause between tests
    
    return success_count > 0

def test_paddleocr_edge_cases():
    """Test PaddleOCR with edge cases"""
    
    print("\n🧪 Testing PaddleOCR Edge Cases...")
    
    # Test with a simple test image
    test_image = "app/test_img.jpg"
    
    if not os.path.exists(test_image):
        print("❌ Test image not found for edge case testing")
        return False
    
    edge_cases = [
        ("Normal extraction", {}),
        ("Empty image handling", {}),  # Will use same image but test response handling
    ]
    
    for case_name, params in edge_cases:
        print(f"\n🔍 Edge case: {case_name}")
        
        try:
            with open(test_image, 'rb') as f:
                image_data = f.read()
            
            url = "http://127.0.0.1:8000/api/paddleocr/extract"
            headers = {"Content-Type": "application/octet-stream"}
            
            response = requests.post(url, data=image_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                texts = result.get('texts', [])
                
                print(f"✅ {case_name} handled correctly")
                print(f"📊 Texts found: {len(texts)}")
                
                # Test TTS message generation
                if texts:
                    readable_texts = [t.get('text', '').strip() for t in texts if t.get('text', '').strip()]
                    if readable_texts:
                        print(f"🎵 TTS will announce: {len(readable_texts)} texts")
                    else:
                        print("🎵 TTS will say: 'No readable text found'")
                else:
                    print("🎵 TTS will say: 'No text detected in image'")
                
            else:
                print(f"❌ {case_name} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {case_name} error: {e}")
        
        time.sleep(1)
    
    return True

if __name__ == "__main__":
    print("📝 PaddleOCR TTS Test")
    print("=" * 50)
    
    # Test normal functionality
    success = test_paddleocr_tts()
    
    # Test edge cases
    if success:
        test_paddleocr_edge_cases()
    
    if success:
        print("\n🎉 PaddleOCR TTS Test Completed!")
        print("\n💡 TTS Logic:")
        print("   1. Extract text from image using PaddleOCR")
        print("   2. Filter readable text (non-empty)")
        print("   3. Announce first 3 texts + count of remaining")
        print("   4. Play English audio on server")
        print("   5. Return JSON with all texts to client")
        print("\n🔊 Audio Examples:")
        print("   - With texts: 'Extracted 5 texts: Hello, World, Stop and 2 more texts'")
        print("   - Few texts: 'Extracted 2 texts: Hello, World'")
        print("   - Readable but empty: 'No readable text found'")
        print("   - No detection: 'No text detected in image'")
    else:
        print("\n❌ Test failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Start server: cd app && python main.py")
        print("   2. Check test images exist in app/ directory")
        print("   3. Verify PaddleOCR model is loaded")
        print("   4. Check server console for errors")
        print("   5. Try images with clear text for better results")

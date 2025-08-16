#!/usr/bin/env python3
"""
Simple PaddleOCR test to debug the issue
"""

import os
import sys

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_paddleocr_only():
    """Test only PaddleOCR initialization"""
    print("Testing PaddleOCR initialization...")
    
    try:
        # Test basic PaddleOCR import
        print("1. Testing PaddleOCR import...")
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR import successful")
        
        # Test our wrapper class
        print("2. Testing our PaddleOCR wrapper...")
        from utils.infer_paddleocr import PaddleOCRInference
        print("✅ Wrapper import successful")
        
        # Test initialization
        print("3. Testing PaddleOCR initialization...")
        ocr = PaddleOCRInference()
        print("✅ PaddleOCR initialized successfully!")
        
        print(f"Device: {ocr.device}")
        print(f"Using GPU: {ocr.use_gpu}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_paddleocr_only()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)

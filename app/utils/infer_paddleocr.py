"""
PaddleOCR Inference Module for PyTorch Integration
Handles OCR text detection and recognition with GPU/CPU detection
"""

# Import PyTorch first to avoid CUDA conflicts
import torch

# Import order fix for PyTorch-PaddlePaddle conflict
try:
    from .import_fix import get_safe_paddle_ocr
    IMPORT_FIX_AVAILABLE = True
except ImportError:
    IMPORT_FIX_AVAILABLE = False

import cv2
import numpy as np
from PIL import Image
import os
import logging

# Safe PaddleOCR import
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"PaddleOCR import failed: {e}")
    PADDLEOCR_AVAILABLE = False

# Suppress PaddleOCR logging
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("ppdet").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

# Set environment variables to reduce paddle logging
os.environ['FLAGS_logtostderr'] = '0'
os.environ['FLAGS_stderrthreshold'] = '3'

class PaddleOCRInference:
    def __init__(self, use_gpu=None):
        """
        Initialize PaddleOCR model with conflict resolution
        
        Args:
            use_gpu (bool): Whether to use GPU. Note: PaddleOCR may use CPU to avoid conflicts
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not available. Please install paddleocr.")
            
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        
        # For conflict resolution, prefer CPU for PaddleOCR
        # This avoids PyTorch-PaddlePaddle CUDA conflicts
        self.use_gpu = False  # Force CPU to avoid conflicts
        self.device = 'cpu'
        
        print(f"Initializing PaddleOCR (CPU mode to avoid PyTorch conflicts)")
        
        try:
            # Use import fix if available
            if IMPORT_FIX_AVAILABLE:
                self.ocr = get_safe_paddle_ocr()
                if self.ocr is None:
                    raise RuntimeError("PaddleOCR initialization failed via import fix")
                print("✅ PaddleOCR initialized via import fix")
            else:
                # Safe basic initialization
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en'
                )
                print("✅ PaddleOCR initialized with basic settings")
                
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load PaddleOCR: {e}")
            print("💡 Run fix_pytorch_paddle_conflict.py to resolve conflicts")
            self.ocr = None
            self.model_loaded = False
            raise
    
    def extract_text(self, image):
        """
        Extract text from image using PaddleOCR
        
        Args:
            image: OpenCV image (BGR format) or PIL Image or path string
            
        Returns:
            dict: OCR results with detected text
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # If it's a file path
                if not os.path.exists(image):
                    raise ValueError(f"Image file not found: {image}")
                image = cv2.imread(image)
            elif isinstance(image, Image.Image):
                # If it's a PIL Image, convert to OpenCV
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                # Already an OpenCV image
                pass
            else:
                raise ValueError("Invalid image format")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Run OCR
            results = self.ocr.ocr(image, cls=True)
            
            # Process results
            extracted_texts = []
            total_texts = 0
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        # line[0] contains the bounding box coordinates
                        # line[1] contains (text, confidence)
                        bbox = line[0]
                        text_info = line[1]
                        
                        if text_info and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = float(text_info[1])
                            
                            # Convert bbox to the expected format
                            formatted_bbox = [
                                [int(bbox[0][0]), int(bbox[0][1])],  # top-left
                                [int(bbox[1][0]), int(bbox[1][1])],  # top-right
                                [int(bbox[2][0]), int(bbox[2][1])],  # bottom-right
                                [int(bbox[3][0]), int(bbox[3][1])]   # bottom-left
                            ]
                            
                            extracted_texts.append({
                                "text": text,
                                "confidence": confidence,
                                "bbox": formatted_bbox
                            })
                            total_texts += 1
            
            return {
                "texts": extracted_texts,
                "total_texts": total_texts,
                "device_used": self.device
            }
            
        except Exception as e:
            raise ValueError(f"Error extracting text: {str(e)}")
    
    def get_text_only(self, image):
        """
        Extract only the text strings without bounding boxes
        
        Args:
            image: OpenCV image or image path
            
        Returns:
            list: List of detected text strings
        """
        result = self.extract_text(image)
        return [item["text"] for item in result["texts"]]
    
    def get_text_with_confidence(self, image, confidence_threshold=0.5):
        """
        Extract text with confidence filtering
        
        Args:
            image: OpenCV image or image path
            confidence_threshold (float): Minimum confidence score
            
        Returns:
            dict: Filtered OCR results
        """
        result = self.extract_text(image)
        filtered_texts = [
            item for item in result["texts"] 
            if item["confidence"] >= confidence_threshold
        ]
        
        return {
            "texts": filtered_texts,
            "total_texts": len(filtered_texts),
            "device_used": result["device_used"]
        }

def infer_image_ocr(image_path):
    """
    Convenience function to extract text from a single image
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: OCR results
    """
    ocr = PaddleOCRInference()
    return ocr.extract_text(image_path)

def infer_image_with_preloaded_model(image):
    """
    Inference function using pre-loaded model from model_loader
    
    Args:
        image: OpenCV image array or PIL Image
        
    Returns:
        dict: OCR results
    """
    from model_loader import get_model
    
    # Get pre-loaded model
    ocr = get_model("paddleocr")
    if ocr is None:
        raise RuntimeError("PaddleOCR model not loaded. Please ensure models are loaded at startup.")
    
    # Run OCR
    return ocr.extract_text(image)

# Test function
if __name__ == "__main__":
    # Example usage
    ocr = PaddleOCRInference()
    
    # Test with an image if available
    test_images = ["../test_img.jpg", "../test_img2.png", "../test_img3.jpg"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting OCR with: {img_path}")
            try:
                results = ocr.extract_text(img_path)
                print(f"Found {results['total_texts']} text regions:")
                for i, text_info in enumerate(results['texts']):
                    print(f"  {i+1}: '{text_info['text']}' (confidence: {text_info['confidence']:.3f})")
            except Exception as e:
                print(f"Error: {e}")
            break
    else:
        print("No test images found for OCR testing")
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from PIL import Image
import io

async def extract_text(image_data: bytes):
    """
    Extract text from image using PaddleOCR PyTorch implementation
    
    Args:
        image_data (bytes): Raw image bytes
        
    Returns:
        dict: OCR results with detected text
    """
    try:
        # Import the inference function
        from utils.infer_paddleocr import infer_image_with_preloaded_model
        
        # Convert bytes to OpenCV image
        np_image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")
        
        # Run OCR using the preloaded model
        result = infer_image_with_preloaded_model(image)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error extracting text: {str(e)}")

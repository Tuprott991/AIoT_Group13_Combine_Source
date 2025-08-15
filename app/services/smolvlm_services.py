from dotenv import load_dotenv
import os
from utils.infer_smolvlm import infer_image_with_preloaded_model
import cv2
import numpy as np
from PIL import Image
import io

async def analyze_image(image_data: bytes, prompt: str = "Describe what you see in this image."):
    """
    Analyze image using SmolVLM model
    
    Args:
        image_data (bytes): Raw image bytes
        prompt (str): Text prompt for the model
        
    Returns:
        dict: Analysis result with description
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Use pre-loaded model for inference
        description = infer_image_with_preloaded_model(image, prompt)
        
        return {
            "description": description,
            "prompt": prompt
        }
        
    except Exception as e:
        raise ValueError(f"Error analyzing image: {str(e)}")

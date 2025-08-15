from dotenv import load_dotenv
import os
from utils.infer_yolic import infer_image_with_preloaded_model
import cv2
import numpy as np

async def detect_hazard(image_data: bytes):
    # Convert the image data to a numpy array
    np_image = np.frombuffer(image_data, np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")
    
    # Perform inference using the pre-loaded model
    results = infer_image_with_preloaded_model(image)

    detections = []
    # Process the results and extract relevant information
    for result in results:
        detections.append({
            "label": result["object"],
            "confidence": float(result["confidence"]),
            "position": str(result["position"]),
            "class_index": int(result["class_index"])
        })

    return detections

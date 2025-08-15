from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.sign_detection_services import detect_sign
import logging

router = APIRouter()

class SignDetectionResponse(BaseModel):
    detections: list[dict]

@router.post("/detect", response_model=SignDetectionResponse, tags=["sign_detection"])
async def detect_sign_endpoint(request: Request):
    try:
        # Read raw binary data from request body
        image_data = await request.body()
        
        # Validate that we have data
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        result = await detect_sign(image_data)
        print(result)
        return SignDetectionResponse(detections=result)
    except Exception as e:
        logging.error(f"Error detecting sign: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

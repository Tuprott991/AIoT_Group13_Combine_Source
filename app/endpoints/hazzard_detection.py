from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.hazzard_detection_services import detect_hazard
import logging

router = APIRouter()

class HazardDetectionResponse(BaseModel):
    detections: list[dict]

@router.post("/detect", response_model=HazardDetectionResponse, tags=["hazard_detection"])
async def detect_hazard_endpoint(request: Request):
    try:
        # Read raw binary data from request body
        image_data = await request.body()
        
        # Validate that we have data
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        result = await detect_hazard(image_data)
        return HazardDetectionResponse(detections=result)
    except Exception as e:
        logging.error(f"Error detecting hazard: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        

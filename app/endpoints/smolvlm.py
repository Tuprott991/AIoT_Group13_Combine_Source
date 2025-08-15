from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from services.smolvlm_services import analyze_image
import logging

router = APIRouter()

class SmolVLMResponse(BaseModel):
    description: str
    prompt: str

@router.post("/analyze", response_model=SmolVLMResponse, tags=["smolvlm"])
async def analyze_image_endpoint(request: Request, prompt: str = Query(default="Describe what you see in this image.")):
    try:
        # Read raw binary data from request body
        image_data = await request.body()
        
        # Validate that we have data
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        result = await analyze_image(image_data, prompt)
        print(result)
        return SmolVLMResponse(**result)
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

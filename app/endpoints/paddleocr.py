from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.paddleocr_services import extract_text
import logging

router = APIRouter()

class OCRResponse(BaseModel):
    texts: list[dict]
    total_texts: int

@router.post("/extract", response_model=OCRResponse, tags=["paddleocr"])
async def extract_text_endpoint(request: Request):
    try:
        # Read raw binary data from request body
        image_data = await request.body()
        
        # Validate that we have data
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        result = await extract_text(image_data)
        return OCRResponse(**result)
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

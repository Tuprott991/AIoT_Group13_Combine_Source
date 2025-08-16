from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.paddleocr_services import extract_text
import logging
from gtts import gTTS
import os
import pygame
import time
import tempfile

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
        
        # Create text for TTS and play audio on server
        try:
            if result and result.get('texts') and len(result['texts']) > 0:
                extracted_texts = []
                for text_item in result['texts']:
                    text_content = text_item.get('text', '').strip()
                    if text_content:
                        extracted_texts.append(text_content)
                
                if extracted_texts:
                    text = f"Extracted {len(extracted_texts)} texts: " + ", ".join(extracted_texts[:3])  # Limit to first 3 texts
                    if len(extracted_texts) > 3:
                        text += f" and {len(extracted_texts) - 3} more texts"
                else:
                    text = "No readable text found"
            else:
                text = "No text detected in image"
            
            # Generate TTS audio
            tts = gTTS(text=text, lang='en')
            
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            
            # Play audio on server only
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing (with timeout)
            timeout = 10  # Maximum 10 seconds
            start_time = time.time()
            while pygame.mixer.music.get_busy() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            print(f"✅ Audio played on server: {text}")
            
            # Clean up temporary file
            os.unlink(temp_file.name)
            
        except Exception as audio_error:
            logging.warning(f"Could not play audio on server: {audio_error}")
        
        # Return normal JSON response to device
        return OCRResponse(**result)
        
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

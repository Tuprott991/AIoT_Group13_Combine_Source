from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from services.smolvlm_services import analyze_image
import logging
from gtts import gTTS
import os
import pygame
import time
import tempfile

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
        
        # Create text for TTS and play audio on server
        try:
            if result and result.get('description'):
                description = result['description'].strip()
                if description:
                    # Limit description length for TTS (first 200 characters)
                    if len(description) > 200:
                        text = f"Image analysis: {description[:200]}... and more details"
                    else:
                        text = f"Image analysis: {description}"
                else:
                    text = "No description generated for this image"
            else:
                text = "Unable to analyze image"
            
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
            timeout = 15  # Maximum 15 seconds for longer descriptions
            start_time = time.time()
            while pygame.mixer.music.get_busy() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            print(f"✅ Audio played on server: {text}")
            
            # Clean up temporary file
            os.unlink(temp_file.name)
            
        except Exception as audio_error:
            logging.warning(f"Could not play audio on server: {audio_error}")
        
        # Return normal JSON response to device
        return SmolVLMResponse(**result)
        
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

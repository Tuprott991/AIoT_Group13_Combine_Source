from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.sign_detection_services import detect_sign
import logging
from gtts import gTTS
import os

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
        
        # Create text for TTS and play audio on server
        try:
            if result and len(result) > 0:
                # Group by position with confidence > 0.3
                position_signs = {
                    'left': set(), 
                    'right': set(), 
                    'center': set()
                }
                
                for detection in result:
                    confidence = float(detection.get('confidence', 0))
                    if confidence > 0.3:  # Only include high confidence detections
                        sign_name = detection.get('sign', detection.get('label', 'biển báo'))
                        position = detection.get('position', 'center')
                        
                        if position in position_signs:
                            position_signs[position].add(sign_name)
                
                # Build Vietnamese audio message by position
                audio_parts = []
                
                # Left position
                if position_signs['left']:
                    left_signs = ', '.join(list(position_signs['left']))
                    audio_parts.append(f"Bên trái có {left_signs}")
                
                # Center position  
                if position_signs['center']:
                    center_signs = ', '.join(list(position_signs['center']))
                    audio_parts.append(f"ở giữa có {center_signs}")
                
                # Right position
                if position_signs['right']:
                    right_signs = ', '.join(list(position_signs['right']))
                    audio_parts.append(f"bên phải có {right_signs}")
                
                if audio_parts:
                    text = "; ".join(audio_parts)
                else:
                    text = "Không phát hiện biển báo có độ tin cậy cao"
            else:
                text = "Không phát hiện biển báo"
            
            # Generate TTS audio in Vietnamese
            tts = gTTS(text=text, lang='vi')
            
            # Create temporary file for audio
            tts.save("sign_detection.mp3")

            os.system(f"start sign_detection.mp3")

            print(f"✅ Audio played on server: {text}")
            
        except Exception as audio_error:
            logging.warning(f"Could not play audio on server: {audio_error}")
        
        # Return normal JSON response to device
        return SignDetectionResponse(detections=result)
        
    except Exception as e:
        logging.error(f"Error detecting sign: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

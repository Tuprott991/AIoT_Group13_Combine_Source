from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.hazzard_detection_services import detect_hazard
import logging
from gtts import gTTS
import os

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
        
        # Create text for TTS and play audio on server
        try:
            if result and len(result) > 0:
                # Group by position, then count each object type with confidence > 0.7
                position_object_counts = {
                    'left': {}, 
                    'right': {}, 
                    'center': {}
                }
                
                for detection in result:
                    confidence = float(detection.get('confidence', 0))
                    if confidence > 0.3:  # Only include high confidence detections
                        hazard_name = detection.get('object', detection.get('label', 'hazard'))
                        position = detection.get('position', 'center')
                        
                        if position in position_object_counts:
                            # Count each object type in each position
                            if hazard_name not in position_object_counts[position]:
                                position_object_counts[position][hazard_name] = 0
                            position_object_counts[position][hazard_name] += 1
                
                # Check if any object type has more than 3 occurrences in same position
                warnings = []
                for position, object_counts in position_object_counts.items():
                    for object_name, count in object_counts.items():
                        if count > 3:  # More than 3 same objects in same position
                            warnings.append(f"{object_name}s on {position}")
                
                if warnings:
                    text = f"Warning! " + "; ".join(warnings)
                else:
                    # Check if we have any detections at all
                    total_detections = sum(sum(counts.values()) for counts in position_object_counts.values())
                    if total_detections > 0:
                        text = f"Normal traffic detected: {total_detections} objects with good confidence."
                    else:
                        text = "No high confidence hazards detected. Road is clear."
            else:
                text = "No hazards detected. Road is clear."
            
            # Generate TTS audio
            tts = gTTS(text=text, lang='en')
            
            # Create temporary file for audio
            tts.save("hazard_detection.mp3")

            os.system(f"start hazard_detection.mp3")

            print(f"✅ Audio played on server: {text}")
            
        except Exception as audio_error:
            logging.warning(f"Could not play audio on server: {audio_error}")
        
        # Return normal JSON response to device
        return HazardDetectionResponse(detections=result)
        
    except Exception as e:
        logging.error(f"Error detecting hazard: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        

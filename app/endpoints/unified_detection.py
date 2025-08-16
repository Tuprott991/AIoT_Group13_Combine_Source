from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
import mode_controller
import logging

# Import all services
from services.hazzard_detection_services import detect_hazard
from services.sign_detection_services import detect_sign
from services.paddleocr_services import extract_text
from services.smolvlm_services import analyze_image

# Import TTS functionality
from gtts import gTTS
import os
import pygame
import time
import tempfile

router = APIRouter()

# Response models
class UnifiedResponse(BaseModel):
    mode: int
    mode_name: str
    result: dict

class ModeChangeResponse(BaseModel):
    previous_mode: int
    current_mode: int
    mode_name: str
    message: str

# Mode control endpoint
@router.post("/mode", response_model=ModeChangeResponse, tags=["mode_control"])
async def change_mode(mode: int):
    """Change detection mode (1=hazard, 2=sign, 3=paddleocr, 4=smolvlm)"""
    try:
        previous_mode = mode_controller.get_current_mode()
        
        if mode_controller.set_current_mode(mode):
            current_mode = mode_controller.get_current_mode()
            mode_name = mode_controller.get_mode_name(current_mode)
            
            print(f"🔄 Mode changed from {previous_mode} to {current_mode} ({mode_name})")
            
            return ModeChangeResponse(
                previous_mode=previous_mode,
                current_mode=current_mode,
                mode_name=mode_name,
                message=f"Successfully changed to {mode_name}"
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 1,2,3,4")
            
    except Exception as e:
        logging.error(f"Error changing mode: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Unified detection endpoint
@router.post("/detect", response_model=UnifiedResponse, tags=["unified_detection"])
async def unified_detect(request: Request, prompt: str = Query(default="Describe what you see in this image.")):
    """Unified detection endpoint - mode determined by global variable"""
    try:
        # Get current mode
        current_mode = mode_controller.get_current_mode()
        mode_name = mode_controller.get_mode_name(current_mode)
        
        print(f"🔍 Processing request in mode {current_mode}: {mode_name}")
        
        # Read image data
        image_data = await request.body()
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Route to appropriate service based on mode
        if current_mode == 1:
            # Hazard Detection
            result = await detect_hazard(image_data)
            await _play_hazard_tts(result)
            response_data = {"detections": result}
            
        elif current_mode == 2:
            # Sign Detection
            result = await detect_sign(image_data)
            await _play_sign_tts(result)
            response_data = {"detections": result}
            
        elif current_mode == 3:
            # PaddleOCR
            result = await extract_text(image_data)
            await _play_ocr_tts(result)
            response_data = result
            
        elif current_mode == 4:
            # SmolVLM
            result = await analyze_image(image_data, prompt)
            await _play_smolvlm_tts(result)
            response_data = result
            
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")
        
        return UnifiedResponse(
            mode=current_mode,
            mode_name=mode_name,
            result=response_data
        )
        
    except Exception as e:
        logging.error(f"Error in unified detection: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# TTS functions for each mode
async def _play_hazard_tts(result):
    """Play TTS for hazard detection results"""
    try:
        if result and len(result) > 0:
            position_object_counts = {'left': {}, 'right': {}, 'center': {}}
            
            for detection in result:
                confidence = float(detection.get('confidence', 0))
                if confidence > 0.3:
                    hazard_name = detection.get('object', detection.get('label', 'hazard'))
                    position = detection.get('position', 'center')
                    
                    if position in position_object_counts:
                        if hazard_name not in position_object_counts[position]:
                            position_object_counts[position][hazard_name] = 0
                        position_object_counts[position][hazard_name] += 1
            
            warnings = []
            for position, object_counts in position_object_counts.items():
                for object_name, count in object_counts.items():
                    if count > 3:
                        warnings.append(f"{object_name}s on {position}")
            
            if warnings:
                text = f"Warning! " + "; ".join(warnings)
            else:
                total_detections = sum(sum(counts.values()) for counts in position_object_counts.values())
                if total_detections > 0:
                    text = f"Normal traffic detected: {total_detections} objects with good confidence."
                else:
                    text = "No high confidence hazards detected. Road is clear."
        else:
            text = "No hazards detected. Road is clear."
            
        await _play_audio(text, 'en')
    except Exception as e:
        logging.warning(f"Could not play hazard TTS: {e}")

async def _play_sign_tts(result):
    """Play TTS for sign detection results"""
    try:
        if result and len(result) > 0:
            position_signs = {'left': set(), 'right': set(), 'center': set()}
            
            for detection in result:
                confidence = float(detection.get('confidence', 0))
                if confidence > 0.3:
                    sign_name = detection.get('sign', detection.get('label', 'biển báo'))
                    position = detection.get('position', 'center')
                    
                    if position in position_signs:
                        position_signs[position].add(sign_name)
            
            audio_parts = []
            if position_signs['left']:
                left_signs = ', '.join(list(position_signs['left']))
                audio_parts.append(f"Bên trái có {left_signs}")
            if position_signs['center']:
                center_signs = ', '.join(list(position_signs['center']))
                audio_parts.append(f"ở giữa có {center_signs}")
            if position_signs['right']:
                right_signs = ', '.join(list(position_signs['right']))
                audio_parts.append(f"bên phải có {right_signs}")
            
            if audio_parts:
                text = "; ".join(audio_parts)
            else:
                text = "Không phát hiện biển báo có độ tin cậy cao"
        else:
            text = "Không phát hiện biển báo"
            
        await _play_audio(text, 'vi')
    except Exception as e:
        logging.warning(f"Could not play sign TTS: {e}")

async def _play_ocr_tts(result):
    """Play TTS for OCR results"""
    try:
        if result and result.get('texts') and len(result['texts']) > 0:
            extracted_texts = []
            for text_item in result['texts']:
                text_content = text_item.get('text', '').strip()
                if text_content:
                    extracted_texts.append(text_content)
            
            if extracted_texts:
                text = f"Extracted {len(extracted_texts)} texts: " + ", ".join(extracted_texts[:3])
                if len(extracted_texts) > 3:
                    text += f" and {len(extracted_texts) - 3} more texts"
            else:
                text = "No readable text found"
        else:
            text = "No text detected in image"
            
        await _play_audio(text, 'en')
    except Exception as e:
        logging.warning(f"Could not play OCR TTS: {e}")

async def _play_smolvlm_tts(result):
    """Play TTS for SmolVLM results"""
    try:
        if result and result.get('description'):
            description = result['description'].strip()
            if description:
                if len(description) > 200:
                    text = f"Image analysis: {description[:200]}... and more details"
                else:
                    text = f"Image analysis: {description}"
            else:
                text = "No description generated for this image"
        else:
            text = "Unable to analyze image"
            
        await _play_audio(text, 'en')
    except Exception as e:
        logging.warning(f"Could not play SmolVLM TTS: {e}")

async def _play_audio(text: str, lang: str):
    """Play TTS audio on server"""
    try:
        tts = gTTS(text=text, lang=lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        
        timeout = 15
        start_time = time.time()
        while pygame.mixer.music.get_busy() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        print(f"✅ Audio played: {text}")
        os.unlink(temp_file.name)
        
    except Exception as e:
        logging.warning(f"Could not play audio: {e}")

# Get current mode endpoint
@router.get("/mode", tags=["mode_control"])
async def get_current_mode():
    """Get current detection mode"""
    current_mode = mode_controller.get_current_mode()
    mode_name = mode_controller.get_mode_name(current_mode)
    
    return {
        "current_mode": current_mode,
        "mode_name": mode_name
    }

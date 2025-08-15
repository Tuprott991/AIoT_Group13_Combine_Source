"""
Model Management Endpoints
Provides endpoints for managing and monitoring AI models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model_loader import (
    get_loaded_models, 
    models_health_check, 
    reload_model, 
    is_model_loaded
)
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ModelReloadRequest(BaseModel):
    model_name: str

class ModelReloadResponse(BaseModel):
    success: bool
    message: str
    model_name: str

@router.get("/status", tags=["model_management"])
async def get_model_status():
    """Get status of all loaded models"""
    try:
        return models_health_check()
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model status")

@router.get("/list", tags=["model_management"])
async def list_loaded_models():
    """Get list of loaded models"""
    try:
        return {
            "loaded_models": get_loaded_models(),
            "available_models": ["yolo_sign_detection", "yolic_hazard_detection", "smolvlm"]
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Error listing models")

@router.post("/reload", response_model=ModelReloadResponse, tags=["model_management"])
async def reload_model_endpoint(request: ModelReloadRequest):
    """Reload a specific model"""
    try:
        if request.model_name not in ["yolo_sign_detection", "yolic_hazard_detection", "smolvlm"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model name. Available models: yolo_sign_detection, yolic_hazard_detection, smolvlm"
            )
        
        success = reload_model(request.model_name)
        
        if success:
            return ModelReloadResponse(
                success=True,
                message=f"Model {request.model_name} reloaded successfully",
                model_name=request.model_name
            )
        else:
            return ModelReloadResponse(
                success=False,
                message=f"Failed to reload model {request.model_name}",
                model_name=request.model_name
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

@router.get("/check/{model_name}", tags=["model_management"])
async def check_model_loaded(model_name: str):
    """Check if a specific model is loaded"""
    try:
        if model_name not in ["yolo_sign_detection", "yolic_hazard_detection", "smolvlm"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model name. Available models: yolo_sign_detection, yolic_hazard_detection, smolvlm"
            )
        
        is_loaded = is_model_loaded(model_name)
        
        return {
            "model_name": model_name,
            "is_loaded": is_loaded,
            "status": "loaded" if is_loaded else "not_loaded"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking model: {str(e)}")

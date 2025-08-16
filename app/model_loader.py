"""
Global Model Loader for AI Festival Project
This module pre-loads all AI models when the server starts to improve performance
"""

import os
import logging
from typing import Dict, Any, Optional
from utils.infer_smolvlm import SmolVLMInference
from utils.infer_yolo import YOLOv8SignDetection
from utils.infer_yolic import YOLICOpenVINOInference

# Global variables to store loaded models
_models: Dict[str, Any] = {}
_model_paths = {
    "yolo_sign_detection": "/kaggle/input/system_model/pytorch/default/1/ai_models/yolov8_sign_detection/yolov8_sign_detection.xml",
    "yolic_hazard_detection": "/kaggle/input/system_model/pytorch/default/1/ai_models/yolic_m2/yolic_m2.xml",
    "smolvlm": "echarlaix/SmolVLM-256M-Instruct-openvino"
}

logger = logging.getLogger(__name__)

def load_all_models() -> None:
    """
    Load all AI models at server startup
    This function should be called once when the FastAPI application starts
    """
    logger.info("Starting to load all AI models...")
    
    try:
        # Load YOLO Sign Detection Model
        logger.info("Loading YOLO Sign Detection model...")
        if os.path.exists(_model_paths["yolo_sign_detection"]):
            _models["yolo_sign_detection"] = YOLOv8SignDetection(
                model_path=_model_paths["yolo_sign_detection"]
            )
            logger.info("✅ YOLO Sign Detection model loaded successfully")
        else:
            logger.warning(f"❌ YOLO model file not found: {_model_paths['yolo_sign_detection']}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO Sign Detection model: {e}")
    
    try:
        # Load YOLIC Hazard Detection Model
        logger.info("Loading YOLIC Hazard Detection model...")
        if os.path.exists(_model_paths["yolic_hazard_detection"]):
            _models["yolic_hazard_detection"] = YOLICOpenVINOInference(
                model_path=_model_paths["yolic_hazard_detection"]
            )
            logger.info("✅ YOLIC Hazard Detection model loaded successfully")
        else:
            logger.warning(f"❌ YOLIC model file not found: {_model_paths['yolic_hazard_detection']}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load YOLIC Hazard Detection model: {e}")
    
    try:
        # Load SmolVLM Model
        logger.info("Loading SmolVLM model...")
        _models["smolvlm"] = SmolVLMInference(model_id=_model_paths["smolvlm"])
        logger.info("✅ SmolVLM model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load SmolVLM model: {e}")
    
    logger.info(f"Model loading completed. Loaded {len(_models)} models successfully.")
    

def get_model(model_name: str) -> Optional[Any]:
    """
    Get a pre-loaded model by name
    
    Args:
        model_name (str): Name of the model ('yolo_sign_detection', 'yolic_hazard_detection', 'smolvlm')
        
    Returns:
        The loaded model instance or None if not found
    """
    model = _models.get(model_name)
    if model is None:
        logger.warning(f"Model '{model_name}' not found in loaded models. Available models: {list(_models.keys())}")
    return model


def is_model_loaded(model_name: str) -> bool:
    """
    Check if a specific model is loaded
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model is loaded, False otherwise
    """
    return model_name in _models


def get_loaded_models() -> Dict[str, str]:
    """
    Get information about all loaded models
    
    Returns:
        Dict mapping model names to their status
    """
    return {name: type(model).__name__ for name, model in _models.items()}


def reload_model(model_name: str) -> bool:
    """
    Reload a specific model
    
    Args:
        model_name (str): Name of the model to reload
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Reloading model: {model_name}")
        
        if model_name == "yolo_sign_detection":
            _models[model_name] = YOLOv8SignDetection(
                model_path=_model_paths[model_name]
            )
        elif model_name == "yolic_hazard_detection":
            _models[model_name] = YOLICOpenVINOInference(
                model_path=_model_paths[model_name]
            )
        elif model_name == "smolvlm":
            _models[model_name] = SmolVLMInference(
                model_id=_model_paths[model_name]
            )
        else:
            logger.error(f"Unknown model name: {model_name}")
            return False
            
        logger.info(f"✅ Model {model_name} reloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reload model {model_name}: {e}")
        return False


# Health check function
def models_health_check() -> Dict[str, Any]:
    """
    Perform health check on all loaded models
    
    Returns:
        Dict with health status of all models
    """
    health_status = {
        "total_models": len(_models),
        "loaded_models": list(_models.keys()),
        "model_details": {}
    }
    
    for name, model in _models.items():
        try:
            # Basic check - if model exists and has expected attributes
            health_status["model_details"][name] = {
                "status": "healthy",
                "type": type(model).__name__,
                "loaded": True
            }
        except Exception as e:
            health_status["model_details"][name] = {
                "status": "unhealthy",
                "error": str(e),
                "loaded": False
            }
    
    return health_status

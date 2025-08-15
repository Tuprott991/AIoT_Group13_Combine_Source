from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router as api_router
from model_loader import load_all_models, models_health_check
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Festival API",
    description="API for hazard detection, sign detection, and image analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all AI models at startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting AI Festival API server...")
    logger.info("📦 Loading AI models...")
    try:
        load_all_models()
        health = models_health_check()
        logger.info(f"✅ Model loading completed: {health['total_models']} models loaded")
        logger.info(f"📋 Loaded models: {health['loaded_models']}")
    except Exception as e:
        logger.error(f"❌ Error during model loading: {e}")
        # Continue starting the server even if some models fail to load
    logger.info("🎯 API server ready!")

# Health check endpoint for models
@app.get("/health/models")
async def model_health():
    """Get health status of all loaded models"""
    return models_health_check()

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)  
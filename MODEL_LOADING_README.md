# AI Festival - Model Pre-Loading System

## Overview

This system pre-loads all AI models when the FastAPI server starts, improving performance by eliminating model loading time during inference requests.

## Features

- **Pre-loaded Models**: All models are loaded once at server startup
- **Performance Optimization**: No model loading delay during API requests  
- **Model Management**: Endpoints to monitor and manage loaded models
- **Health Checks**: Monitor model status and health
- **Error Resilience**: Server continues to run even if some models fail to load

## Loaded Models

The system loads the following models at startup:

1. **YOLO Sign Detection** (`yolo_sign_detection`)
   - Path: `ai_models/yolov8_sign_detection/yolov8_sign_detection.xml`
   - Purpose: Traffic sign detection

2. **YOLIC Hazard Detection** (`yolic_hazard_detection`)
   - Path: `ai_models/yolic_m2/yolic_m2.xml`
   - Purpose: Road hazard detection

3. **SmolVLM** (`smolvlm`)
   - Model: `echarlaix/SmolVLM-256M-Instruct-openvino`
   - Purpose: Vision-language model for image analysis

## API Endpoints

### Detection Endpoints
- `POST /api/hazzard_detect/detect` - Hazard detection
- `POST /api/sign_detect/detect` - Sign detection  
- `POST /api/smolvlm/analyze` - Image analysis with AI
- `POST /api/paddleocr/extract` - Text extraction (placeholder)

### Model Management Endpoints
- `GET /api/models/status` - Get health status of all models
- `GET /api/models/list` - List all loaded models
- `POST /api/models/reload` - Reload a specific model
- `GET /api/models/check/{model_name}` - Check if model is loaded
- `GET /health/models` - Global health check endpoint

## Usage

### Starting the Server

```bash
cd app
python main.py
```

The server will automatically load all models at startup:

```
🚀 Starting AI Festival API server...
📦 Loading AI models...
Loading YOLO Sign Detection model...
✅ YOLO Sign Detection model loaded successfully
Loading YOLIC Hazard Detection model...
✅ YOLIC Hazard Detection model loaded successfully
Loading SmolVLM model...
✅ SmolVLM model loaded successfully
✅ Model loading completed: 3 models loaded
📋 Loaded models: ['yolo_sign_detection', 'yolic_hazard_detection', 'smolvlm']
🎯 API server ready!
```

### Making Detection Requests

All detection endpoints now use pre-loaded models for faster inference:

```bash
# Hazard detection
curl -X POST "http://localhost:8000/api/hazzard_detect/detect" \
  --header "Content-Type: application/octet-stream" \
  --data-binary "@image.jpg"

# Sign detection
curl -X POST "http://localhost:8000/api/sign_detect/detect" \
  --header "Content-Type: application/octet-stream" \
  --data-binary "@image.jpg"

# SmolVLM analysis
curl -X POST "http://localhost:8000/api/smolvlm/analyze?prompt=What do you see?" \
  --header "Content-Type: application/octet-stream" \
  --data-binary "@image.jpg"
```

### Model Management

Check model health:
```bash
curl http://localhost:8000/api/models/status
```

List loaded models:
```bash
curl http://localhost:8000/api/models/list
```

Reload a specific model:
```bash
curl -X POST "http://localhost:8000/api/models/reload" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "yolo_sign_detection"}'
```

## Architecture

### Files Structure

```
app/
├── model_loader.py              # Central model loading and management
├── main.py                      # FastAPI app with startup model loading
├── endpoints/
│   ├── model_management.py      # Model management endpoints
│   ├── hazzard_detection.py     # Hazard detection endpoint
│   ├── sign_detection.py        # Sign detection endpoint
│   └── smolvlm.py               # SmolVLM endpoint
├── services/
│   ├── hazzard_detection_services.py  # Uses pre-loaded YOLIC model
│   ├── sign_detection_services.py     # Uses pre-loaded YOLO model
│   └── smolvlm_services.py            # Uses pre-loaded SmolVLM model
└── utils/
    ├── infer_yolo.py            # Added infer_image_with_preloaded_model()
    ├── infer_yolic.py           # Added infer_image_with_preloaded_model()
    └── infer_smolvlm.py         # Added infer_image_with_preloaded_model()
```

### Model Loading Flow

1. **Server Startup**: `main.py` calls `load_all_models()` from `model_loader.py`
2. **Model Storage**: Models are stored in global `_models` dictionary
3. **Service Calls**: Services use `get_model()` to access pre-loaded models
4. **Inference**: Models perform inference without initialization overhead

## Performance Benefits

- **Faster API Response**: No model loading time during requests
- **Memory Efficiency**: Models loaded once and shared across requests
- **Better User Experience**: Consistent response times
- **Resource Optimization**: Reduced CPU/GPU initialization overhead

## Error Handling

- Server continues to start even if some models fail to load
- Graceful error messages for missing models
- Model management endpoints for runtime troubleshooting
- Health check endpoints for monitoring

## Monitoring

Use the health check endpoints to monitor model status:

```bash
# Global health check
curl http://localhost:8000/health/models

# Detailed model status
curl http://localhost:8000/api/models/status
```

This will return detailed information about each model's health and status.

# 🧪 API Testing Guide

## Quick Start

### 1. Start Your Server
```bash
cd app
python main.py
```

Wait for the models to load. You should see:
```
🚀 Starting AI Festival API server...
📦 Loading AI models...
✅ YOLO Sign Detection model loaded successfully
✅ YOLIC Hazard Detection model loaded successfully  
✅ SmolVLM model loaded successfully
🎯 API server ready!
```

### 2. Run Quick Tests

#### Option A: Automated Testing (Recommended)
```bash
# Run comprehensive test suite
python test_api.py

# Run simple quick test
python simple_test.py
```

#### Option B: Manual Testing
```bash
# Test server health
curl http://127.0.0.1:8000/docs

# Test model health
curl http://127.0.0.1:8000/health/models

# Test with your image (replace 'your_image.jpg' with actual image)
curl -X POST "http://127.0.0.1:8000/api/hazzard_detect/detect" \
  --header "Content-Type: application/octet-stream" \
  --data-binary "@your_image.jpg"
```

## Test Scripts Included

### 1. `test_api.py` - Comprehensive Test Suite
- Tests all endpoints automatically
- Finds test images automatically
- Provides detailed results and timing
- Best for thorough testing

```bash
python test_api.py
```

### 2. `simple_test.py` - Quick Individual Tests
- Test specific endpoints
- Simple pass/fail results
- Good for debugging specific issues

```bash
# Test everything quickly
python simple_test.py

# Test specific endpoints
python simple_test.py hazard your_image.jpg
python simple_test.py sign your_image.jpg
python simple_test.py analyze your_image.jpg "What do you see?"
```

### 3. `test_commands.sh` - Manual Commands
- Shows all curl commands
- Good for copying commands to terminal
- Platform independent (works on Windows with Git Bash)

```bash
bash test_commands.sh
```

## API Endpoints Reference

| Endpoint | Method | Purpose | Input |
|----------|--------|---------|-------|
| `/health/models` | GET | Check model health | None |
| `/api/models/status` | GET | Model status details | None |
| `/api/hazzard_detect/detect` | POST | Detect road hazards | Raw image bytes |
| `/api/sign_detect/detect` | POST | Detect traffic signs | Raw image bytes |
| `/api/smolvlm/analyze` | POST | AI image analysis | Raw image bytes + prompt |
| `/api/paddleocr/extract` | POST | Extract text from images | Raw image bytes |

## Example Responses

### Hazard Detection
```json
{
  "detections": [
    {
      "label": "pothole",
      "confidence": 0.95,
      "position": [100, 150, 200, 250]
    }
  ]
}
```

### Sign Detection
```json
{
  "detections": [
    {
      "label": "Giới hạn tốc độ (50km/h)",
      "confidence": 0.88,
      "position": [50, 75, 150, 175]
    }
  ]
}
```

### SmolVLM Analysis
```json
{
  "description": "I can see a road with several traffic signs including speed limit signs and warning signs. There are some vehicles in the distance and the road appears to be in good condition.",
  "prompt": "Describe what you see in this image."
}
```

## Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :8000

# Try different port
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### Models Not Loading
1. Check model files exist in `ai_models/` directory
2. Check console output for specific model errors
3. Ensure sufficient memory/disk space
4. Check model paths in `model_loader.py`

### API Requests Failing
1. Verify server is running: `curl http://127.0.0.1:8000/docs`
2. Check image file exists and is readable
3. Ensure Content-Type header is set correctly
4. Check server logs for error details

### Test Images Not Found
Place test images in project root or app directory:
- `test_img.jpg`
- `test_sign.jpg` 
- Any `.jpg` or `.png` image files

## Interactive Testing

### Swagger UI
Open in browser: http://127.0.0.1:8000/docs

### Postman/Insomnia
1. Create POST request to endpoint
2. Set Content-Type: `application/octet-stream`
3. Add image file as binary body
4. Send request

## Performance Notes

- First request may be slower (model warmup)
- Subsequent requests should be fast (~1-3 seconds)
- Large images take longer to process
- Model pre-loading eliminates initialization delay

## Next Steps

1. ✅ Run basic tests to ensure everything works
2. ✅ Test with your own images
3. ✅ Check response formats match your needs
4. ✅ Integrate with your frontend/client application
5. ✅ Monitor performance and optimize as needed

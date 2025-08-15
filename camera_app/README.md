# ESP32 Camera Simulator

This application simulates an ESP32 camera that captures images from your laptop's camera and sends them to the AI Festival API for analysis.

## Features

- **Mode 1: Hazard Detection** - Detects hazards and plays corresponding audio files (1-11.mp3)
- **Mode 2: Sign Detection** - Detects traffic signs and plays corresponding audio files (1-52.mp3)  
- **Mode 3: OCR** - Extracts text from images and reads it aloud using TTS
- **Mode 4: SmolVLM** - Analyzes images with AI and describes them using TTS

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your AI Festival API server is running on `http://127.0.0.1:8000`

3. Create audio files for hazard and sign detection:
   - Place hazard audio files in `audio/hazard/` (1.mp3 to 11.mp3)
   - Place sign audio files in `audio/sign/` (1.mp3 to 52.mp3)

## Usage

1. Run the camera simulator:
```bash
python camera_esp32_simulator.py
```

2. Controls:
   - **1-4**: Switch between modes
   - **SPACE**: Capture image and analyze
   - **Q**: Quit application

## Audio Setup

For hazard and sign detection modes, you need to provide audio files:

### Hazard Detection Audio
Place MP3 files in `audio/hazard/`:
- `1.mp3` to `11.mp3` (corresponding to hazard class indices)

### Sign Detection Audio  
Place MP3 files in `audio/sign/`:
- `1.mp3` to `52.mp3` (corresponding to sign class indices)

## Requirements

- Python 3.8+
- Webcam/Camera
- AI Festival API server running locally
- Audio files for hazard/sign detection (optional)

## API Endpoints Used

- `POST /api/hazzard_detect/detect` - Hazard detection
- `POST /api/sign_detect/detect` - Sign detection
- `POST /api/paddleocr/extract` - OCR text extraction
- `POST /api/smolvlm/analyze` - Image analysis with SmolVLM

## Troubleshooting

1. **Camera not found**: Make sure your camera is not being used by another application
2. **API connection error**: Ensure the AI Festival server is running on port 8000
3. **Audio not playing**: Check that audio files exist in the correct directories
4. **TTS not working**: Make sure pyttsx3 is properly installed and audio drivers are working

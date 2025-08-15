#!/bin/bash
# AI Festival API Testing Script
# This script provides manual testing commands for your API

echo "🚀 AI Festival API Testing Commands"
echo "=================================="
echo ""

# API Configuration
API_BASE="http://127.0.0.1:8000"

echo "📋 Available Test Commands:"
echo ""

echo "1. 🏥 Check Server Health:"
echo "   curl $API_BASE/docs"
echo ""

echo "2. 🤖 Check Model Health:"
echo "   curl $API_BASE/health/models"
echo ""

echo "3. ⚙️ Model Management:"
echo "   # List models:"
echo "   curl $API_BASE/api/models/list"
echo ""
echo "   # Check model status:"
echo "   curl $API_BASE/api/models/status"
echo ""
echo "   # Check specific model:"
echo "   curl $API_BASE/api/models/check/yolo_sign_detection"
echo ""

echo "4. 🔍 Detection Endpoints (replace 'image.jpg' with your test image):"
echo ""
echo "   # Hazard Detection:"
echo "   curl -X POST \"$API_BASE/api/hazzard_detect/detect\" \\"
echo "        --header \"Content-Type: application/octet-stream\" \\"
echo "        --data-binary \"@image.jpg\""
echo ""
echo "   # Sign Detection:"
echo "   curl -X POST \"$API_BASE/api/sign_detect/detect\" \\"
echo "        --header \"Content-Type: application/octet-stream\" \\"
echo "        --data-binary \"@image.jpg\""
echo ""
echo "   # SmolVLM Analysis:"
echo "   curl -X POST \"$API_BASE/api/smolvlm/analyze\" \\"
echo "        --header \"Content-Type: application/octet-stream\" \\"
echo "        --data-binary \"@image.jpg\""
echo ""
echo "   # SmolVLM with Custom Prompt:"
echo "   curl -X POST \"$API_BASE/api/smolvlm/analyze?prompt=What%20traffic%20signs%20do%20you%20see?\" \\"
echo "        --header \"Content-Type: application/octet-stream\" \\"
echo "        --data-binary \"@image.jpg\""
echo ""
echo "   # PaddleOCR Text Extraction:"
echo "   curl -X POST \"$API_BASE/api/paddleocr/extract\" \\"
echo "        --header \"Content-Type: application/octet-stream\" \\"
echo "        --data-binary \"@image.jpg\""
echo ""

echo "5. 📖 API Documentation:"
echo "   Open in browser: $API_BASE/docs"
echo ""

echo "💡 Testing Tips:"
echo "- Make sure your server is running: python app/main.py"
echo "- Use actual image files in your test commands"
echo "- Check server logs for detailed error messages"
echo "- Test with different image types (jpg, png, etc.)"
echo ""

# Check if server is running
echo "🔍 Quick Server Check:"
if curl -s -f $API_BASE/docs > /dev/null 2>&1; then
    echo "✅ Server is running at $API_BASE"
else
    echo "❌ Server is not responding at $API_BASE"
    echo "   Start server with: python app/main.py"
fi

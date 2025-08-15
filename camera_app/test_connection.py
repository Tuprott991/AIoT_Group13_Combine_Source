import requests
import time

def test_api_connection():
    """Test if the AI Festival API server is running"""
    api_base_url = "http://127.0.0.1:8000"
    
    print("🔍 Testing API server connection...")
    
    try:
        # Test health endpoint
        response = requests.get(f"{api_base_url}/health/models", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API server is running!")
            print(f"📊 Models loaded: {health_data.get('total_models', 0)}")
            print(f"📋 Available models: {', '.join(health_data.get('loaded_models', []))}")
            return True
        else:
            print(f"❌ API server responded with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Is it running on http://127.0.0.1:8000?")
        print("💡 Start the server with: python app/main.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ API server is not responding (timeout)")
        return False
    except Exception as e:
        print(f"❌ Error testing API connection: {e}")
        return False

def main():
    print("🚀 ESP32 Camera Simulator - API Connection Test")
    print("=" * 50)
    
    if test_api_connection():
        print("\n✅ All systems ready! You can now run the camera simulator:")
        print("   python camera_esp32_simulator.py")
    else:
        print("\n❌ Please start the AI Festival API server first:")
        print("   cd ..")
        print("   python app/main.py")

if __name__ == "__main__":
    main()

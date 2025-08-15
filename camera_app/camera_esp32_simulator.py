import cv2
import requests
import numpy as np
import pygame
import pyttsx3
import threading
import os
import time
import json
from io import BytesIO
from PIL import Image

class CameraESP32Simulator:
    def __init__(self):
        self.api_base_url = "http://127.0.0.1:8000/api"
        self.current_mode = 1
        self.modes = {
            1: "Hazard Detection",
            2: "Sign Detection", 
            3: "OCR",
            4: "SmolVLM"
        }
        self.camera = None
        self.running = False
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        
        # Audio file paths
        self.audio_base_path = os.path.join(os.path.dirname(__file__), "audio")
        
        print("🎥 ESP32 Camera Simulator Initialized")
        print("📋 Available modes:")
        for mode, name in self.modes.items():
            print(f"   {mode}: {name}")
        print("\n🎮 Controls:")
        print("   1-4: Switch modes")
        print("   Space: Capture and analyze")
        print("   Q: Quit")
        
    def initialize_camera(self):
        """Initialize the camera"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("❌ Error: Could not open camera")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully")
        return True
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if not self.camera:
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            print("❌ Error: Could not capture frame")
            return None
        return frame
    
    def frame_to_bytes(self, frame):
        """Convert OpenCV frame to bytes"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Convert to bytes
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format='JPEG', quality=85)
        return img_bytes.getvalue()
    
    def call_hazard_detection_api(self, image_bytes):
        """Call hazard detection API"""
        try:
            url = f"{self.api_base_url}/hazzard_detect/detect"
            headers = {'Content-Type': 'application/octet-stream'}
            
            response = requests.post(url, data=image_bytes, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            print(f"🚨 Hazard Detection Results: {len(result['detections'])} detections")
            
            # Play audio for detected hazards
            for detection in result['detections']:
                class_index = detection.get('class_index', 0)
                confidence = detection.get('confidence', 0)
                class_name = detection.get('class_name', 'Unknown')
                
                print(f"   - {class_name} (confidence: {confidence:.2f})")
                
                # Play corresponding audio file
                self.play_hazard_audio(class_index)
                
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling hazard detection API: {e}")
            return None
    
    def call_sign_detection_api(self, image_bytes):
        """Call sign detection API"""
        try:
            url = f"{self.api_base_url}/sign_detect/detect"
            headers = {'Content-Type': 'application/octet-stream'}
            
            response = requests.post(url, data=image_bytes, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            print(f"🚦 Sign Detection Results: {len(result['detections'])} detections")
            
            # Play audio for detected signs
            for detection in result['detections']:
                class_index = detection.get('class_index', 0)
                confidence = detection.get('confidence', 0)
                class_name = detection.get('class_name', 'Unknown')
                
                print(f"   - {class_name} (confidence: {confidence:.2f})")
                
                # Play corresponding audio file
                self.play_sign_audio(class_index)
                
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling sign detection API: {e}")
            return None
    
    def call_ocr_api(self, image_bytes):
        """Call OCR API"""
        try:
            url = f"{self.api_base_url}/paddleocr/extract"
            headers = {'Content-Type': 'application/octet-stream'}
            
            response = requests.post(url, data=image_bytes, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            print(f"📝 OCR Results: {result['total_texts']} texts detected")
            
            # Speak detected texts
            all_texts = []
            for text_item in result['texts']:
                text = text_item.get('text', '')
                confidence = text_item.get('confidence', 0)
                print(f"   - '{text}' (confidence: {confidence:.2f})")
                if text.strip():
                    all_texts.append(text)
            
            if all_texts:
                combined_text = "Detected text: " + ". ".join(all_texts)
                self.speak_text(combined_text)
            else:
                self.speak_text("No text detected in the image")
                
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling OCR API: {e}")
            return None
    
    def call_smolvlm_api(self, image_bytes):
        """Call SmolVLM API"""
        try:
            url = f"{self.api_base_url}/smolvlm/analyze"
            headers = {'Content-Type': 'application/octet-stream'}
            params = {'prompt': 'Describe what you see in this image in detail.'}
            
            response = requests.post(url, data=image_bytes, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            description = result.get('description', 'No description available')
            
            print(f"🤖 SmolVLM Analysis:")
            print(f"   {description}")
            
            # Speak the description
            self.speak_text(description)
                
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling SmolVLM API: {e}")
            return None
    
    def play_hazard_audio(self, class_index):
        """Play audio file for hazard detection (1-11.mp3)"""
        if 1 <= class_index <= 11:
            audio_file = os.path.join(self.audio_base_path, "hazard", f"{class_index}.mp3")
            if os.path.exists(audio_file):
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"❌ Error playing hazard audio {class_index}: {e}")
            else:
                print(f"⚠️ Audio file not found: {audio_file}")
    
    def play_sign_audio(self, class_index):
        """Play audio file for sign detection (1-52.mp3)"""
        if 1 <= class_index <= 52:
            audio_file = os.path.join(self.audio_base_path, "sign", f"{class_index}.mp3")
            if os.path.exists(audio_file):
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"❌ Error playing sign audio {class_index}: {e}")
            else:
                print(f"⚠️ Audio file not found: {audio_file}")
    
    def speak_text(self, text):
        """Use TTS to speak text"""
        def speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ Error with TTS: {e}")
        
        # Run TTS in a separate thread to avoid blocking
        tts_thread = threading.Thread(target=speak)
        tts_thread.daemon = True
        tts_thread.start()
    
    def process_frame(self, frame):
        """Process captured frame based on current mode"""
        print(f"\n📸 Capturing and analyzing with {self.modes[self.current_mode]}...")
        
        # Convert frame to bytes
        image_bytes = self.frame_to_bytes(frame)
        
        # Call appropriate API based on mode
        if self.current_mode == 1:
            result = self.call_hazard_detection_api(image_bytes)
        elif self.current_mode == 2:
            result = self.call_sign_detection_api(image_bytes)
        elif self.current_mode == 3:
            result = self.call_ocr_api(image_bytes)
        elif self.current_mode == 4:
            result = self.call_smolvlm_api(image_bytes)
        
        return result
    
    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        self.running = True
        print(f"\n🚀 ESP32 Camera Simulator started in mode: {self.modes[self.current_mode]}")
        print("Press SPACE to capture and analyze, or Q to quit")
        
        try:
            while self.running:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Add mode indicator to frame
                mode_text = f"Mode {self.current_mode}: {self.modes[self.current_mode]}"
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Capture | 1-4: Mode | Q: Quit", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('ESP32 Camera Simulator', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Shutting down ESP32 Camera Simulator...")
                    break
                elif key == ord(' '):  # Space key for capture
                    self.process_frame(frame)
                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    new_mode = int(chr(key))
                    if new_mode != self.current_mode:
                        self.current_mode = new_mode
                        print(f"\n🔄 Switched to mode {self.current_mode}: {self.modes[self.current_mode]}")
                        
        except KeyboardInterrupt:
            print("\n👋 Shutting down ESP32 Camera Simulator...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("✅ Cleanup completed")

def main():
    print("🎥 ESP32 Camera Simulator")
    print("=" * 50)
    
    simulator = CameraESP32Simulator()
    simulator.run()

if __name__ == "__main__":
    main()

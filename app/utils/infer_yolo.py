import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

# === CLASS NAMES ===
class_names = [
    "Đường người đi bộ cắt ngang", "Đường giao nhau (ngã ba bên phải)", "Cấm đi ngược chiều",
    "Phải đi vòng sang bên phải", "Giao nhau với đường đồng cấp", "Giao nhau với đường không ưu tiên",
    "Chỗ ngoặt nguy hiểm vòng bên trái", "Cấm rẽ trái", "Bến xe buýt", "Nơi giao nhau chạy theo vòng xuyến",
    "Cấm dừng và đỗ xe", "Chỗ quay xe", "Biển gộp làn đường theo phương tiện", "Đi chậm", "Cấm xe tải",
    "Đường bị thu hẹp về phía phải", "Giới hạn chiều cao", "Cấm quay đầu", "Cấm ô tô khách và ô tô tải",
    "Cấm rẽ phải và quay đầu", "Cấm ô tô", "Đường bị thu hẹp về phía trái", "Gồ giảm tốc phía trước",
    "Cấm xe hai và ba bánh", "Kiểm tra", "Chỉ dành cho xe máy*", "Chướng ngoại vật phía trước", "Trẻ em",
    "Xe tải và xe công*", "Cấm mô tô và xe máy", "Chỉ dành cho xe tải*", "Đường có camera giám sát",
    "Cấm rẽ phải", "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải", "Cấm xe sơ-mi rơ-moóc",
    "Cấm rẽ trái và phải", "Cấm đi thẳng và rẽ phải", "Đường giao nhau (ngã ba bên trái)",
    "Giới hạn tốc độ (50km/h)", "Giới hạn tốc độ (60km/h)", "Giới hạn tốc độ (80km/h)",
    "Giới hạn tốc độ (40km/h)", "Các xe chỉ được rẽ trái", "Chiều cao tĩnh không thực tế",
    "Nguy hiểm khác", "Đường một chiều", "Cấm đỗ xe", "Cấm ô tô quay đầu xe (được rẽ trái)",
    "Giao nhau với đường sắt có rào chắn", "Cấm rẽ trái và quay đầu xe",
    "Chỗ ngoặt nguy hiểm vòng bên phải", "Chú ý chướng ngại vật – vòng tránh sang bên phải"
]

class YOLOv8SignDetection:
    def __init__(self, model_path="ai_models/yolov8_sign_detection/yolov8_traffic_sign.pt", conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize YOLOv8 sign detection with PyTorch
        
        Args:
            model_path (str): Path to the PyTorch model file (.pt)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the YOLO model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Thresholds
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Model path: {model_path}")
        
    def preprocess_image(self, image):
        """
        Preprocess image for YOLOv8 inference (handled internally by ultralytics)
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            numpy.ndarray: Image ready for inference
        """
        # YOLO handles preprocessing internally, just return the image
        return image
    
    def postprocess(self, results):
        """
        Post-process YOLOv8 outputs from ultralytics
        
        Args:
            results: Results from ultralytics YOLO model
            
        Returns:
            list: List of detections with format [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates, confidence, and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence threshold
                    if confidence > self.conf_threshold:
                        # Get class name
                        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                        
                        detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
        
        return detections
    
    def get_position(self, x1, x2, image_width):
        """
        Determine position of detected sign (left, center, right)
        
        Args:
            x1, x2: Left and right coordinates of bounding box
            image_width: Width of the image
            
        Returns:
            str: Position - "left", "center", or "right"
        """
        center_x = (x1 + x2) / 2
        
        if center_x < image_width * 0.35:
            return "left"
        elif center_x > image_width * 0.65:
            return "right"
        else:
            return "center"
    
    def inference(self, image):
        """
        Run inference on image and return detected signs with positions
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            list: List of dictionaries containing sign info and position
        """
        # Run inference using ultralytics YOLO
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Post-process results
        detections = self.postprocess(results)
        
        # Format output
        detected_signs = []
        image_height, image_width = image.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = detection
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # Get position
            position = self.get_position(x1, x2, image_width)
            
            detected_signs.append({
                'sign': class_name,
                'position': position,
                'confidence': float(confidence),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class_id': int(class_id)
            })
        
        return detected_signs
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image with proper Vietnamese font support
        
        Args:
            image: OpenCV image
            detections: List of detections
            
        Returns:
            image: Image with drawn detections
        """
        image_copy = image.copy()
        
        if not detections:
            return image_copy
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font that supports Vietnamese characters
        font = self._load_vietnamese_font()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['sign']
            position = detection['position']
            
            # Choose color based on position (RGB format for PIL)
            if position == "left":
                bbox_color = (0, 255, 0)    # Green
                text_color = (255, 255, 255)  # White text
            elif position == "right":
                bbox_color = (255, 0, 0)    # Red  
                text_color = (255, 255, 255)  # White text
            else:
                bbox_color = (0, 0, 255)    # Blue
                text_color = (255, 255, 255)  # White text
            
            # Draw bounding box using PIL
            draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=2)
            
            # Prepare label text
            label = f"{class_name} ({position})"
            conf_text = f"{confidence:.2f}"
            
            try:
                # Get text dimensions using PIL
                label_bbox = draw.textbbox((0, 0), label, font=font)
                conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
                
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]
                conf_width = conf_bbox[2] - conf_bbox[0]
                conf_height = conf_bbox[3] - conf_bbox[1]
                
                # Calculate background rectangle size
                bg_width = max(label_width, conf_width) + 10
                bg_height = label_height + conf_height + 10
                
                # Ensure background doesn't go outside image bounds
                bg_y1 = max(0, y1 - bg_height)
                bg_x2 = min(pil_image.width, x1 + bg_width)
                
                # Draw background rectangle
                draw.rectangle([x1, bg_y1, bg_x2, y1], fill=bbox_color)
                
                # Draw text
                text_y = bg_y1 + 5
                draw.text((x1 + 5, text_y), label, font=font, fill=text_color)
                draw.text((x1 + 5, text_y + label_height + 2), conf_text, font=font, fill=text_color)
                
            except Exception as e:
                print(f"Warning: Text rendering failed: {e}")
                # Simple fallback
                draw.rectangle([x1, y1-30, x1+200, y1], fill=bbox_color)
                draw.text((x1 + 5, y1 - 25), label[:20], font=font, fill=text_color)
        
        # Convert back to OpenCV format (RGB to BGR)
        result_rgb = np.array(pil_image)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        return result_bgr
    
    def _load_vietnamese_font(self):
        """
        Load a font that supports Vietnamese characters
        
        Returns:
            ImageFont: Font object
        """
        # Try common Vietnamese-supporting fonts on different systems
        font_paths = [
            # Windows fonts
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf", 
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
        ]
        
        # Try each font path
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 16)
                    # Test if font can render Vietnamese characters
                    test_text = "Đường"
                    return font
                except Exception:
                    continue
        
        # Fallback to default font
        try:
            return ImageFont.load_default()
        except:
            # If even default font fails, return None and handle in calling code
            return None

def infer_image(image_path, model_path="ai_models/yolov8_sign_detection/yolov8_traffic_sign.pt"):
    """
    Convenience function to infer a single image
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the PyTorch model
        
    Returns:
        tuple: (detections, annotated_image)
    """
    # Initialize detector
    detector = YOLOv8SignDetection(model_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Run inference
    detections = detector.inference(image)
    
    # Draw detections
    annotated_image = detector.draw_detections(image, detections)
    
    return detections, annotated_image


def infer_image_with_preloaded_model(image):
    """
    Inference function using pre-loaded model from model_loader
    
    Args:
        image (numpy.ndarray): OpenCV image array
        
    Returns:
        list: List of detections with object, confidence, and position
    """
    from model_loader import get_model
    
    # Get pre-loaded model
    detector = get_model("yolo_sign_detection")
    if detector is None:
        raise RuntimeError("YOLO Sign Detection model not loaded. Please ensure models are loaded at startup.")
    
    # Run inference
    detections = detector.inference(image)
    
    # Convert to expected format for services
    results = []
    for detection in detections:
        results.append({
            "sign": detection.get("sign", "unknown"),
            "confidence": detection.get("confidence", 0.0),
            "position": detection.get("position", "don't know huhu"),
            "class_index": detection.get("class_id", -1)
        })
    
    return results

def infer_folder(folder_path, model_path="ai_models/yolov8_sign_detection/yolov8_traffic_sign.pt", save_results=True):
    """
    Infer all images in a folder
    
    Args:
        folder_path (str): Path to folder containing images
        model_path (str): Path to the PyTorch model
        save_results (bool): Whether to save annotated images
        
    Returns:
        dict: Dictionary with filename as key and detections as value
    """
    # Initialize detector
    detector = YOLOv8SignDetection(model_path)
    
    results = {}
    output_dir = "sign_detection_results"
    
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process all images in folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                detections = detector.inference(image)
                results[filename] = detections
                
                # Print results for this image
                print(f"\n{filename}:")
                if detections:
                    for detection in detections:
                        print(f"  - {detection['sign']} at {detection['position']} (confidence: {detection['confidence']:.3f})")
                    
                    # Save annotated image
                    if save_results:
                        annotated_image = detector.draw_detections(image, detections)
                        output_path = os.path.join(output_dir, f"detected_{filename}")
                        cv2.imwrite(output_path, annotated_image)
                        print(f"  Saved annotated image: {output_path}")
                else:
                    print("  - No signs detected")
    
    return results

if __name__ == "__main__":
    # Example usage
    model_path = "ai_models/yolov8_sign_detection/yolov8_traffic_sign.pt"
    
    # Test with a single image
    if os.path.exists("test_sign4.jpg"):
        print("Testing single image inference:")
        try:
            detections, annotated_image = infer_image("test_sign4.jpg", model_path)
            print("Detected signs:")
            for detection in detections:
                print(f"  - {detection['sign']} at {detection['position']} (confidence: {detection['confidence']:.3f})")
            
            # Save the annotated image
            cv2.imwrite("test_sign_detected.jpg", annotated_image)
            print("Saved annotated image as 'test_sign_detected.jpg'")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # # Test with images folder
    # if os.path.exists("images"):
    #     print("\nTesting folder inference:")
    #     try:
    #         results = infer_folder("images", model_path)
    #         print(f"\nProcessed {len(results)} images")
    #     except Exception as e:
    #         print(f"Error: {e}")
    # else:
    #     print("No 'images' folder found. Create one with test images for batch processing.")

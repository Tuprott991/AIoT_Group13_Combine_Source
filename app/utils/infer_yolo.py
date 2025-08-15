import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from openvino.runtime import Core
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
    def __init__(self, model_path="ai_models/yolov8_sign_detection/yolov8_sign_detection.xml", conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize YOLOv8 sign detection with OpenVINO
        
        Args:
            model_path (str): Path to the OpenVINO model XML file
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        
        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Get input shape
        self.input_shape = self.input_layer.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Thresholds
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Expected input size: {self.input_width}x{self.input_height}")
        
    def preprocess_image(self, image):
        """
        Preprocess image for YOLOv8 inference
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            tuple: (preprocessed_image, scale_factors)
        """
        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Resize image while maintaining aspect ratio
        resized_image = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and change from HWC to CHW format
        input_image = rgb_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        
        # Calculate scale factors for coordinate conversion
        scale_x = original_width / self.input_width
        scale_y = original_height / self.input_height
        
        return input_image, (scale_x, scale_y)
    
    def postprocess(self, outputs, scale_factors):
        """
        Post-process YOLOv8 outputs to extract bounding boxes and classes
        
        Args:
            outputs: Raw model outputs
            scale_factors: Scale factors for coordinate conversion
            
        Returns:
            list: List of detections with format [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        scale_x, scale_y = scale_factors
        
        # YOLOv8 output format: [batch, 84, 8400] where 84 = 4 coords + 80 classes
        # But for custom model, it might be [batch, 4+num_classes, detections]
        predictions = outputs[0]
        
        # Handle different output formats
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
            
        # Transpose if needed (YOLOv8 outputs are usually [84, 8400])
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # Now [8400, 84]
        
        detections = []
        
        for prediction in predictions:
            # Extract coordinates and confidence
            x_center, y_center, width, height = prediction[:4]
            
            # Extract class scores (skip first 4 coordinate values)
            class_scores = prediction[4:]
            
            # Get the class with highest score
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Filter by confidence threshold
            if confidence > self.conf_threshold:
                # Convert center coordinates to corner coordinates
                x1 = (x_center - width / 2) * scale_x
                y1 = (y_center - height / 2) * scale_y
                x2 = (x_center + width / 2) * scale_x
                y2 = (y_center + height / 2) * scale_y
                
                # Get class name
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                
                detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
        
        # Apply Non-Maximum Suppression
        if detections:
            detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detections
            
        Returns:
            list: Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Convert to numpy array for easier manipulation
        detections = np.array(detections, dtype=object)
        
        # Extract coordinates and scores
        boxes = np.array([det[:4] for det in detections], dtype=np.float32)
        scores = np.array([det[4] for det in detections], dtype=np.float32)
        
        # Apply OpenCV's NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        if len(indices) > 0:
            # Flatten indices if needed
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [i[0] for i in indices]
            
            return [detections[i] for i in indices]
        
        return []
    
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
        # Preprocess image
        input_image, scale_factors = self.preprocess_image(image)
        
        # Run inference
        results = self.compiled_model([input_image])
        
        # Post-process results
        detections = self.postprocess(results[self.output_layer], scale_factors)
        
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

def infer_image(image_path, model_path="ai_models/yolov8_sign_detection/yolov8_sign_detection.xml"):
    """
    Convenience function to infer a single image
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the OpenVINO model
        
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

def infer_folder(folder_path, model_path="ai_models/yolov8_sign_detection/yolov8_sign_detection.xml", save_results=True):
    """
    Infer all images in a folder
    
    Args:
        folder_path (str): Path to folder containing images
        model_path (str): Path to the OpenVINO model
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
    model_path = "ai_models/yolov8_sign_detection/yolov8_sign_detection.xml"
    
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

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

# Constants from original file
NumCell = 104  # number of cells
NumClass = 11  # number of classes except background class

# Points and cells definition from original file
points_list = [(288, 166), (322, 166), (356, 166), (390, 166), (424, 166), (458, 166), (492, 166), (526, 166),
               (220, 200), (254, 200), (288, 200), (322, 200), (356, 200), (390, 200), (424, 200), (458, 200),
               (492, 200), (526, 200), (560, 200), (594, 200),
               (220, 234), (254, 234), (288, 234), (322, 234), (356, 234), (390, 234), (424, 234), (458, 234),
               (492, 234), (526, 234), (560, 234), (594, 234), (628, 234),
               (254, 268), (288, 268), (322, 268), (356, 268), (390, 268), (424, 268), (458, 268), (492, 268),
               (526, 268), (560, 268), (594, 268), (628, 268),
               (0, 268), (53, 268), (106, 268), (159, 268), (212, 268), (265, 268), (318, 268), (371, 268), (424, 268),
               (477, 268), (530, 268), (583, 268), (636, 268), (689, 268), (742, 268), (795, 268),
               (0, 321), (53, 321), (106, 321), (159, 321), (212, 321), (265, 321), (318, 321), (371, 321), (424, 321),
               (477, 321), (530, 321), (583, 321), (636, 321), (689, 321), (742, 321), (795, 321), (848, 321),
               (0, 374), (53, 374), (106, 374), (159, 374), (212, 374), (265, 374), (318, 374), (371, 374), (424, 374),
               (477, 374), (530, 374), (583, 374), (636, 374), (689, 374), (742, 374), (795, 374), (848, 374),
               (0, 427), (53, 427), (106, 427), (159, 427), (212, 427), (265, 427), (318, 427), (371, 427), (424, 427),
               (477, 427), (530, 427), (583, 427), (636, 427), (689, 427), (742, 427), (795, 427), (848, 427),
               (0, 480), (53, 480), (106, 480), (159, 480), (212, 480), (265, 480), (318, 480), (371, 480), (424, 480),
               (477, 480), (530, 480), (583, 480), (636, 480), (689, 480), (742, 480), (795, 480), (848, 480),
               (184, 0), (244, 0), (304, 0), (364, 0), (424, 0), (484, 0), (544, 0), (604, 0), (244, 60), (304, 60),
               (364, 60), (424, 60), (484, 60), (544, 60), (604, 60), (664, 60)]

cell_list = [[points_list[0], points_list[11]], [points_list[1], points_list[12]], [points_list[2], points_list[13]],
            [points_list[3], points_list[14]], [points_list[4], points_list[15]],
            [points_list[5], points_list[16]], [points_list[6], points_list[17]], [points_list[7], points_list[18]],
            [points_list[8], points_list[21]], [points_list[9], points_list[22]],
            [points_list[10], points_list[23]], [points_list[11], points_list[24]], [points_list[12], points_list[25]],
            [points_list[13], points_list[26]], [points_list[14], points_list[27]],
            [points_list[15], points_list[28]], [points_list[16], points_list[29]], [points_list[17], points_list[30]],
            [points_list[18], points_list[31]], [points_list[19], points_list[32]],
            [points_list[20], points_list[33]], [points_list[21], points_list[34]], [points_list[22], points_list[35]],
            [points_list[23], points_list[36]], [points_list[24], points_list[37]],
            [points_list[25], points_list[38]], [points_list[26], points_list[39]], [points_list[27], points_list[40]],
            [points_list[28], points_list[41]], [points_list[29], points_list[42]],
            [points_list[30], points_list[43]], [points_list[31], points_list[44]], [points_list[45], points_list[62]],
            [points_list[46], points_list[63]], [points_list[47], points_list[64]],
            [points_list[48], points_list[65]], [points_list[49], points_list[66]], [points_list[50], points_list[67]],
            [points_list[51], points_list[68]], [points_list[52], points_list[69]],
            [points_list[53], points_list[70]], [points_list[54], points_list[71]], [points_list[55], points_list[72]],
            [points_list[56], points_list[73]], [points_list[57], points_list[74]],
            [points_list[58], points_list[75]], [points_list[59], points_list[76]], [points_list[60], points_list[77]],
            [points_list[61], points_list[79]], [points_list[62], points_list[80]],
            [points_list[63], points_list[81]], [points_list[64], points_list[82]], [points_list[65], points_list[83]],
            [points_list[66], points_list[84]], [points_list[67], points_list[85]],
            [points_list[68], points_list[86]], [points_list[69], points_list[87]], [points_list[70], points_list[88]],
            [points_list[71], points_list[89]], [points_list[72], points_list[90]],
            [points_list[73], points_list[91]], [points_list[74], points_list[92]], [points_list[75], points_list[93]],
            [points_list[76], points_list[94]], [points_list[78], points_list[96]],
            [points_list[79], points_list[97]], [points_list[80], points_list[98]], [points_list[81], points_list[99]],
            [points_list[82], points_list[100]], [points_list[83], points_list[101]],
            [points_list[84], points_list[102]], [points_list[85], points_list[103]],
            [points_list[86], points_list[104]], [points_list[87], points_list[105]],
            [points_list[88], points_list[106]],
            [points_list[89], points_list[107]], [points_list[90], points_list[108]],
            [points_list[91], points_list[109]], [points_list[92], points_list[110]],
            [points_list[93], points_list[111]],
            [points_list[95], points_list[113]], [points_list[96], points_list[114]],
            [points_list[97], points_list[115]], [points_list[98], points_list[116]],
            [points_list[99], points_list[117]],
            [points_list[100], points_list[118]], [points_list[101], points_list[119]],
            [points_list[102], points_list[120]], [points_list[103], points_list[121]],
            [points_list[104], points_list[122]],
            [points_list[105], points_list[123]], [points_list[106], points_list[124]],
            [points_list[107], points_list[125]], [points_list[108], points_list[126]],
            [points_list[109], points_list[127]],
            [points_list[110], points_list[128]], [points_list[129], points_list[137]],
            [points_list[130], points_list[138]], [points_list[131], points_list[139]],
            [points_list[132], points_list[140]],
            [points_list[133], points_list[141]], [points_list[134], points_list[142]],
            [points_list[135], points_list[143]], [points_list[136], points_list[144]]]

class_names = ["Bump", "Column", "Dent", "Fence", "Creature", "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone",
               "TrafficSign"]

class YOLICPyTorchInference:
    def __init__(self, model_path="ai_models/yolic_m2/yolic_m2.pth.tar"):
        """
        Initialize PyTorch inference for YOLIC model
        
        Args:
            model_path (str): Path to the PyTorch model file (.pth.tar)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model architecture
        self.model = mobilenet_v2()
        self.model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
        
        # Load the trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            torch.Tensor: Preprocessed image ready for inference
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing transforms
        input_tensor = self.preprocess(rgb_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        return input_batch
    
    
    def get_position(self, center_x, image_width):
        """
        Determine position based on center x coordinate
        
        Args:
            center_x (int): Center x coordinate of `detect`ed object
            image_width (int): Width of the image
            
        Returns:
            str: Position - "left", "center", or "right"
        """
        # Adjust thresholds to better match visual perception
        left_threshold = image_width * 0.35   # 35% for left
        right_threshold = image_width * 0.65  # 65% for right (smaller center region)
        
        if center_x < left_threshold:
            return "left"
        elif center_x > right_threshold:
            return "right"
        else:
            return "center"
    
    def inference(self, image):
        """
        Run inference on image and return detected objects with positions
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            list: List of dictionaries containing object name and position
        """
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            result = self.model(input_data)
        
        # Convert to numpy and apply sigmoid activation
        output = torch.sigmoid(result).cpu().numpy()[0]
        
        # Process predictions
        predictions = np.where(output > 0.5, 1, 0)
        detected_objects = []
        
        image_height, image_width = image.shape[:2]
        cell = 0
        normal = np.asarray([0] * NumClass + [1])
        
        for rect in cell_list:
            x1, y1 = rect[0]
            x2, y2 = rect[1]
            
            # Get predictions for this cell
            each = predictions[cell:cell + NumClass + 1]
            each_score = output[cell:cell + NumClass + 1]
            
            # Check if this cell has any detections (not background only)
            if not (each == normal).all():
                # Get indices of detected classes
                index = [i for i, x in enumerate(each) if x == 1]
                
                # If no class detected, use the one with highest score
                if len(index) == 0:
                    max_idx = each_score.argmax()
                    if max_idx != NumClass:  # Not background
                        index.append(max_idx)
                
                # Process detected objects
                for i in index:
                    if i == NumClass:  # Skip background class
                        continue
                    
                    # Calculate center point of the cell (these are already in 224x224 space)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Scale coordinates back to original image size
                    scaled_center_x = int(center_x * image_width / 224)
                    scaled_center_y = int(center_y * image_height / 224)
                    
                    # Determine position based on the cell coordinates in the original coordinate system
                    # Get the coordinate range for position calculation
                    all_x_coords = [point[0] for point in points_list]
                    max_x = max(all_x_coords)
                    position = self.get_position(center_x, max_x)
                    
                    # Add to detected objects
                    detected_objects.append({
                        'object': class_names[i],
                        'position': position,
                        'center_x': scaled_center_x,
                        'center_y': scaled_center_y,
                        'confidence': float(each_score[i]),
                        'class_index': i
                    })
            
            cell += NumClass + 1
        
        return detected_objects

def infer_image(image_path, model_path="ai_models/yolic_m2/yolic_m2.pth.tar"):
    """
    Convenience function to infer a single image
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the PyTorch model
        
    Returns:
        list: List of detected objects with positions
    """
    # Initialize inference
    yolic_inference = YOLICPyTorchInference(model_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Run inference
    detections = yolic_inference.inference(image)
    
    return detections


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
    yolic_inference = get_model("yolic_hazard_detection")
    if yolic_inference is None:
        raise RuntimeError("YOLIC Hazard Detection model not loaded. Please ensure models are loaded at startup.")
    
    # Run inference
    detections = yolic_inference.inference(image)
    
    # Convert to expected format for services
    results = []
    for detection in detections:
        results.append({
            "object": detection.get("object", "unknown"),
            "confidence": detection.get("confidence", 0.0),
            "position": detection.get("position", [0, 0, 0, 0]),
            "class_index": detection.get("class_index", -1)
        })
    
    return results  

def infer_folder(folder_path, model_path="ai_models/yolic_m2/yolic_m2.pth.tar"):
    """
    Infer all images in a folder
    
    Args:
        folder_path (str): Path to folder containing images
        model_path (str): Path to the PyTorch model
        
    Returns:
        dict: Dictionary with filename as key and detections as value
    """
    # Initialize inference
    yolic_inference = YOLICPyTorchInference(model_path)
    
    results = {}
    
    # Process all images in folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                detections = yolic_inference.inference(image)
                results[filename] = detections
                
                # Print results for this image
                print(f"\n{filename}:")
                if detections:
                    for detection in detections:
                        print(f"  - {detection['object']} at {detection['position']} (confidence: {detection['confidence']:.3f})")
                else:
                    print("  - No objects detected")
    
    return results

# if __name__ == "__main__":
#     # Example usage
#     model_path = "ai_models/yolic_m2/yolic_m2.pth.tar"
    
#     # Test with a single image
#     if os.path.exists("test_img4.jpg"):
#         print("Testing single image inference:")
#         try:
#             detections = infer_image("test_img4.jpg", model_path)
#             print("Detected objects:")
#             for detection in detections:
#                 print(f"  - {detection['object']} at {detection['position']} (confidence: {detection['confidence']:.3f})")
#         except Exception as e:
#             print(f"Error: {e}")
    
#     # # Test with images folder
#     # if os.path.exists("images"):
#     #     print("\nTesting folder inference:")
#     #     try:
#     #         results = infer_folder("images", model_path)
#     #         print(f"\nProcessed {len(results)} images")
#     #     except Exception as e:
#     #         print(f"Error: {e}")
#     # else:
#     #     print("No 'images' folder found. Please create one with test images.")

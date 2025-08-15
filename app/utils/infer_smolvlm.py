from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import torch
import os
import time

class SmolVLMInference:
    def __init__(self, model_id="echarlaix/SmolVLM-256M-Instruct-openvino"):
        """
        Initialize SmolVLM model for visual question answering
        
        Args:
            model_id (str): HuggingFace model ID for SmolVLM OpenVINO model
        """
        print("Loading SmolVLM model...")
        try:
            self.model = OVModelForVisualCausalLM.from_pretrained(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Set pad_token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("SmolVLM model loaded successfully!")
            # print(f"Model type: {type(self.model)}")
            # print(f"Tokenizer type: {type(self.tokenizer)}")
            # print(f"Processor type: {type(self.processor)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def describe_image(self, image_path, prompt="Describe what you see in this image.", max_tokens=150):
        """
        Generate description for an image
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated description
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # Load and process the image
        image = Image.open(image_path)
        print(f"Image loaded: {image.size}, mode: {image.mode}")
        
        try:
            # Try different input formats for SmolVLM
            
            # Method 1: Use the standard format with image token
            formatted_prompt = f"<image>\n{prompt}"
            inputs = self.processor(text=formatted_prompt, images=[image], return_tensors="pt")
            print("Method 1 successful: Using <image> token format")
            
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            try:
                # Method 2: Try without image token
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
                formatted_prompt = prompt
                print("Method 2 successful: Using simple text format")
                
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                try:
                    # Method 3: Try with different format
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                    formatted_prompt = prompt
                    print("Method 3 successful: Using images-first format")
                    
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")
                    raise ValueError(f"Failed to process inputs with all methods. Errors: {e1}, {e2}, {e3}")
        
        # print(f"Input keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        # Generate response with improved parameters
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            # Fallback generation
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        # Remove the input prompt if it appears in the response
        for prompt_variant in [formatted_prompt, prompt, f"<image>\n{prompt}"]:
            if prompt_variant in response:
                response = response.replace(prompt_variant, "").strip()
                break
        
        # Clean up the response by removing repetitive patterns
        lines = response.split('.')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines and len(line) > 10:
                unique_lines.append(line)
                seen_lines.add(line)
                
        response = '. '.join(unique_lines)
        if response and not response.endswith('.'):
            response += '.'
        
        return response
    
    def analyze_traffic_scene(self, image_path):
        """
        Analyze a traffic scene image for road safety
        
        Args:
            image_path (str): Path to the traffic scene image
            
        Returns:
            str: Analysis of the traffic scene
        """
        prompt = "Analyze this traffic scene. Describe any vehicles, road signs, pedestrians, and potential safety concerns you observe."
        return self.describe_image(image_path, prompt, max_tokens=200)
    
    def identify_objects(self, image_path):
        """
        Identify objects in the image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: List of identified objects
        """
        prompt = "List all the objects and items you can see in this image."
        return self.describe_image(image_path, prompt, max_tokens=100)

def infer_image_description(image_path, prompt="Describe what you see in this image."):
    """
    Convenience function to get image description
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt for description
        
    Returns:
        str: Generated description
    """
    vlm = SmolVLMInference()
    return vlm.describe_image(image_path, prompt)


def infer_image_with_preloaded_model(image, prompt="Describe what you see in this image.", max_tokens=150):
    """
    Inference function using pre-loaded model from model_loader
    
    Args:
        image: PIL Image or numpy array
        prompt (str): Text prompt for the model
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Generated description
    """
    from model_loader import get_model
    from PIL import Image
    import numpy as np
    import cv2
    
    # Get pre-loaded model
    vlm = get_model("smolvlm")
    if vlm is None:
        raise RuntimeError("SmolVLM model not loaded. Please ensure models are loaded at startup.")
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB if it's from OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Format prompt
    formatted_prompt = f"<image>\n{prompt}"
    
    try:
        # Process the image and prompt
        inputs = vlm.processor(text=formatted_prompt, images=[image], return_tensors="pt")
        
        # Generate response
        outputs = vlm.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=vlm.tokenizer.eos_token_id,
            eos_token_id=vlm.tokenizer.eos_token_id
        )
        
        # Decode the response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        description = vlm.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return description.strip()
        
    except Exception as e:
        raise RuntimeError(f"Error during SmolVLM inference: {str(e)}")

if __name__ == "__main__":
    # Initialize the model
    vlm = SmolVLMInference()
    
    # Test with different image files
    test_images = ["test_sign_detected.jpg", "test_img2.png", "test_image.jpg"]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n=== Analyzing {image_path} ===")
            
            try:
                
                # General description
                start_time = time.time()
                print("General Description:")
                description = vlm.describe_image(image_path)
                print(f"  {description}")
                print(f"  Time taken: {time.time() - start_time:.2f} seconds")
                
                # # Traffic scene analysis
                # print("\nTraffic Scene Analysis:")
                # traffic_analysis = vlm.analyze_traffic_scene(image_path)
                # print(f"  {traffic_analysis}")
                
                # # Object identification
                # print("\nObject Identification:")
                # objects = vlm.identify_objects(image_path)
                # print(f"  {objects}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Image file not found: {image_path}")
    
    if not any(os.path.exists(img) for img in test_images):
        print("No test images found. Please ensure you have image files to test with.")

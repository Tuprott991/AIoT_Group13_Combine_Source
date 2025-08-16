from transformers import AutoTokenizer, AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image
import torch
import os
import time

class SmolVLMInference:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-256M-Instruct"):
        """
        Initialize SmolVLM model for visual question answering using PyTorch
        
        Args:
            model_id (str): HuggingFace model ID for SmolVLM PyTorch model
        """
        print("Loading SmolVLM model with PyTorch...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            # Load model, tokenizer, and processor - use specific Idefics3 class
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True  # Allow custom model architectures
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Set pad_token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device if not using device_map
            if not torch.cuda.is_available():
                self.model.to(self.device)
            
            self.model.eval()
            
            print("✅ SmolVLM model loaded successfully!")
            print(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to a simpler model if SmolVLM is not available
            try:
                print("Trying fallback model...")
                from transformers import LlavaForConditionalGeneration
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
                self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                if not torch.cuda.is_available():
                    self.model.to(self.device)
                
                self.model.eval()
                print("✅ Fallback model loaded successfully!")
            except Exception as e2:
                print(f"❌ Fallback model also failed: {e2}")
                raise
    
    def describe_image(self, image_path, prompt="Describe what you see in this image.", max_tokens=150):
        """
        Generate description for an image using PyTorch
        
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
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded: {image.size}, mode: {image.mode}")
        
        try:
            # Format prompt for vision model
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # Process inputs
            inputs = self.processor(
                text=formatted_prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print(f"Input keys: {list(inputs.keys())}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} on {value.device}")
            
        except Exception as e:
            print(f"Error processing inputs: {e}")
            raise ValueError(f"Failed to process inputs: {e}")
        
        # Generate response
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
        except Exception as e:
            print(f"Error during generation: {e}")
            # Fallback generation with simpler parameters
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"] if "pixel_values" in inputs else inputs["images"],
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            except Exception as e2:
                print(f"Fallback generation also failed: {e2}")
                raise ValueError(f"Generation failed: {e}, {e2}")
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        # Remove the input prompt if it appears in the response
        for prompt_variant in [formatted_prompt, prompt, f"USER: <image>\n{prompt}\nASSISTANT:"]:
            if prompt_variant in response:
                response = response.replace(prompt_variant, "").strip()
                break
        
        # Additional cleanup
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        if "USER:" in response:
            response = response.split("USER:")[0].strip()
        
        # Clean up the response by removing repetitive patterns
        lines = response.split('.')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines and len(line) > 5:
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

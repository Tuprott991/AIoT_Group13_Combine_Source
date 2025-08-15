import os
import paddle
import paddle2onnx
import yaml
import argparse
import shutil
from pathlib import Path

def export_trained_model_to_inference(model_dir, config_path, output_dir):
    """
    Export trained PaddleOCR model to inference format
    
    Args:
        model_dir (str): Directory containing trained model files (.pdparams, .pdopt, .pdstate)
        config_path (str): Path to the config.yml file
        output_dir (str): Directory to save inference model
    """
    try:
        # Import PaddleOCR modules
        import sys
        sys.path.append('/content/PaddleOCR_repo')  # Adjust path as needed
        
        from ppocr.modeling.architectures import build_model
        from ppocr.utils.save_load import load_model
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Build model
        model = build_model(config['Architecture'])
        
        # Find the latest checkpoint
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pdparams')]
        if not checkpoint_files:
            raise FileNotFoundError("No .pdparams files found")
        
        # Use the most recent checkpoint (or 'latest.pdparams' if exists)
        if 'latest.pdparams' in checkpoint_files:
            checkpoint_path = os.path.join(model_dir, 'latest')
        else:
            # Find the most recent checkpoint
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            checkpoint_path = os.path.join(model_dir, checkpoint_files[-1].replace('.pdparams', ''))
        
        # Load model weights
        load_model(config, model, checkpoint_path)
        model.eval()
        
        # Create inference model directory
        inference_dir = os.path.join(output_dir, 'inference')
        os.makedirs(inference_dir, exist_ok=True)
        
        # Define input shape based on config
        if 'Train' in config and 'dataset' in config['Train']:
            transforms = config['Train']['dataset'].get('transforms', [])
            input_shape = [3, 32, 320]  # Default
            for transform in transforms:
                if 'RecResizeImg' in transform:
                    input_shape = transform['RecResizeImg']['image_shape']
                    break
        else:
            input_shape = [3, 32, 320]  # Default for recognition
        
        # Create dummy input
        dummy_input = paddle.randn([1] + input_shape)
        
        # Export to static model
        paddle.jit.save(
            model,
            os.path.join(inference_dir, 'inference'),
            input_spec=[paddle.static.InputSpec(shape=[None] + input_shape, dtype='float32')]
        )
        
        print(f"Inference model exported to: {inference_dir}")
        return inference_dir
        
    except Exception as e:
        print(f"Error exporting trained model: {str(e)}")
        return None

def convert_paddle_to_onnx(model_dir, output_dir, model_name="vietnamese_ocr_model"):
    """
    Convert PaddleOCR model to ONNX format
    
    Args:
        model_dir (str): Directory containing the PaddleOCR model files
        output_dir (str): Directory to save the ONNX model
        model_name (str): Name for the output ONNX model
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find model files
    model_files = {
        'params': None,
        'model': None,
        'config': None,
        'trained_params': None
    }
    
    for file in os.listdir(model_dir):
        if file.endswith('.pdparams'):
            model_files['trained_params'] = os.path.join(model_dir, file)
        elif file.endswith('.pdmodel'):
            model_files['model'] = os.path.join(model_dir, file)
        elif file.endswith('.pdiparams'):
            model_files['params'] = os.path.join(model_dir, file)
        elif file == 'config.yml':
            model_files['config'] = os.path.join(model_dir, file)
    
    # If we don't have inference model files, try to export from trained model
    if not model_files['model'] and model_files['trained_params'] and model_files['config']:
        print("No inference model found. Attempting to export from trained model...")
        inference_dir = export_trained_model_to_inference(
            model_dir, model_files['config'], output_dir
        )
        if inference_dir:
            # Update model files to point to exported inference model
            for file in os.listdir(inference_dir):
                if file.endswith('.pdmodel'):
                    model_files['model'] = os.path.join(inference_dir, file)
                elif file.endswith('.pdiparams'):
                    model_files['params'] = os.path.join(inference_dir, file)
    
    # Check if all required files are found
    if not model_files['params'] or not model_files['model']:
        print("Error: Could not find .pdiparams or .pdmodel files")
        print("Available files:")
        for key, value in model_files.items():
            print(f"  {key}: {value}")
        return False
    
    try:
        # Load config file to get model information
        config = None
        if model_files['config']:
            with open(model_files['config'], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config: {model_files['config']}")
        
        # Convert to ONNX
        print(f"Converting model from: {model_dir}")
        print(f"Model file: {model_files['model']}")
        print(f"Params file: {model_files['params']}")
        
        onnx_model = paddle2onnx.command.c_paddle_to_onnx(
            model_file=model_files['model'],
            params_file=model_files['params'],
            opset_version=11,
            enable_onnx_checker=True
        )
        
        # Save ONNX model
        onnx_output_path = os.path.join(output_dir, f"{model_name}.onnx")
        with open(onnx_output_path, "wb") as f:
            f.write(onnx_model)
        
        print(f"Successfully converted model to ONNX: {onnx_output_path}")
        
        # Copy config file to output directory
        if model_files['config']:
            config_output_path = os.path.join(output_dir, "config.yml")
            shutil.copy2(model_files['config'], config_output_path)
            print(f"Copied config file to: {config_output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

def find_latest_model(base_dir):
    """
    Find the latest model directory based on modification time
    
    Args:
        base_dir (str): Base directory to search for model folders
        
    Returns:
        str: Path to the latest model directory
    """
    
    model_dirs = []
    
    # Look for directories that might contain models
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if directory contains model files
            has_model = any(f.endswith('.pdparams') or f.endswith('.pdmodel') 
                          for f in os.listdir(item_path))
            if has_model:
                model_dirs.append((item_path, os.path.getmtime(item_path)))
    
    if not model_dirs:
        # Also check the base directory itself
        has_model = any(f.endswith('.pdparams') or f.endswith('.pdmodel') 
                      for f in os.listdir(base_dir))
        if has_model:
            return base_dir
        return None
    
    # Return the most recently modified directory
    latest_dir = max(model_dirs, key=lambda x: x[1])[0]
    return latest_dir

def main():
    parser = argparse.ArgumentParser(description='Convert PaddleOCR model to ONNX')
    parser.add_argument('--model_dir', type=str, 
                       default='/content/output/rec_vietnamese',
                       help='Directory containing PaddleOCR model files')
    parser.add_argument('--output_dir', type=str, 
                       default='/content/onnx_models',
                       help='Directory to save ONNX model')
    parser.add_argument('--model_name', type=str, 
                       default='vietnamese_ocr_model',
                       help='Name for the output ONNX model')
    parser.add_argument('--auto_find', action='store_true',
                       help='Automatically find the latest model directory')
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    
    # Auto-find latest model if requested
    if args.auto_find:
        latest_model = find_latest_model(model_dir)
        if latest_model:
            model_dir = latest_model
            print(f"Found latest model directory: {model_dir}")
        else:
            print("No model directories found!")
            return
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory does not exist: {model_dir}")
        return
    
    print(f"Converting model from: {model_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # List available files for debugging
    print("\nAvailable files in model directory:")
    for file in os.listdir(model_dir):
        print(f"  {file}")
    
    # Perform conversion
    success = convert_paddle_to_onnx(model_dir, args.output_dir, args.model_name)
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print(f"ONNX model saved to: {args.output_dir}")
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
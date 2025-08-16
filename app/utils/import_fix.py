"""
Import order fix for PyTorch-PaddlePaddle conflict
Import this module before any model imports to resolve CUDA conflicts
"""

# Import PyTorch first to claim CUDA resources
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} loaded first")
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available")
    TORCH_AVAILABLE = False

# Then import PaddlePaddle
try:
    import paddle
    print(f"✅ PaddlePaddle {paddle.__version__} loaded after PyTorch")
    PADDLE_AVAILABLE = True
except ImportError:
    print("⚠️ PaddlePaddle not available")
    PADDLE_AVAILABLE = False

# Finally PaddleOCR
try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR imported successfully")
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PaddleOCR not available: {e}")
    PADDLEOCR_AVAILABLE = False

def get_safe_paddle_ocr():
    """
    Get PaddleOCR instance with error handling and conflict resolution
    
    Returns:
        PaddleOCR instance or None if failed
    """
    if not PADDLEOCR_AVAILABLE:
        print("❌ PaddleOCR not available")
        return None
        
    try:
        # Use CPU only to avoid CUDA conflicts with PyTorch
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False  # Reduce verbose output
        )
        print("✅ PaddleOCR initialized successfully (CPU mode)")
        return ocr
        
    except Exception as e:
        print(f"❌ PaddleOCR initialization failed: {e}")
        return None

def check_cuda_conflicts():
    """
    Check for potential CUDA conflicts between frameworks
    
    Returns:
        dict: Status of each framework's CUDA support
    """
    status = {
        'torch_cuda': False,
        'paddle_cuda': False,
        'conflicts': []
    }
    
    if TORCH_AVAILABLE:
        try:
            status['torch_cuda'] = torch.cuda.is_available()
            if status['torch_cuda']:
                print(f"✅ PyTorch CUDA available: {torch.cuda.device_count()} devices")
        except Exception as e:
            print(f"⚠️ PyTorch CUDA check failed: {e}")
    
    if PADDLE_AVAILABLE:
        try:
            import paddle
            if hasattr(paddle.device, 'is_compiled_with_cuda'):
                status['paddle_cuda'] = paddle.device.is_compiled_with_cuda()
                if status['paddle_cuda']:
                    print("✅ PaddlePaddle compiled with CUDA")
        except Exception as e:
            print(f"⚠️ PaddlePaddle CUDA check failed: {e}")
            status['conflicts'].append(f"PaddlePaddle CUDA error: {e}")
    
    # Check for known conflict patterns
    if status['torch_cuda'] and status['paddle_cuda']:
        status['conflicts'].append("Both PyTorch and PaddlePaddle have CUDA - potential conflict")
    
    return status

# Run conflict check on import
if __name__ == "__main__":
    print("🔍 Checking CUDA conflicts...")
    conflicts = check_cuda_conflicts()
    if conflicts['conflicts']:
        print("⚠️ Potential conflicts detected:")
        for conflict in conflicts['conflicts']:
            print(f"  - {conflict}")
    else:
        print("✅ No obvious conflicts detected")

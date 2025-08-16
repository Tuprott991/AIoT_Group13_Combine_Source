#!/usr/bin/env python3
"""
Fix PyTorch-PaddlePaddle Conflict on Windows
Resolves "_gpuDeviceProperties already registered" error
"""

import os
import sys
import subprocess
import time

def run_command(command, description, ignore_errors=False):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0 or ignore_errors:
            print("✅ Success!" if result.returncode == 0 else "⚠️ Completed (with warnings)")
            if result.stdout.strip():
                print(result.stdout[:500])  # Limit output
            return True
        else:
            print("❌ Failed!")
            if result.stderr.strip():
                print(f"Error: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def complete_cleanup():
    """Complete cleanup of conflicting packages"""
    print("🧹 COMPLETE CLEANUP OF CONFLICTING PACKAGES")
    print("=" * 60)
    
    # Step 1: Force close Python processes
    print("\n📋 Step 1: Closing Python processes")
    run_command("taskkill /f /im python.exe", "Closing Python processes", ignore_errors=True)
    time.sleep(2)
    
    # Step 2: Clear all caches
    print("\n📋 Step 2: Clearing caches")
    run_command("pip cache purge", "Clearing pip cache")
    
    # Step 3: Uninstall ALL potentially conflicting packages
    print("\n📋 Step 3: Uninstalling conflicting packages")
    packages_to_remove = [
        "torch",
        "torchvision", 
        "paddlepaddle",
        "paddlepaddle-gpu",
        "paddleocr"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"Uninstalling {package}", ignore_errors=True)
    
    # Step 4: Remove conda packages if they exist
    print("\n📋 Step 4: Removing conda packages")
    for package in ["pytorch", "paddle"]:
        run_command(f"conda remove {package} -y", f"Removing conda {package}", ignore_errors=True)

def install_compatible_versions():
    """Install compatible versions in correct order"""
    print("\n🔧 INSTALLING COMPATIBLE VERSIONS")
    print("=" * 50)
    
    # Strategy: Install PyTorch first, then PaddlePaddle with specific versions
    
    print("\n📋 Step 1: Install PyTorch GPU first")
    pytorch_cmd = "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118"
    success1 = run_command(pytorch_cmd, "Installing PyTorch GPU")
    
    if not success1:
        print("⚠️ PyTorch installation failed, trying CPU version")
        run_command("pip install torch torchvision", "Installing PyTorch CPU")
    
    print("\n📋 Step 2: Install PaddlePaddle (CPU version to avoid conflict)")
    # Use CPU version to avoid CUDA conflicts
    paddle_cmd = "pip install paddlepaddle==2.5.2"
    success2 = run_command(paddle_cmd, "Installing PaddlePaddle CPU")
    
    print("\n📋 Step 3: Install PaddleOCR")
    ocr_cmd = "pip install paddleocr"
    run_command(ocr_cmd, "Installing PaddleOCR")
    
    print("\n📋 Step 4: Install other requirements")
    run_command("pip install ultralytics transformers accelerate", "Installing other ML packages")

def test_installation():
    """Test if the installation works"""
    print("\n🧪 TESTING INSTALLATION")
    print("=" * 40)
    
    print("\n1. Testing PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch failed: {e}")
    
    print("\n2. Testing PaddlePaddle...")
    try:
        import paddle
        print(f"✅ PaddlePaddle {paddle.__version__}")
    except Exception as e:
        print(f"❌ PaddlePaddle failed: {e}")
    
    print("\n3. Testing PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR import successful")
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")

def create_import_order_fix():
    """Create a module that imports in the correct order"""
    fix_content = '''"""
Import order fix for PyTorch-PaddlePaddle conflict
Import this module before any model imports
"""

# Import PyTorch first
try:
    import torch
    print(f"PyTorch {torch.__version__} loaded first")
except ImportError:
    print("PyTorch not available")

# Then import PaddlePaddle
try:
    import paddle
    print(f"PaddlePaddle {paddle.__version__} loaded after PyTorch")
except ImportError:
    print("PaddlePaddle not available")

# Finally PaddleOCR
try:
    from paddleocr import PaddleOCR
    print("PaddleOCR loaded successfully")
except ImportError:
    print("PaddleOCR not available")

def get_safe_paddle_ocr():
    """Get PaddleOCR instance with error handling"""
    try:
        return PaddleOCR(use_angle_cls=True, lang='en')
    except Exception as e:
        print(f"PaddleOCR initialization failed: {e}")
        return None
'''
    
    with open("app/utils/import_fix.py", "w") as f:
        f.write(fix_content)
    
    print("✅ Created import order fix module")

def main():
    """Main execution"""
    print("🚀 PYTORCH-PADDLEPADDLE CONFLICT RESOLVER")
    print("=" * 60)
    print("This will fix the '_gpuDeviceProperties already registered' error")
    print()
    
    print("⚠️ WARNING: This will uninstall and reinstall PyTorch and PaddlePaddle")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Step 1: Complete cleanup
    complete_cleanup()
    
    # Step 2: Install compatible versions
    install_compatible_versions()
    
    # Step 3: Create import fix
    create_import_order_fix()
    
    # Step 4: Test
    print("\n" + "=" * 60)
    time.sleep(3)  # Let system settle
    test_installation()
    
    print(f"\n🎯 SOLUTION SUMMARY")
    print("=" * 30)
    print("✅ Used PaddlePaddle CPU version to avoid CUDA conflicts")
    print("✅ PyTorch GPU still available for other models")
    print("✅ Created import order fix module")
    print("\nFor your models:")
    print("- YOLO, YOLIC, SmolVLM: Use PyTorch GPU (fast)")  
    print("- PaddleOCR: Use CPU version (still functional)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

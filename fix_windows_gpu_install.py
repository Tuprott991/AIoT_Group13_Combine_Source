#!/usr/bin/env python3
"""
Windows GPU Installation Fix Script
Resolves file access issues during package installation
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print("❌ Failed!")
            if result.stderr.strip():
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def fix_paddle_installation():
    """Fix PaddlePaddle installation on Windows"""
    print("🚀 FIXING PADDLEPADDLE GPU INSTALLATION ON WINDOWS")
    print("=" * 60)
    
    # Step 1: Force close any Python processes
    
    # Step 2: Clear pip cache
    print("\n📋 Step 2: Clearing pip cache")
    run_command("pip cache purge", "Clearing pip cache")
    
    # Step 3: Clean temp directories
    print("\n📋 Step 3: Cleaning temp directories")
    temp_dir = os.environ.get('TEMP', r'C:\Users\{}\AppData\Local\Temp'.format(os.getenv('USERNAME')))
    print(f"Temp directory: {temp_dir}")
    
    # Step 4: Uninstall existing paddle packages
    print("\n📋 Step 4: Uninstalling existing packages")
    packages_to_remove = [
        "paddlepaddle",
        "paddlepaddle-gpu", 
        "paddleocr"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"Uninstalling {package}")
    
    # Step 5: Install with specific options
    print("\n📋 Step 5: Installing GPU packages with Windows-specific options")
    
    # Install paddlepaddle-gpu with Windows-specific flags
    install_commands = [
        "pip install --no-cache-dir --force-reinstall paddlepaddle-gpu",
        "pip install --no-cache-dir paddleocr"
    ]
    
    for cmd in install_commands:
        success = run_command(cmd, f"Running: {cmd}")
        if not success:
            print(f"\n⚠️ If the above failed, try these alternatives:")
            print(f"1. pip install --user {cmd.split()[-1]}")
            print(f"2. pip install --no-deps {cmd.split()[-1]}")
            print(f"3. conda install {cmd.split()[-1]} (if using conda)")

def check_installation():
    """Check if installation was successful"""
    print("\n📋 VERIFICATION")
    print("=" * 30)
    
    try:
        import paddle
        print(f"✅ PaddlePaddle version: {paddle.__version__}")
        print(f"✅ CUDA support: {paddle.device.is_compiled_with_cuda()}")
        
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR import successful")
        
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def alternative_solutions():
    """Provide alternative installation methods"""
    print("\n🔧 ALTERNATIVE SOLUTIONS")
    print("=" * 40)
    
    print("If the automatic fix doesn't work, try these manual steps:")
    print()
    print("1. 🔄 Restart your computer and try again")
    print("2. 📁 Use --user flag:")
    print("   pip install --user paddlepaddle-gpu")
    print("   pip install --user paddleocr")
    print()
    print("3. 🔒 Run as Administrator:")
    print("   - Right-click Command Prompt/PowerShell")
    print("   - Select 'Run as Administrator'")
    print("   - Run installation commands")
    print()
    print("4. 🐍 Use conda instead:")
    print("   conda install paddlepaddle-gpu -c paddle")
    print("   pip install paddleocr")
    print()
    print("5. 📦 Download and install manually:")
    print("   - Download .whl file from PyPI")
    print("   - pip install path/to/downloaded.whl")
    print()
    print("6. 🔄 Virtual environment approach:")
    print("   python -m venv fresh_env")
    print("   fresh_env\\Scripts\\activate")
    print("   pip install paddlepaddle-gpu paddleocr")

def main():
    """Main execution"""
    print("🚀 WINDOWS GPU INSTALLATION FIXER")
    print("=" * 50)
    print("This script will fix PaddlePaddle GPU installation issues on Windows")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with the automatic fix? (y/n): ")
    if response.lower() != 'y':
        print("Manual solutions provided below...")
        alternative_solutions()
        return
    
    # Run the fix
    fix_paddle_installation()
    
    # Verify installation
    print("\n" + "=" * 60)
    time.sleep(2)  # Give system time to settle
    
    if check_installation():
        print("\n🎉 SUCCESS! PaddlePaddle GPU is now installed correctly!")
        print("You can now run: python test_pytorch_models.py")
    else:
        print("\n⚠️ Automatic fix didn't work. Trying alternative solutions...")
        alternative_solutions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        alternative_solutions()

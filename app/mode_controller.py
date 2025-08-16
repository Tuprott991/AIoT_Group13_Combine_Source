# Global mode configuration
# Mode 1: Hazard Detection
# Mode 2: Sign Detection  
# Mode 3: PaddleOCR
# Mode 4: SmolVLM

current_mode = 1  # Default mode is hazard detection

def get_current_mode():
    """Get current detection mode"""
    return current_mode

def set_current_mode(mode: int):
    """Set current detection mode"""
    global current_mode
    if mode in [1, 2, 3, 4]:
        current_mode = mode
        return True
    return False

def get_mode_name(mode: int):
    """Get mode name for display"""
    mode_names = {
        1: "Hazard Detection",
        2: "Sign Detection", 
        3: "PaddleOCR",
        4: "SmolVLM"
    }
    return mode_names.get(mode, "Unknown")

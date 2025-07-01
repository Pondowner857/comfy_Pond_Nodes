import time
import json
import threading
import requests
import logging
import sys
from server import PromptServer

# Try to intercept and filter HiDream logs
class HiDreamFilter(logging.Filter):
    def filter(self, record):
        # If log message is from HiDream and contains specific keywords, filter it out
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if record.msg.startswith("HiDream:") and (
                "unloading all models" in record.msg or 
                "Cleaning up all cached models" in record.msg or
                "Cache cleared" in record.msg
            ):
                return False
        return True

# Apply log filter
for handler in logging.root.handlers:
    handler.addFilter(HiDreamFilter())

class MemoryManager:
    def __init__(self):
        self.server = PromptServer.instance
        self.enabled = False
        self.interval = 60  # Default 60 seconds
        self.timer = None
        self.timer_lock = threading.Lock()
        self.verbose = False  # Default to not outputting verbose information
        self.last_free_time = 0  # Last time memory was freed
        self.min_interval = 0.1  # Minimum interval time (seconds) - changed to 0.1 seconds
        
    def start(self):
        """Start memory cleanup timer"""
        with self.timer_lock:
            if self.timer is not None:
                self.stop()
            
            self.enabled = True
            self.timer = threading.Timer(self.interval, self._timer_callback)
            self.timer.daemon = True
            self.timer.start()
            # Only output a message on first startup
            if self.verbose:
                print(f"Memory manager started, interval: {self.interval} seconds")
    
    def stop(self):
        """Stop memory cleanup timer"""
        with self.timer_lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
                self.enabled = False
                if self.verbose:
                    print("Memory manager stopped")
    
    def _timer_callback(self):
        """Timer callback function, execute memory cleanup and reset timer"""
        self.free_memory()
        
        # Reset timer for loop calling
        if self.enabled:
            with self.timer_lock:
                self.timer = threading.Timer(self.interval, self._timer_callback)
                self.timer.daemon = True
                self.timer.start()
    
    def free_memory(self):
        """Call API to free memory"""
        try:
            # Check if minimum interval time has passed
            current_time = time.time()
            if current_time - self.last_free_time < self.min_interval:
                if self.verbose:
                    print(f"Memory release too frequent, skipping this release")
                return {"status": "skipped", "reason": "too frequent"}
            
            self.last_free_time = current_time
            
            # Use silent mode to request memory release
            with open("/dev/null", "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                
                # Temporarily redirect standard output and error output
                if not self.verbose:
                    sys.stdout = devnull
                    sys.stderr = devnull
                
                try:
                    # Directly use request to call ComfyUI API
                    url = "http://127.0.0.1:8188/free"
                    payload = {
                        "unload_models": True, 
                        "free_memory": True
                    }
                    response = requests.post(url, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                    else:
                        result = {"error": f"Status code: {response.status_code}"}
                        
                    return result
                finally:
                    # Restore standard output and error output
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
        except Exception as e:
            if self.verbose:
                print(f"Error freeing memory: {str(e)}")
            return {"error": str(e)}

# Create global instance
memory_manager = MemoryManager()

class MemoryManagerNode:
    """
    Memory manager node
    Periodically calls API to free memory
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "interval_seconds": ("FLOAT", {"default": 60, "min": 0.1, "max": 60, "step": 0.1}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "manage_memory"
    CATEGORY = "🐳Pond/Tools"
    
    def manage_memory(self, enabled, interval_seconds, verbose):
        global memory_manager
        
        # Update settings
        memory_manager.interval = interval_seconds
        memory_manager.verbose = verbose
        
        # Start or stop based on settings
        if enabled and not memory_manager.enabled:
            memory_manager.start()
            status = f"Memory manager started, interval: {interval_seconds} seconds"
        elif not enabled and memory_manager.enabled:
            memory_manager.stop()
            status = "Memory manager stopped"
        elif enabled and memory_manager.enabled and memory_manager.interval != interval_seconds:
            # If interval changes, restart timer
            memory_manager.stop()
            memory_manager.interval = interval_seconds
            memory_manager.start()
            status = f"Memory manager updated, interval: {interval_seconds} seconds"
        elif enabled:
            status = f"Memory manager running, interval: {interval_seconds} seconds"
        else:
            status = "Memory manager stopped"
            
        return (status,)
    
    @classmethod
    def IS_CHANGED(cls, enabled, interval_seconds, verbose):
        # Re-execute node after settings change
        global memory_manager
        if enabled != memory_manager.enabled or interval_seconds != memory_manager.interval or verbose != memory_manager.verbose:
            return True
        return False

# Stop timer when script ends
import atexit

@atexit.register
def cleanup():
    global memory_manager
    if memory_manager.enabled:
        memory_manager.stop()

# Register node
NODE_CLASS_MAPPINGS = {
    "MemoryManagerNode": MemoryManagerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryManagerNode": "🐳Memory Manager"
}
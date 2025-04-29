# TensorRT.py
"""
TensorRT-based aimbot and triggerbot with overlay UI.
This tool uses AI inference to detect targets and assist with aiming in games.
"""

# Standard library imports
import os
import sys
import time
import json
import random
import logging
import argparse
import threading
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, Deque
from pathlib import Path
from collections import deque
from threading import Event

# Third-party imports
try:
    import cv2
    import numpy as np
    import mss
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pynput import keyboard
    from pynput.mouse import Listener as MouseListener, Button
    import tkinter as tk
    from tkinter import font, messagebox, ttk
except ImportError as e:
    print(f"Error: Required dependency not found: {e}")
    print("Please install required dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Local imports
try:
    from mouse_driver.ghub_mouse import LogiFck
except ImportError:
    print("Warning: LogiFck mouse driver not found. Mouse functionality will be limited.")
    # Create a fallback implementation
    class LogiFck:
        def __init__(self, dll_path=None):
            self.gmok = False
            print(f"Could not load mouse driver from {dll_path}")
        
        def move_relative(self, x, y):
            print(f"Mouse move: {x}, {y} (NOT IMPLEMENTED)")
            
        def press(self, button):
            print(f"Mouse press: {button} (NOT IMPLEMENTED)")
            
        def release(self, button):
            print(f"Mouse release: {button} (NOT IMPLEMENTED)")


# Argument parsing for command-line configuration
parser = argparse.ArgumentParser(description="TensorRT-based aim assistance tool")
parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                    default="INFO", help="Set the logging level")
parser.add_argument("--model-path", type=str, default="model_fp16_320.trt", 
                    help="Path to the TensorRT engine file")
parser.add_argument("--config-path", type=str, default="config.json",
                    help="Path to configuration file")
args = parser.parse_args()

# Configure logging based on command-line argument
numeric_level = getattr(logging, args.log_level, logging.INFO)
logging.basicConfig(level=numeric_level, 
                    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the application."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "aimbot": {
            "enabled": True,
            "key": "alt",
            "speed": 350,
            "target_choice": "neck",
            "smoothing_factor": 0.3,
            "prediction_time": 0.06,
            "panic_key": "end"
        },
        "triggerbot": {
            "enabled": True,
            "key": "alt",
            "min_click_delay": 0.09,
            "max_click_delay": 0.3,
            "min_release_delay": 0.1,
            "max_release_delay": 0.3
        },
        "display": {
            "boxes_enabled": True,
            "scanning_box_enabled": True,
            "input_width": 320,
            "input_height": 320
        },
        "keys": {
            "allowed_keys": ["alt", "shift", "caps lock", "x", "c", "v", "b", "z", "end"]
        },
        "aiming": {
            "offset_x": 2,
            "offset_y": 3,
            "target_offsets": {
                "head": 0.1,
                "neck": 0.2,
                "chest": 0.3,
                "legs": 0.8,
                "balls": 0.5
            }
        },
        "system": {
            "monitor_index": 0
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = config_path
        
        # Set default values first
        self._set_default_values()
        
        # Then try to load from file
        try:
            loaded_config = self.load_config()
            
            # Make sure we got a dictionary back
            if not isinstance(loaded_config, dict):
                logger.error(f"Invalid configuration format: {type(loaded_config)}")
                return
                
            # Update config with loaded values
            self.config = loaded_config
            
            # Extract values from config to individual properties
            self._extract_config_values()
        except Exception as e:
            logger.error(f"Error during configuration loading: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _set_default_values(self):
        """Set default values for configuration properties."""
        # Initialize config dictionary with defaults
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Extract default values to individual properties
        self.aimbot_enabled = True
        self.trigger_enabled = True
        self.boxes_enabled = True
        self.scanning_box_enabled = True
        self.aimbot_speed = 350
        self.aim_target_choice = "neck"
        self.aimbot_key = "alt"
        self.triggerbot_key = "alt"
        self.panic_key = "end"
        self.allowed_keys = ["alt", "shift", "caps lock", "x", "c", "v", "b", "z", "end"]
        self.input_width = 320
        self.input_height = 320
        self.offset_x = 2
        self.offset_y = 3
        self.target_offsets = {"head": 0.1, "neck": 0.2, "chest": 0.3, "legs": 0.8, "balls": 0.5}
        self.smoothing_factor = 0.3
        self.prediction_time = 0.06
        self.monitor_index = 0
        self.trigger_min_click_delay = 0.09
        self.trigger_max_click_delay = 0.3
        self.trigger_min_release_delay = 0.1
        self.trigger_max_release_delay = 0.3
    
    def _extract_config_values(self):
        """Extract values from config dictionary to individual properties."""
        try:
            # Only extract if we have a valid config dictionary
            if not isinstance(self.config, dict):
                logger.error("Cannot extract values: config is not a dictionary")
                return
                
            # Aimbot settings
            if "aimbot" in self.config and isinstance(self.config["aimbot"], dict):
                aimbot = self.config["aimbot"]
                if "enabled" in aimbot:
                    self.aimbot_enabled = aimbot["enabled"]
                if "key" in aimbot:
                    self.aimbot_key = aimbot["key"]
                if "speed" in aimbot:
                    self.aimbot_speed = aimbot["speed"]
                if "target_choice" in aimbot:
                    self.aim_target_choice = aimbot["target_choice"]
                if "smoothing_factor" in aimbot:
                    self.smoothing_factor = aimbot["smoothing_factor"]
                if "prediction_time" in aimbot:
                    self.prediction_time = aimbot["prediction_time"]
                if "panic_key" in aimbot:
                    self.panic_key = aimbot["panic_key"]
                    
            # Triggerbot settings
            if "triggerbot" in self.config and isinstance(self.config["triggerbot"], dict):
                trigger = self.config["triggerbot"]
                if "enabled" in trigger:
                    self.trigger_enabled = trigger["enabled"]
                if "key" in trigger:
                    self.triggerbot_key = trigger["key"]
                if "min_click_delay" in trigger:
                    self.trigger_min_click_delay = trigger["min_click_delay"]
                if "max_click_delay" in trigger:
                    self.trigger_max_click_delay = trigger["max_click_delay"]
                if "min_release_delay" in trigger:
                    self.trigger_min_release_delay = trigger["min_release_delay"]
                if "max_release_delay" in trigger:
                    self.trigger_max_release_delay = trigger["max_release_delay"]
                    
            # Display settings
            if "display" in self.config and isinstance(self.config["display"], dict):
                display = self.config["display"]
                if "boxes_enabled" in display:
                    self.boxes_enabled = display["boxes_enabled"]
                if "scanning_box_enabled" in display:
                    self.scanning_box_enabled = display["scanning_box_enabled"]
                if "input_width" in display:
                    self.input_width = display["input_width"]
                if "input_height" in display:
                    self.input_height = display["input_height"]
                    
            # Keys settings
            if "keys" in self.config and isinstance(self.config["keys"], dict):
                keys = self.config["keys"]
                if "allowed_keys" in keys:
                    self.allowed_keys = keys["allowed_keys"]
                    
            # Aiming settings
            if "aiming" in self.config and isinstance(self.config["aiming"], dict):
                aiming = self.config["aiming"]
                if "offset_x" in aiming:
                    self.offset_x = aiming["offset_x"]
                if "offset_y" in aiming:
                    self.offset_y = aiming["offset_y"]
                if "target_offsets" in aiming and isinstance(aiming["target_offsets"], dict):
                    self.target_offsets = aiming["target_offsets"]
                    
            # System settings
            if "system" in self.config and isinstance(self.config["system"], dict):
                system = self.config["system"]
                if "monitor_index" in system:
                    self.monitor_index = system["monitor_index"]
                    
        except Exception as e:
            logger.error(f"Error extracting config values: {e}")
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default if not exists.
        
        Returns:
            Dict: Configuration dictionary
        """
        # Check if file exists
        if not os.path.exists(self.config_path):
            logger.info(f"Configuration file not found, creating default: {self.config_path}")
            try:
                # Save default config
                with open(self.config_path, 'w') as f:
                    json.dump(self.DEFAULT_CONFIG, f, indent=4)
            except Exception as e:
                logger.error(f"Error creating default config file: {e}")
            return self.DEFAULT_CONFIG.copy()
        
        # Try to load existing file
        try:
            with open(self.config_path, 'r') as f:
                data = f.read().strip()
                if not data:  # Empty file
                    logger.warning(f"Empty config file: {self.config_path}")
                    return self.DEFAULT_CONFIG.copy()
                    
                loaded_config = json.loads(data)
                if not isinstance(loaded_config, dict):
                    logger.error(f"Invalid config format, expected dict, got: {type(loaded_config)}")
                    return self.DEFAULT_CONFIG.copy()
                    
                # Merge with defaults to ensure all keys exist
                return self._merge_with_defaults(loaded_config)
        except json.JSONDecodeError as e:
            logger.error(f"Config file is not valid JSON: {e}")
            return self.DEFAULT_CONFIG.copy()
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_with_defaults(self, user_config: Dict) -> Dict:
        """Recursively merge user config with defaults.
        
        Args:
            user_config: User configuration
            
        Returns:
            Dict: Merged configuration
        """
        if not isinstance(user_config, dict):
            return self.DEFAULT_CONFIG.copy()
            
        result = self.DEFAULT_CONFIG.copy()
        
        # Manual merge to ensure type checking at each level
        for key, default_value in result.items():
            if key in user_config:
                user_value = user_config[key]
                
                # Recursively merge nested dictionaries
                if isinstance(default_value, dict) and isinstance(user_value, dict):
                    result[key] = self._merge_dicts(default_value, user_value)
                else:
                    # For non-dict values, use user value (with type checking if needed)
                    result[key] = user_value
        
        return result
    
    def _merge_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """Merge two dictionaries.
        
        Args:
            dict1: First dictionary (base)
            dict2: Second dictionary (overrides)
            
        Returns:
            Dict: Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_dicts(result[key], value)
            else:
                # Override or add new keys
                result[key] = value
                
        return result
        
    def save_config(self, config: Dict = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save, uses current config if None
        """
        if config is None:
            config = self.config
            
        # Ensure config is a dictionary before saving
        if not isinstance(config, dict):
            logger.error(f"Cannot save config: not a dictionary, got {type(config)}")
            return
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
                logger.info(f"Configuration saved to {self.config_path}")
        except IOError as e:
            logger.error(f"Error saving config: {e}")
    
    def update(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value.
        
        Args:
            section: Section name
            key: Key to update
            value: New value
        """
        try:
            # Ensure we have a dictionary
            if not isinstance(self.config, dict):
                logger.error(f"Cannot update config: config is not a dictionary")
                return
                
            # Create section if it doesn't exist or if it's not a dictionary
            if section not in self.config or not isinstance(self.config[section], dict):
                self.config[section] = {}
                
            # Update the value
            self.config[section][key] = value
            
            # Update the direct access property
            if section == "aimbot":
                if key == "enabled":
                    self.aimbot_enabled = value
                elif key == "key":
                    self.aimbot_key = value
                elif key == "speed":
                    self.aimbot_speed = value
                elif key == "target_choice":
                    self.aim_target_choice = value
                elif key == "smoothing_factor":
                    self.smoothing_factor = value
                elif key == "prediction_time":
                    self.prediction_time = value
                elif key == "panic_key":
                    self.panic_key = value
            elif section == "triggerbot":
                if key == "enabled":
                    self.trigger_enabled = value
                elif key == "key":
                    self.triggerbot_key = value
                elif key == "min_click_delay":
                    self.trigger_min_click_delay = value
                elif key == "max_click_delay":
                    self.trigger_max_click_delay = value
                elif key == "min_release_delay":
                    self.trigger_min_release_delay = value
                elif key == "max_release_delay":
                    self.trigger_max_release_delay = value
            elif section == "display":
                if key == "boxes_enabled":
                    self.boxes_enabled = value
                elif key == "scanning_box_enabled":
                    self.scanning_box_enabled = value
                elif key == "input_width":
                    self.input_width = value
                elif key == "input_height":
                    self.input_height = value
            elif section == "system":
                if key == "monitor_index":
                    self.monitor_index = value
            else:
                logger.warning(f"Unknown config section: {section}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            import traceback
            logger.error(traceback.format_exc())


# Create configuration instance
config = Config(args.config_path)

# Runtime state (not persisted in config)
aimbot_active = False
trigger_active = False
detection_valid = False
left_mouse_pressed = False
smoothed_green_box_coords = None
smoothed_red_box_coords = None

# Exit and frame synchronization events
exit_event = threading.Event()
new_frame_event = threading.Event()
panic_triggered = threading.Event()

# Add a thread-safe flag for triggerbot
trigger_should_run = Event()

# Add a list to track all running threads for clean shutdown
all_threads = []

# Define TensorRT Engine Manager
class TRTEngineManager:
    """Manages TensorRT engine loading and inference."""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        self.input_shape = None
        self.output_shape = None
        self.input_name = None
        self.output_name = None
        self.d_input = None
        self.d_output = None
        self.h_input = None
        self.h_output = None
        self.start_event = None
        self.end_event = None

    def initialize(self) -> bool:
        try:
            if not os.path.exists(self.engine_path):
                logger.error(f"Engine file not found: {self.engine_path}")
                return False
            self.runtime = trt.Runtime(self.TRT_LOGGER)
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            if not engine_data:
                logger.error("Failed to read engine data")
                return False
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if not self.engine:
                logger.error("Failed to deserialize engine")
                return False
            self.context = self.engine.create_execution_context()
            if not self.context:
                logger.error("Failed to create execution context")
                return False
            # Always use TensorRT 8+ API
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)
            self.input_shape = self.engine.get_tensor_shape(self.input_name)
            self.output_shape = self.engine.get_tensor_shape(self.output_name)
            logger.info(f"Tensor shapes: input={self.input_shape}, output={self.output_shape}")
            self.stream = cuda.Stream()
            input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
            output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
            self.d_input = cuda.mem_alloc(input_size)
            self.d_output = cuda.mem_alloc(output_size)
            self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            self.start_event = cuda.Event()
            self.end_event = cuda.Event()
            # Warm-up
            for _ in range(3):
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            logger.info("TensorRT engine initialized successfully (TensorRT 8+ mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT engine: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_inference(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        try:
            # OPTIMIZATION: Manual preprocessing is faster than blobFromImage
            # Resize the frame
            resized_frame = cv2.resize(frame, (config.input_width, config.input_height))
            
            # Normalize pixel values to [0,1]
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            
            # Convert from HWC to CHW format (Height, Width, Channels) -> (Channels, Height, Width)
            chw_frame = np.transpose(normalized_frame, (2, 0, 1))
            
            # Add batch dimension
            blob = chw_frame[np.newaxis]
            
            # Flatten for TensorRT input
            blob_flat = blob.reshape(-1)
            
            if blob_flat.shape[0] != np.prod(self.input_shape):
                logger.error(f"Input shape mismatch: got {blob_flat.shape[0]}, expected {np.prod(self.input_shape)}")
                return None, 0.0
            if blob_flat.dtype != np.float32:
                logger.error(f"Input dtype mismatch: got {blob_flat.dtype}, expected float32")
                return None, 0.0
                
            np.copyto(self.h_input, blob_flat)
            self.start_event.record(self.stream)
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            
            # FIX: Set tensor addresses before every inference (TensorRT 8+ requirement)
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.end_event.record(self.stream)
            self.stream.synchronize()
            inference_time = self.end_event.time_till(self.start_event) / 1000.0
            return self.h_output.reshape(self.output_shape), inference_time
        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, 0.0
    
    def cleanup(self) -> None:
        logger.info("TensorRT resources released")

# Initialize TensorRT engine
try:
    engine_manager = TRTEngineManager(args.model_path)
    if not engine_manager.initialize():
        logger.error("Failed to initialize TensorRT engine. Exiting.")
        sys.exit(1)
except Exception as e:
    logger.error(f"TensorRT initialization error: {e}")
    logger.error("Please make sure the TensorRT engine file exists and is valid.")
    sys.exit(1)

# Screen and bbox related functions
def get_screen_info():
    """Get information about available monitors.
    
    Returns:
        tuple: (screen_width, screen_height, monitor)
    """
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            monitor_index = min(config.monitor_index, len(monitors) - 1)
            monitor = monitors[monitor_index + 1] if monitor_index >= 0 else monitors[0]
            screen_width = monitor['width']
            screen_height = monitor['height']
        return screen_width, screen_height, monitor
    except Exception as e:
        logger.error(f"Error getting screen info: {e}")
        # Return sensible defaults if we can't get screen info
        return 1920, 1080, {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def get_center_bbox(width, height, monitor):
    """Calculate a centered bounding box for screen capture.
    
    Args:
        width: Box width
        height: Box height
        monitor: Monitor information
        
    Returns:
        dict: Bounding box for screen capture
    """
    center_x = monitor['width'] // 2
    center_y = monitor['height'] // 2
    top_left_x = center_x - (width // 2)
    top_left_y = center_y - (height // 2)
    return {
        'top': top_left_y + monitor['top'],
        'left': top_left_x + monitor['left'],
        'width': width,
        'height': height
    }

# Initialize screen information
screen_width, screen_height, monitor = get_screen_info()
screen_bbox = get_center_bbox(config.input_width, config.input_height, monitor)
screen_center_x = monitor['left'] + screen_width // 2
screen_center_y = monitor['top'] + screen_height // 2

logger.info(f"Screen Width: {screen_width}, Screen Height: {screen_height}")
logger.info(f"Screen BBox: {screen_bbox}")
logger.info(f"Screen Center: ({screen_center_x}, {screen_center_y})")

# Synchronization primitives and frame container
frame_lock = threading.Lock()
boxes_lock = threading.Lock()
click_lock = threading.Lock()
frame_queue = deque(maxlen=1)

latest_frame = None
click_in_progress = False

green_box_coords = None
red_box_coords = None

# Initialize mouse controller
try:
    mouse_controller = LogiFck('ghub_mouse.dll')
    if not mouse_controller.gmok:
        logger.warning("Mouse controller initialized but not operational.")
except Exception as e:
    logger.error(f"Failed to initialize mouse controller: {e}")
    mouse_controller = LogiFck()  # Use stub implementation

# UI Classes for better organization
class OverlayWindow:
    """Transparent overlay window for displaying bounding boxes."""
    
    def __init__(self, screen_width, screen_height):
        """Initialize the overlay window.
        
        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen
        """
        self.root = tk.Tk()
        self.title = "Overlay"
        self.root.title(self.title)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.9)
        self.root.attributes('-fullscreen', True)
        self.root.attributes("-transparentcolor", "white")
        self.root.configure(background='white')
        
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, 
                              bg='white', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Track previous box positions for smoother interpolation
        self.prev_green_box = None
        self.prev_red_box = None
        self.box_velocity = {"green": [0, 0, 0, 0], "red": [0, 0, 0, 0]}
        self.velocity_decay = 0.8  # How quickly velocity decreases
        
    def update_scanning_box(self, enable, x1, y1, x2, y2):
        """Update the scanning box display.
        
        Args:
            enable: Whether to show the scanning box
            x1, y1, x2, y2: Coordinates of the scanning box
        """
        if enable:
            scanning_box = (x1, y1, x2, y2)
            if not self.canvas.find_withtag("scanning_box"):
                self.canvas.create_rectangle(scanning_box, outline='yellow', width=2, tag="scanning_box")
            else:
                self.canvas.coords("scanning_box", *scanning_box)
        else:
            self.canvas.delete("scanning_box")
            
    def smooth_box_movement(self, box_type, new_box, prev_box, smoothing_factor=0.3):
        """Apply advanced smoothing to box movements with velocity prediction.
        
        Args:
            box_type: Type of box ("green" or "red")
            new_box: New box coordinates
            prev_box: Previous box coordinates
            smoothing_factor: Smoothing factor (lower = smoother)
            
        Returns:
            Smoothed box coordinates
        """
        if new_box is None:
            if prev_box is None:
                return None
                
            # Target lost - fade out box by shrinking toward center
            cx = (prev_box[0] + prev_box[2]) / 2
            cy = (prev_box[1] + prev_box[3]) / 2
            
            # Use velocity to predict fade-out direction
            velocity = self.box_velocity[box_type]
            # Apply velocity decay
            self.box_velocity[box_type] = [v * self.velocity_decay for v in velocity]
            
            # Calculate new coordinates with shrinking and momentum
            width = abs(prev_box[2] - prev_box[0]) * 0.9  # Shrink by 10%
            height = abs(prev_box[3] - prev_box[1]) * 0.9  # Shrink by 10%
            
            # If box is very small, remove it
            if width < 3 or height < 3:
                return None
                
            # Apply velocity to center position
            cx += velocity[0] + velocity[2]
            cy += velocity[1] + velocity[3]
            
            return (
                cx - width / 2,
                cy - height / 2, 
                cx + width / 2,
                cy + height / 2
            )
            
        if prev_box is None:
            # First appearance - no smoothing
            return new_box
            
        # Calculate velocity (movement delta)
        velocity = [
            new_box[0] - prev_box[0],
            new_box[1] - prev_box[1],
            new_box[2] - prev_box[2],
            new_box[3] - prev_box[3]
        ]
        
        # Update velocity with exponential smoothing
        self.box_velocity[box_type] = [
            0.3 * v + 0.7 * old_v
            for v, old_v in zip(velocity, self.box_velocity[box_type])
        ]
        
        # Apply smoothing with velocity prediction
        result = []
        for i in range(4):
            # Predicted position based on velocity
            prediction = prev_box[i] + self.box_velocity[box_type][i]
            # Blend between prediction and actual position
            smooth_val = smoothing_factor * new_box[i] + (1 - smoothing_factor) * prediction
            result.append(smooth_val)
            
        return tuple(result)
    
    def update_detection_boxes(self, enable, green_box=None, red_box=None):
        """Update the detection bounding boxes with enhanced smoothing.
        
        Args:
            enable: Whether to show the bounding boxes
            green_box: Coordinates of the green bounding box (outer)
            red_box: Coordinates of the red bounding box (aim point)
        """
        if enable:
            # Apply advanced smoothing for green box
            smoothed_green = self.smooth_box_movement("green", green_box, self.prev_green_box)
            self.prev_green_box = smoothed_green
            
            # Apply advanced smoothing for red box
            smoothed_red = self.smooth_box_movement("red", red_box, self.prev_red_box)
            self.prev_red_box = smoothed_red
            
            if smoothed_green is not None:
                if not self.canvas.find_withtag("green_box"):
                    self.canvas.create_rectangle(smoothed_green, outline='green', width=2, tag="green_box")
                else:
                    self.canvas.coords("green_box", *smoothed_green)
            else:
                self.canvas.delete("green_box")
                
            if smoothed_red is not None:
                if not self.canvas.find_withtag("red_box"):
                    self.canvas.create_rectangle(smoothed_red, outline='red', width=2, tag="red_box")
                else:
                    self.canvas.coords("red_box", *smoothed_red)
            else:
                self.canvas.delete("red_box")
        else:
            self.canvas.delete("green_box")
            self.canvas.delete("red_box")
            
    def get_root(self):
        """Get the root window.
        
        Returns:
            The root Tk window
        """
        return self.root


class ControlPanel:
    """Modern, compact, tabbed, and scrollable control panel for configuring the application."""
    def __init__(self, config, save_callback):
        self.config = config
        self.save_callback = save_callback
        self.window = tk.Toplevel()
        self.window.title("Control Panel")
        self.window.geometry("420x600+50+50")
        self.window.configure(background='#2d2d30')
        self.window.minsize(420, 400)
        self.window.resizable(True, True)
        self.modern_font = font.Font(family="Segoe UI", size=10, weight="normal")
        # Available keys for keybind selection
        self.available_keys = self.config.allowed_keys
        
        # Configure styles for ttk widgets
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#2d2d30', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2d2d30', foreground='#00FF00', 
                      padding=[10, 5], font=('Segoe UI', 10))
        style.map('TNotebook.Tab', background=[('selected', '#3d3d42')], 
                foreground=[('selected', '#00FF00')])
        
        # Create the notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create control panel
        self.create_control_panel()

    def create_control_panel(self):
        # --- General Tab ---
        general_tab = tk.Frame(self.notebook, bg="#2d2d30")
        self.notebook.add(general_tab, text="General")

        # --- Aimbot Tab ---
        aimbot_tab = tk.Frame(self.notebook, bg="#2d2d30")
        self.notebook.add(aimbot_tab, text="Aimbot")

        # --- Triggerbot Tab ---
        trigger_tab = tk.Frame(self.notebook, bg="#2d2d30")
        self.notebook.add(trigger_tab, text="Triggerbot")

        # --- System Tab ---
        system_tab = tk.Frame(self.notebook, bg="#2d2d30")
        self.notebook.add(system_tab, text="System")

        # --- General Tab Content ---
        self._add_checkbox(general_tab, "Enable Aimbot", self.config.aimbot_enabled, self._update_aimbot, 0)
        self._add_checkbox(general_tab, "Enable Triggerbot", self.config.trigger_enabled, self._update_trigger, 1)
        self._add_checkbox(general_tab, "Show Bounding Boxes", self.config.boxes_enabled, self._update_boxes, 2)
        self._add_checkbox(general_tab, "Show Scanning Box", self.config.scanning_box_enabled, self._update_scan_box, 3)

        # --- Aimbot Tab Content ---
        self._add_slider(aimbot_tab, "Aimbot Speed", 1, 1000, self.config.aimbot_speed, self._update_speed, 0)
        self._add_slider(aimbot_tab, "Smoothing", 0.0, 1.0, self.config.smoothing_factor, self._update_smoothing, 1, resolution=0.01)
        self._add_slider(aimbot_tab, "Prediction Time (s)", 0.0, 0.2, self.config.prediction_time, self._update_prediction, 2, resolution=0.01)
        self._add_option_menu(aimbot_tab, "Target", ["head", "neck", "chest", "closest"], self.config.aim_target_choice, self._update_target, 3)
        self._add_option_menu(aimbot_tab, "Aimbot Key", self.available_keys, self.config.aimbot_key, self._update_keys, 4, is_aimbot=True)

        # --- Triggerbot Tab Content ---
        self._add_option_menu(trigger_tab, "Triggerbot Key", self.available_keys, self.config.triggerbot_key, self._update_keys, 0, is_trigger=True)
        self._add_slider(trigger_tab, "Min Click Delay (s)", 0.01, 0.5, self.config.trigger_min_click_delay, self._update_trigger_min_click, 1, resolution=0.01)
        self._add_slider(trigger_tab, "Max Click Delay (s)", 0.01, 0.5, self.config.trigger_max_click_delay, self._update_trigger_max_click, 2, resolution=0.01)
        self._add_slider(trigger_tab, "Min Release Delay (s)", 0.01, 0.5, self.config.trigger_min_release_delay, self._update_trigger_min_release, 3, resolution=0.01)
        self._add_slider(trigger_tab, "Max Release Delay (s)", 0.01, 0.5, self.config.trigger_max_release_delay, self._update_trigger_max_release, 4, resolution=0.01)

        # --- System Tab Content ---
        self._add_spinbox(system_tab, "Monitor Index", 0, 4, self.config.monitor_index, self._update_monitor, 0)
        self._add_spinbox(system_tab, "Input Width", 64, 1920, self.config.input_width, self._update_input_width, 1)
        self._add_spinbox(system_tab, "Input Height", 64, 1920, self.config.input_height, self._update_input_height, 2)

        # --- Bottom Buttons --- (removed Save Settings button since auto-save is enabled)
        btn_frame = tk.Frame(self.window, bg="#2d2d30")
        btn_frame.pack(fill="x", pady=5)
        # Status label to show auto-saving is enabled
        status_label = tk.Label(btn_frame, text="Auto-save enabled", bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        status_label.pack(side="left", padx=10)
        exit_btn = tk.Button(btn_frame, text="Exit", command=self._exit_program, bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        exit_btn.pack(side="right", padx=10)
        help_btn = tk.Button(btn_frame, text="?", command=self._show_help, bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        help_btn.pack(side="right", padx=10)
        
        return self.window

    def _add_checkbox(self, parent, label, value, command, row):
        var = tk.BooleanVar(value=value)
        
        # Direct command callback that immediately updates when checked/unchecked
        def on_checkbox_change(*args):
            command(var.get())
        
        var.trace_add("write", on_checkbox_change)  # Add trace to detect changes
        
        frame = tk.Frame(parent, bg="#2d2d30")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        cb = tk.Checkbutton(frame, text=label, var=var, 
                            bg="#2d2d30", fg="#00FF00", selectcolor="#1e1e1e",
                            activebackground="#2d2d30", font=self.modern_font)
        cb.pack(anchor="w")
        
        # Store the variable for later access
        setattr(self, f"{label.replace(' ', '_').lower()}_var", var)

    def _add_slider(self, parent, label, min_val, max_val, value, command, row, resolution=1):
        frame = tk.Frame(parent, bg="#2d2d30")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        tk.Label(frame, text=label, bg="#2d2d30", fg="#00FF00", font=self.modern_font).pack(anchor="w")
        
        # Use a container frame for the slider and value label
        slider_frame = tk.Frame(frame, bg="#2d2d30")
        slider_frame.pack(fill="x")
        
        # Create the slider
        var = tk.DoubleVar(value=value)
        
        # Value label to show current value
        value_label = tk.Label(slider_frame, textvariable=var, width=5, 
                              bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        value_label.pack(side="right")
        
        slider = tk.Scale(slider_frame, from_=min_val, to=max_val, resolution=resolution,
                         orient="horizontal", variable=var, showvalue=False,
                         bg="#2d2d30", fg="#00FF00", highlightthickness=0,
                         troughcolor="#1e1e1e", activebackground="#00FF00")
        slider.pack(fill="x", side="left", expand=True)
        
        # Direct command callback that immediately updates when value changes
        def on_slider_change(*args):
            command(var.get())
        
        var.trace_add("write", on_slider_change)  # Add trace to detect changes
        
        # Store the variable for later access
        setattr(self, f"{label.replace(' ', '_').lower()}_var", var)

    def _add_option_menu(self, parent, label, options, value, command, row, is_aimbot=False, is_trigger=False):
        frame = tk.Frame(parent, bg="#2d2d30")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        tk.Label(frame, text=label, bg="#2d2d30", fg="#00FF00", font=self.modern_font).pack(anchor="w")
        var = tk.StringVar(value=value)
        
        # Direct command callback that immediately updates when selection changes
        def on_option_select(*args):
            selected_value = var.get()
            if is_aimbot:
                command(selected_value, is_aimbot=True)
            elif is_trigger:
                command(selected_value, is_trigger=True)
            else:
                command(selected_value)
        
        var.trace_add("write", on_option_select)  # Add trace to detect changes
        
        om = tk.OptionMenu(frame, var, *options)
        om.config(bg="#2d2d30", fg="#00FF00", font=self.modern_font, activebackground="#2d2d30")
        om["menu"].config(bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        om.pack(fill="x")
        
        if is_aimbot:
            self.aimbot_key_var = var
        elif is_trigger:
            self.triggerbot_key_var = var
        else:
            setattr(self, f"{label.replace(' ', '_').lower()}_var", var)

    def _add_spinbox(self, parent, label, from_, to, value, command, row):
        frame = tk.Frame(parent, bg="#2d2d30")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        tk.Label(frame, text=label, bg="#2d2d30", fg="#00FF00", font=self.modern_font).pack(anchor="w")
        var = tk.IntVar(value=value)
        
        # Add trace to update when value changes
        def on_value_change(*args):
            try:
                # Get the current value and update
                current_val = var.get()
                command(current_val)
            except:
                pass  # In case the value is temporarily invalid
                
        var.trace_add("write", on_value_change)
        
        sb = tk.Spinbox(frame, from_=from_, to=to, textvariable=var, width=8, 
                       font=self.modern_font, bg="#2d2d30", fg="#00FF00")
        sb.pack(anchor="w")
        setattr(self, f"{label.replace(' ', '_').lower()}_var", var)

    # --- Update methods for each control ---
    def _update_aimbot(self, value):
        self.config.update("aimbot", "enabled", value)
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_trigger(self, value):
        self.config.update("triggerbot", "enabled", value)
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_boxes(self, value):
        self.config.update("display", "boxes_enabled", value)
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_scan_box(self, value):
        self.config.update("display", "scanning_box_enabled", value)
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_speed(self, value):
        self.config.update("aimbot", "speed", int(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_smoothing(self, value):
        self.config.update("aimbot", "smoothing_factor", float(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_prediction(self, value):
        self.config.update("aimbot", "prediction_time", float(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_target(self, value):
        self.config.update("aimbot", "target_choice", value)
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_keys(self, value, is_aimbot=False, is_trigger=False):
        if is_aimbot:
            self.config.update("aimbot", "key", value)
        elif is_trigger:
            self.config.update("triggerbot", "key", value)
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_trigger_min_click(self, value):
        self.config.update("triggerbot", "min_click_delay", float(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_trigger_max_click(self, value):
        self.config.update("triggerbot", "max_click_delay", float(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_trigger_min_release(self, value):
        self.config.update("triggerbot", "min_release_delay", float(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_trigger_max_release(self, value):
        self.config.update("triggerbot", "max_release_delay", float(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_monitor(self, value):
        self.config.update("display", "monitor_index", int(value))
        self.config.save_config()
        self._show_saved_notification()
    
    def _update_input_width(self, value):
        global screen_bbox, screen_center_x, screen_center_y
        self.config.update("display", "input_width", int(value))
        self.config.save_config()
        
        # Update the screen bbox to match the new width
        screen_width, screen_height, monitor = get_screen_info()
        screen_bbox = get_center_bbox(self.config.input_width, self.config.input_height, monitor)
        screen_center_x = monitor['left'] + screen_width // 2
        screen_center_y = monitor['top'] + screen_height // 2
        
        self._show_saved_notification()
    
    def _update_input_height(self, value):
        global screen_bbox, screen_center_x, screen_center_y
        self.config.update("display", "input_height", int(value))
        self.config.save_config()
        
        # Update the screen bbox to match the new height
        screen_width, screen_height, monitor = get_screen_info()
        screen_bbox = get_center_bbox(self.config.input_width, self.config.input_height, monitor)
        screen_center_x = monitor['left'] + screen_width // 2
        screen_center_y = monitor['top'] + screen_height // 2
        
        self._show_saved_notification()

    def _save_settings(self):
        self.config.save_config()
    def _exit_program(self):
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            logger.info("Exit requested by user from Control Panel")
            # Set exit event first to signal all threads to stop
            exit_event.set()
            # Stop any active triggerbot
            trigger_should_run.clear()
            if hasattr(triggerbot_controller, 'stop'):
                triggerbot_controller.stop()
            # Release mouse if held down
            if hasattr(mouse_controller, 'release'):
                try:
                    mouse_controller.release(1)
                except:
                    pass
            # Destroy windows
            self.window.destroy()
    def _show_help(self):
        help_text = """
        AI AimShit & Trigger Sot Help\n\nThis tool uses AI to detect targets and provides aim assistance and triggerbot functionality.\n\nAimbot: Hold the configured key to automatically move your aim toward detected targets.\nTriggerbot: Hold the configured key to automatically fire when your crosshair is over a target.\n\nWARNING: Using this tool in online competitive games may violate terms of service and result in a ban.\n.
        """
        help_window = tk.Toplevel(self.window)
        help_window.title("Help")
        help_window.geometry("500x400+100+100")
        help_window.configure(background='#2d2d30')
        help_label = tk.Label(help_window, text=help_text, justify="left", bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        help_label.pack(padx=20, pady=20)
        close_button = tk.Button(help_window, text="Close", command=help_window.destroy, bg="#2d2d30", fg="#00FF00", font=self.modern_font)
        close_button.pack(pady=10)
    def get_window(self):
        return self.window

    def _show_saved_notification(self):
        """Show a notification that settings were saved."""
        self.window.title(f"Control Panel - Settings Saved")
        self.window.after(1000, lambda: self.window.title("Control Panel"))
    

def overlay_window():
    """Create and manage the overlay UI."""
    global green_box_coords, red_box_coords, smoothed_green_box_coords, smoothed_red_box_coords
    
    # Create the overlay window
    overlay = OverlayWindow(screen_width, screen_height)
    
    # Create the control panel
    control_panel = ControlPanel(config, config.save_config)
    
    # --- SMOOTHING STATE ---
    fade_factor = 0.15  # How quickly to fade out boxes when lost (0.1-0.2 is good)
    
    def update_loop():
        """Update the overlay display."""
        global smoothed_green_box_coords, smoothed_red_box_coords
        try:
            # Update scanning box
            if config.scanning_box_enabled:
                overlay.update_scanning_box(
                    True,
                    screen_center_x - (config.input_width // 2),
                    screen_center_y - (config.input_height // 2),
                    screen_center_x + (config.input_width // 2),
                    screen_center_y + (config.input_height // 2)
                )
            else:
                overlay.update_scanning_box(False, 0, 0, 0, 0)
            
            # Get the latest box coordinates with thread safety
            with boxes_lock:
                target_green = green_box_coords
                target_red = red_box_coords
            
            # Update detection boxes with advanced smoothing
            overlay.update_detection_boxes(
                config.boxes_enabled,
                target_green,
                target_red
            )
            
            # Update the UI
            overlay.get_root().update_idletasks()
            
            # OPTIMIZATION: Use higher refresh rate (120fps) for smoother animations
            if not exit_event.is_set():
                overlay.get_root().after(8, update_loop)  # ~120 FPS (1000/8 ≈ 120)
            else:
                overlay.get_root().destroy()
                
        except Exception as e:
            logger.error(f"UI update error: {e}")
            
    # Start the update loop with higher refresh rate
    overlay.get_root().after(8, update_loop)
    
    # --- OVERRIDE CLOSE BUTTON TO EXIT FULLY ---
    def on_close():
        logger.info("Exit requested by user from window close button")
        # Set exit event to signal all threads to stop
        exit_event.set()
        # Stop any active triggerbot
        trigger_should_run.clear()
        if hasattr(triggerbot_controller, 'stop'):
            triggerbot_controller.stop()
        # Release mouse if held down
        if hasattr(mouse_controller, 'release'):
            try:
                mouse_controller.release(1)
            except:
                pass
        try:
            # Destroy the windows
            if hasattr(control_panel, 'window') and control_panel.window:
                control_panel.window.destroy()
            overlay.get_root().destroy()
        except:
            pass
    overlay.get_root().protocol("WM_DELETE_WINDOW", on_close)
    control_panel.get_window().protocol("WM_DELETE_WINDOW", on_close)
    
    # Start the main loop
    overlay.get_root().mainloop()

# Create thread for overlay window
overlay_thread = threading.Thread(target=overlay_window, daemon=True, name="overlay_thread")
all_threads.append(overlay_thread)
overlay_thread.start()

# Modularized aimbot and mouse control
class AimbotController:
    """Controller for aimbot functionality and mouse movement."""
    
    def __init__(self, config, mouse_controller):
        """Initialize the aimbot controller.
        
        Args:
            config: Configuration instance
            mouse_controller: Mouse controller instance
        """
        self.config = config
        self.mouse_controller = mouse_controller
        
        # Previous mouse position tracking for smoothing
        self.last_move_x = 0
        self.last_move_y = 0
        
        # OPTIMIZATION: Expanded history storage for better tracking
        self.position_history = deque(maxlen=8)  # Store last 8 positions for advanced smoothing
        
        # OPTIMIZATION: Better velocity tracking with separate components
        self.velocity_x = 0  # Track target velocity for prediction
        self.velocity_y = 0
        self.velocities = deque(maxlen=5)  # Store recent velocities for smoothing
        
        self.last_target_x = None
        self.last_target_y = None
        self.last_update_time = time.time()
        
        # OPTIMIZATION: Adaptive parameters
        self.adaptive_sharpness = 1.5  # Dynamically adjusted based on distance
        self.distance_scale_factor = 1.0  # Scale factor adjusted based on distance
        
        # OPTIMIZATION: Pre-compute lookup tables for common calculations
        self._init_lookup_tables()
        
    def _init_lookup_tables(self):
        """Initialize lookup tables for faster calculation."""
        # Pre-compute easing function values
        self.ease_lookup = {}
        for i in range(101):  # 0 to 100
            t = i / 100.0
            for sharpness in [1.2, 1.5, 1.8, 2.0]:
                key = (i, int(sharpness * 10))
                self.ease_lookup[key] = t ** (1.0 / sharpness)
        
    def custom_ease(self, t, sharpness=2.0):
        """Improved easing function with adjustable curve and lookup acceleration.
        
        Args:
            t: Input value between 0 and 1
            sharpness: Sharpness of the curve, lower = sharper
            
        Returns:
            Eased value between 0 and 1
        """
        # OPTIMIZATION: Use lookup table for common values
        t_int = int(t * 100)
        sharpness_int = int(sharpness * 10)
        
        key = (t_int, sharpness_int)
        if key in self.ease_lookup:
            return self.ease_lookup[key]
            
        # Fall back to calculation for non-cached values
        return t ** (1.0 / sharpness)
        
    def move_mouse_towards_target(self, delta_x, delta_y, max_speed=None):
        """Move the mouse towards a target position.
        
        Args:
            delta_x: X distance to target
            delta_y: Y distance to target
            max_speed: Maximum speed to move
        """
        if max_speed is None:
            max_speed = self.config.aimbot_speed
            
        # Skip tiny movements that aren't noticeable
        if abs(delta_x) < 0.2 and abs(delta_y) < 0.2:
            return
            
        if not self.mouse_controller.gmok:
            logger.error("Mouse controller not initialized properly.")
            return
        
        # OPTIMIZATION: Improved velocity calculation and prediction
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        
        if self.last_target_x is not None and time_delta > 0:
            # Calculate instantaneous velocity (pixels per second)
            current_velocity_x = (delta_x - self.last_target_x) / time_delta
            current_velocity_y = (delta_y - self.last_target_y) / time_delta
            
            # Store velocity for history
            self.velocities.append((current_velocity_x, current_velocity_y))
            
            # OPTIMIZATION: Improved velocity smoothing using median filter
            # Median is more robust to outliers than exponential moving average
            if len(self.velocities) >= 3:
                velx_list = [v[0] for v in self.velocities]
                vely_list = [v[1] for v in self.velocities]
                
                # Use median for better outlier rejection
                self.velocity_x = sorted(velx_list)[len(velx_list)//2]
                self.velocity_y = sorted(vely_list)[len(vely_list)//2]
            else:
                # Fall back to EMA when not enough samples
                velocity_smoothing = 0.7
                self.velocity_x = velocity_smoothing * current_velocity_x + (1 - velocity_smoothing) * self.velocity_x
                self.velocity_y = velocity_smoothing * current_velocity_y + (1 - velocity_smoothing) * self.velocity_y
            
            # OPTIMIZATION: Improved motion prediction
            prediction_time = self.config.prediction_time
            
            # Scale prediction based on velocity consistency
            if len(self.velocities) >= 3:
                # Calculate velocity variance (lower = more consistent)
                velx_var = np.var([v[0] for v in self.velocities])
                vely_var = np.var([v[1] for v in self.velocities])
                
                # More consistent velocity = more aggressive prediction
                consistency = 1.0 / (1.0 + velx_var + vely_var)
                prediction_weight = min(1.0, consistency * 2.0)
                
                # Only predict if velocity is consistent and not too high (prevents overshooting)
                if abs(self.velocity_x) < 500 and abs(self.velocity_y) < 500:
                    pred_x = self.velocity_x * prediction_time * prediction_weight
                    pred_y = self.velocity_y * prediction_time * prediction_weight
                    
                    # Apply prediction
                    delta_x += pred_x
                    delta_y += pred_y
        
        # Store current position for next frame's velocity calculation
        self.last_target_x = delta_x
        self.last_target_y = delta_y
        self.last_update_time = current_time
            
        # OPTIMIZATION: Apply the offset adjustment based on distance
        # Closer distances require more precise offset adjustment
        distance = np.hypot(delta_x, delta_y)
        offset_scale = 1.0
        if distance < 10:
            offset_scale = 0.5  # Reduce offset for very close targets
        elif distance > 100:
            offset_scale = 1.5  # Increase offset for far targets
            
        delta_x += self.config.offset_x * offset_scale
        delta_y += self.config.offset_y * offset_scale
        
        # Calculate the distance to target
        distance = np.hypot(delta_x, delta_y)
        if distance == 0:
            return
            
        # Calculate direction vectors
        direction_x = delta_x / distance
        direction_y = delta_y / distance
        
        # OPTIMIZATION: More adaptive dynamic speed calculation
        if distance < 5:
            # Very close - very precise movement (microadjustments)
            speed_factor = 0.25
            self.adaptive_sharpness = 2.0  # Gentler curve
        elif distance < 15:
            # Close - precise movement
            speed_factor = 0.5
            self.adaptive_sharpness = 1.8
        elif distance < 50:
            # Medium distance
            speed_factor = 0.8
            self.adaptive_sharpness = 1.5
        elif distance < 100:
            # Medium-far distance
            speed_factor = 1.0
            self.adaptive_sharpness = 1.3
        else:
            # Far away - aggressive movement
            speed_factor = 1.3
            self.adaptive_sharpness = 1.2  # Sharper curve
            
        # Store speed factor for future reference
        self.distance_scale_factor = speed_factor
        
        # Calculate the base move distance using our custom easing function
        normalized_distance = min(distance / max_speed, 1.0)
        ease_value = self.custom_ease(normalized_distance, sharpness=self.adaptive_sharpness)
        
        # Apply the speed factor and calculate actual move distance
        move_distance = ease_value * max_speed * speed_factor
        
        # Calculate raw movement values
        raw_move_x = direction_x * move_distance
        raw_move_y = direction_y * move_distance
        
        # OPTIMIZATION: Adaptive smoothing based on distance
        # Less smoothing for distant targets, more for close targets
        smoothing_weight = 0.7
        if distance < 20:
            smoothing_weight = 0.6  # More smoothing for close targets
        elif distance > 100:
            smoothing_weight = 0.8  # Less smoothing for far targets
            
        move_x = int(smoothing_weight * raw_move_x + (1 - smoothing_weight) * self.last_move_x)
        move_y = int(smoothing_weight * raw_move_y + (1 - smoothing_weight) * self.last_move_y)
        
        # OPTIMIZATION: Ensure small movements aren't lost with intelligent directional bias
        if abs(delta_x) > 0.5 and move_x == 0:
            move_x = 1 if delta_x > 0 else -1
            
            # Add a small boost to really small movements to ensure they register
            if abs(delta_x) < 2.0 and abs(raw_move_x) > 0.1:
                move_x = 2 if delta_x > 0 else -2
                
        if abs(delta_y) > 0.5 and move_y == 0:
            move_y = 1 if delta_y > 0 else -1
            
            # Add a small boost to really small movements to ensure they register
            if abs(delta_y) < 2.0 and abs(raw_move_y) > 0.1:
                move_y = 2 if delta_y > 0 else -2
        
        # Store current movement for next frame's smoothing
        self.last_move_x = move_x
        self.last_move_y = move_y
        
        # Add to position history for advanced analysis
        self.position_history.append((delta_x, delta_y))
        
        # Finally move the mouse
        self.mouse_controller.move_relative(move_x, move_y)


class TriggerBotController:
    """Controller for triggerbot functionality."""
    
    def __init__(self, config, mouse_controller):
        self.config = config
        self.mouse_controller = mouse_controller
        self.click_in_progress = False
        self.click_lock = threading.Lock()
        self.trigger_thread = None
        self.should_stop = threading.Event()
        
    def trigger_action(self, detection_valid):
        """Perform a mouse click when a target is detected."""
        with self.click_lock:
            if self.click_in_progress:
                return
            self.click_in_progress = True
        self.should_stop.clear()
        try:
            while self.config.trigger_enabled and trigger_active and detection_valid and not exit_event.is_set() and not self.should_stop.is_set():
                if not trigger_should_run.is_set():
                    break
                if left_mouse_pressed:
                    time.sleep(0.001)
                    continue
                self.mouse_controller.press(1)
                click_duration = random.uniform(
                    self.config.trigger_min_click_delay, 
                    self.config.trigger_max_click_delay
                )
                click_duration += random.uniform(-0.01, 0.01)
                time.sleep(max(0.01, click_duration))
                self.mouse_controller.release(1)
                release_duration = random.uniform(
                    self.config.trigger_min_release_delay, 
                    self.config.trigger_max_release_delay
                )
                release_duration += random.uniform(-0.01, 0.01)
                time.sleep(max(0.01, release_duration))
        except Exception as e:
            logger.error(f"Trigger bot error: {e}")
        finally:
            try:
                self.mouse_controller.release(1)
            except:
                pass
            with self.click_lock:
                self.click_in_progress = False
    def stop(self):
        self.should_stop.set()

# Create aimbot and triggerbot controllers
aimbot_controller = AimbotController(config, mouse_controller)
triggerbot_controller = TriggerBotController(config, mouse_controller)
click_lock = triggerbot_controller.click_lock
click_in_progress = False


def key_matches(key, target_str):
    """Check if a key matches a target string.
    
    Args:
        key: The key to check
        target_str: The target string to match
        
    Returns:
        bool: Whether the key matches the target
    """
    target_str = target_str.lower()
    special_keys = {
        "alt": [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r],
        "shift": [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r],
        "caps lock": [keyboard.Key.caps_lock],
        "end": [keyboard.Key.end]
    }
    
    if target_str in special_keys:
        return key in special_keys[target_str]
    else:
        if hasattr(key, 'char') and key.char is not None:
            return key.char.lower() == target_str
        return False


def on_press(key):
    """Handle key press events.
    
    Args:
        key: The key that was pressed
    """
    global aimbot_active, trigger_active
    
    try:
        # Check for panic key
        if key_matches(key, config.panic_key):
            logger.warning("Panic key pressed! Disabling all functionality.")
            panic_triggered.set()
            config.update("aimbot", "enabled", False)
            config.update("triggerbot", "enabled", False)
            return
            
        if key_matches(key, config.aimbot_key):
            aimbot_active = True
            logger.info("Aimbot key pressed.")
            
        if key_matches(key, config.triggerbot_key):
            trigger_active = True
            logger.info("Trigger Bot key pressed.")
            
    except Exception as e:
        logger.error(f"Key press error: {e}")


def on_release(key):
    """Handle key release events.
    
    Args:
        key: The key that was released
    """
    global aimbot_active, trigger_active, smoothed_green_box_coords, smoothed_red_box_coords
    
    try:
        if key_matches(key, config.aimbot_key):
            aimbot_active = False
            logger.info("Aimbot key released.")
            
        if key_matches(key, config.triggerbot_key):
            trigger_active = False
            logger.info("Trigger Bot key released.")
            
        if not aimbot_active and not trigger_active:
            with boxes_lock:
                smoothed_green_box_coords = None
                smoothed_red_box_coords = None
                
    except Exception as e:
        logger.error(f"Key release error: {e}")


def on_mouse_click(x, y, button, pressed):
    """Handle mouse click events.
    
    Args:
        x: X position of the click
        y: Y position of the click
        button: The button that was clicked
        pressed: Whether the button was pressed or released
    """
    global left_mouse_pressed
    
    if button == Button.left:
        left_mouse_pressed = pressed


# Start keyboard and mouse listeners
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

mouse_listener = MouseListener(on_click=on_mouse_click)
mouse_listener.start()

def capture_screen():
    global latest_frame
    try:
        sct = mss.mss()
        while not exit_event.is_set():
            frame_raw = sct.grab(screen_bbox)
            # OPTIMIZATION: Replace cv2.cvtColor with faster numpy direct slicing
            frame_bgr = np.array(frame_raw)[:, :, :3]  # Direct slice BGR channels
            with frame_lock:
                latest_frame = frame_bgr
                frame_queue.append(frame_bgr)
            new_frame_event.set()
            time.sleep(0.001)
    except Exception as e:
        logger.error(f"Screen capture error: {e}")
 
capture_thread = threading.Thread(target=capture_screen, daemon=True)
capture_thread.start()
 
# Warm-up the inference engine
# for _ in range(5):
#     engine_manager.context.execute_v2(bindings=[int(engine_manager.bindings['d_input']), int(engine_manager.bindings['d_output'])])
 
# Screen capture with FPS tracking and error recovery
class ScreenCaptureManager:
    """Manages screen capture and provides frames for inference."""
    
    def __init__(self, config, screen_bbox):
        """Initialize the screen capture manager.
        
        Args:
            config: Configuration instance
            screen_bbox: Bounding box for screen capture
        """
        self.config = config
        self.screen_bbox = screen_bbox
        self.frame_lock = threading.Lock()
        self.frame_queue = deque(maxlen=3)  # Store a few frames to prevent frame drops
        self.latest_frame = None
        self.capture_fps = 0
        self.restart_required = threading.Event()
        self.thread = None
        
    def start(self):
        """Start the screen capture thread."""
        self.thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.thread.start()
        logger.info("Screen capture started")
        
    def _capture_thread(self):
        """Screen capture thread function."""
        try:
            sct = mss.mss()
            frame_times = deque(maxlen=100)  # For FPS calculation
            last_fps_display = time.time()
            
            while not exit_event.is_set() and not self.restart_required.is_set():
                try:
                    start_time = time.time()
                    
                    # Capture screen
                    frame_raw = sct.grab(self.screen_bbox)
                    # OPTIMIZATION: Replace cv2.cvtColor with faster numpy direct slicing
                    frame_bgr = np.array(frame_raw)[:, :, :3]  # Direct slice BGR channels
                    
                    # Store frame
                    with self.frame_lock:
                        self.latest_frame = frame_bgr
                        self.frame_queue.append(frame_bgr)
                        
                    # Signal that a new frame is available
                    new_frame_event.set()
                    
                    # Frame timing
                    frame_time = time.time() - start_time
                    frame_times.append(frame_time)
                    
                    # Calculate and display FPS occasionally
                    if time.time() - last_fps_display > 2.0:  # Every 2 seconds
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        self.capture_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                        logger.debug(f"Screen capture FPS: {self.capture_fps:.1f}")
                        last_fps_display = time.time()
                        
                    # Sleep to maintain a reasonable capture rate and reduce CPU usage
                    # OPTIMIZATION: Target higher FPS (240 instead of 120)
                    target_frame_time = 1.0 / 240.0
                    sleep_time = max(0, target_frame_time - frame_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Screen capture error: {e}")
                    # If something goes wrong, wait a bit and continue
                    time.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Fatal screen capture error: {e}")
            self.restart_required.set()
            
    def get_frame(self):
        """Get the latest frame.
        
        Returns:
            numpy.ndarray: The latest frame
        """
        with self.frame_lock:
            if not self.frame_queue:
                return None
            return self.frame_queue[-1].copy()
            
    def restart_if_needed(self):
        """Check if the capture thread needs to be restarted and do so if needed."""
        if self.restart_required.is_set():
            logger.warning("Restarting screen capture thread due to error")
            self.restart_required.clear()
            if self.thread and self.thread.is_alive():
                # Can't really terminate the thread, but we can wait for it to exit
                self.thread.join(timeout=1.0)
            self.start()
            return True
        return False


# Create a screen capture manager
screen_capture = ScreenCaptureManager(config, screen_bbox)
screen_capture.start()


# Inference loop with performance tracking
class InferenceLoop:
    """Main inference loop for object detection and aimbot functionality."""
    
    def __init__(self, config, engine_manager, screen_capture, aimbot_controller, triggerbot_controller):
        """Initialize the inference loop.
        
        Args:
            config: Configuration instance
            engine_manager: TensorRT engine manager
            screen_capture: Screen capture manager
            aimbot_controller: Aimbot controller
            triggerbot_controller: Triggerbot controller
        """
        self.config = config
        self.engine_manager = engine_manager
        self.screen_capture = screen_capture
        self.aimbot_controller = aimbot_controller
        self.triggerbot_controller = triggerbot_controller
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.frame_times = deque(maxlen=100)
        self.fps = 0
        self.inference_fps = 0
        self.last_fps_log = time.time()
        self.frame_count = 0
        
        # Detection state
        self.target_position_x = None
        self.target_position_y = None
        self.target_tracking_weight = 0.8  # How much weight to give to new detections vs. previous position
        self.detection_valid = False
        
    def run(self):
        """Run the inference loop."""
        try:
            logger.info("Starting inference loop")
            
            while not exit_event.is_set():
                try:
                    # Performance tracking
                    loop_start_time = time.time()
                    
                    # Check for screen capture errors and restart if needed
                    if self.screen_capture.restart_if_needed():
                        continue
                    
                    # Wait for a new frame with timeout
                    new_frame_event.wait(timeout=0.01)
                    new_frame_event.clear()
                    
                    # Get frame
                    frame = self.screen_capture.get_frame()
                    if frame is None:
                        time.sleep(0.001)
                        continue
                        
                    # Skip processing if no keys are active
                    if not (aimbot_active or trigger_active):
                        with boxes_lock:
                            global green_box_coords, red_box_coords
                            green_box_coords = None
                            red_box_coords = None
                        time.sleep(0.001)
                        continue
                        
                    # Run inference
                    inference_start = time.time()
                    output, inference_time = self.engine_manager.run_inference(frame)
                    inference_end = time.time()
                    
                    # Track inference performance
                    self.inference_times.append(inference_time)
                    
                    # Process detection results
                    self._process_detection(output, frame)
                    
                    # Update performance metrics
                    frame_time = time.time() - loop_start_time
                    self.frame_times.append(frame_time)
                    self.frame_count += 1
                    
                    # Log performance occasionally
                    if time.time() - self.last_fps_log > 5.0:  # Every 5 seconds
                        self._log_performance()
                        self.last_fps_log = time.time()
                        
                except Exception as e:
                    logger.error(f"Inference loop error: {e}")
                    time.sleep(0.1)  # Prevent tight loop on error
                    
        except KeyboardInterrupt:
            logger.info("Inference loop interrupted by user")
        except Exception as e:
            logger.error(f"Fatal inference loop error: {e}")
        finally:
            logger.info("Inference loop exiting")
            exit_event.set()
            
    def _process_detection(self, output, frame):
        """Process detection results.
        
        Args:
            output: Detection output from model
            frame: Input frame
        """
        global green_box_coords, red_box_coords, smoothed_green_box_coords, smoothed_red_box_coords
        
        try:
            # Parse detection output
            scores_threshold = 0.5
            detection_data = output[0]
            boxes = detection_data[:4, :]
            scores = detection_data[4:, :]
            max_scores = scores.max(axis=0)
            max_indices = scores.argmax(axis=0)
            valid_mask = (max_scores > scores_threshold) & (max_indices == 0)
            valid_indices = np.where(valid_mask)[0]
            
            if valid_indices.size > 0:
                # Get best detection
                best_index = valid_indices[np.argmax(max_scores[valid_indices])]
                box_coords = boxes[:, best_index]
                
                # Extract box coordinates
                x_center, y_center, width, height = box_coords
                x1 = x_center - (width / 2)
                y1 = y_center - (height / 2)
                x2 = x_center + (width / 2)
                y2 = y_center + (height / 2)
                
                # Calculate aim point based on target choice
                cx = (x1 + x2) / 2.0
                cy = y1 + (y2 - y1) * self.config.target_offsets.get(self.config.aim_target_choice, 0.1)
                
                # Transform coordinates to screen space
                bbox_center_x = screen_bbox['left'] + cx
                bbox_center_y = screen_bbox['top'] + cy
                
                # Apply temporal smoothing to target position for more stable aiming
                if self.target_position_x is None or self.target_position_y is None:
                    # First detection, initialize position
                    self.target_position_x = bbox_center_x
                    self.target_position_y = bbox_center_y
                else:
                    # Smooth the target position using exponential weighted average
                    self.target_position_x = self.target_tracking_weight * bbox_center_x + (1 - self.target_tracking_weight) * self.target_position_x
                    self.target_position_y = self.target_tracking_weight * bbox_center_y + (1 - self.target_tracking_weight) * self.target_position_y
                    
                # Calculate delta from screen center to smoothed target position
                delta_x = self.target_position_x - screen_center_x
                delta_y = self.target_position_y - screen_center_y
                self.detection_valid = True
                
                # Apply aimbot movement with improved algorithm
                if self.config.aimbot_enabled and aimbot_active and not panic_triggered.is_set():
                    # Calculate distance to determine if we're close enough for micro-adjustments
                    dist_to_target = np.hypot(delta_x, delta_y)
                    
                    # Use different strategies based on distance
                    if dist_to_target > 100:
                        # Far target - use faster movement
                        self.aimbot_controller.move_mouse_towards_target(delta_x, delta_y, max_speed=self.config.aimbot_speed)
                    elif dist_to_target > 30:
                        # Medium distance - use moderate speed
                        self.aimbot_controller.move_mouse_towards_target(delta_x, delta_y, max_speed=self.config.aimbot_speed * 0.8)
                    else:
                        # Close target - use precise micro-adjustments
                        self.aimbot_controller.move_mouse_towards_target(delta_x, delta_y, max_speed=self.config.aimbot_speed * 0.5)
                
                # Update the bounding boxes with the new detection
                new_green_box = (
                    screen_bbox['left'] + x1,
                    screen_bbox['top'] + y1,
                    screen_bbox['left'] + x2,
                    screen_bbox['top'] + y2
                )
                aim_box_size = 10
                new_red_box = (
                    self.target_position_x - aim_box_size / 2,  # Use smoothed position for the aim point
                    self.target_position_y - aim_box_size / 2,
                    self.target_position_x + aim_box_size / 2,
                    self.target_position_y + aim_box_size / 2
                )
                
                # Apply smooth transition to the UI elements
                if smoothed_green_box_coords is None:
                    smoothed_green_box_coords = new_green_box
                else:
                    smoothed_green_box_coords = tuple(self.config.smoothing_factor * (n - o) + o for n, o in zip(new_green_box, smoothed_green_box_coords))
                if smoothed_red_box_coords is None:
                    smoothed_red_box_coords = new_red_box
                else:
                    smoothed_red_box_coords = tuple(self.config.smoothing_factor * (n - o) + o for n, o in zip(new_red_box, smoothed_red_box_coords))
                with boxes_lock:
                    green_box_coords = smoothed_green_box_coords
                    red_box_coords = smoothed_red_box_coords
                
                # Trigger bot logic
                screen_x1 = screen_bbox['left'] + x1
                screen_y1 = screen_bbox['top'] + y1
                screen_x2 = screen_bbox['left'] + x2
                screen_y2 = screen_bbox['top'] + y2
                
                if (screen_x1 <= screen_center_x <= screen_x2 and 
                    screen_y1 <= screen_center_y <= screen_y2 and
                    not panic_triggered.is_set()):
                    trigger_should_run.set()
                    if not click_in_progress:
                        triggerbot_controller.should_stop.clear()
                        threading.Thread(target=triggerbot_controller.trigger_action, 
                                        args=(self.detection_valid,), daemon=True).start()
                    else:
                        trigger_should_run.clear()
                        triggerbot_controller.stop()
                else:
                    trigger_should_run.clear()
                    triggerbot_controller.stop()
            else:
                self.detection_valid = False
                # Keep the last target position for a short time to avoid jittery behavior
                # Reset target position tracking after a short timeout of no detection
                if self.frame_count % 10 == 0:
                    self.target_position_x = None
                    self.target_position_y = None
                    
                with boxes_lock:
                    green_box_coords = None
                    red_box_coords = None
                    
        except Exception as e:
            logger.error(f"Detection processing error: {e}")
            self.detection_valid = False
            
    def _log_performance(self):
        """Log performance metrics."""
        if not self.frame_times:
            return
            
        # Calculate FPS
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Calculate inference time
        if self.inference_times:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            self.inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        else:
            avg_inference_time = 0
            self.inference_fps = 0
            
        # Log performance
        logger.info(f"Performance: {self.fps:.1f} FPS, Inference: {avg_inference_time*1000:.1f}ms ({self.inference_fps:.1f} FPS)")
        
        # Log detection status
        if self.detection_valid:
            logger.info(f"Detection: Valid")
        else:
            logger.info(f"Detection: No targets")


def cleanup_resources():
    """Clean up resources before exit."""
    logger.info("Cleaning up resources...")
    
    try:
        # Release mouse button if pressed
        mouse_controller.release(1)
    except:
        pass
        
    try:
        # Stop keyboard and mouse listeners
        listener.stop()
        mouse_listener.stop()
    except:
        pass
        
    try:
        # Clean up TensorRT resources
        engine_manager.cleanup()
    except:
        pass
    
    # Try to terminate all threads
    for thread in all_threads:
        if thread.is_alive():
            logger.info(f"Waiting for thread {thread.name} to terminate...")
            thread.join(timeout=1.0)
    
    # Final termination message
    logger.info("Resources cleaned up. Exiting.")
    # Force exit if needed
    os._exit(0)

# Create and run the inference loop
inference = InferenceLoop(config, engine_manager, screen_capture, aimbot_controller, triggerbot_controller)

# Use CUDA events for high-resolution timing
start_event = cuda.Event()
end_event = cuda.Event()

# Main program execution wrapped in a try-except block
if __name__ == "__main__":
    try:
        # OPTIMIZATION: Set process priority to high for better performance
        try:
            import psutil
            process = psutil.Process(os.getpid())
            if sys.platform == 'win32':
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("Process priority set to HIGH")
            else:
                # For Unix-based systems
                process.nice(-10)  # Lower value = higher priority (range is usually -20 to 19)
                logger.info("Process priority increased")
        except Exception as e:
            logger.warning(f"Could not set process priority: {e}")
            
        # Main inference loop, following exactly the same structure as the original working code
        logger.info("Starting main inference loop")
        
        # Track inference performance
        inference_times = deque(maxlen=100)
        frame_times = deque(maxlen=100)
        last_fps_log = time.time()
        frame_count = 0
        
        while not exit_event.is_set():
            try:
                start_time = time.time()
                
                # Wait for a new frame with timeout
                new_frame_event.wait(timeout=0.001)
                with frame_lock:
                    frame = frame_queue.pop() if frame_queue else None
                new_frame_event.clear()
                
                if frame is None:
                    continue
                    
                if not (aimbot_active or trigger_active):
                    with boxes_lock:
                        green_box_coords = None
                        red_box_coords = None
                        target_position_x = None
                        target_position_y = None
                    time.sleep(0.001)
                    continue
                
                # OPTIMIZATION: Manual preprocessing (now handled in the run_inference method)
                blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(config.input_width, config.input_height),
                                           mean=(0, 0, 0), swapRB=False, crop=False)
                # Remove the batch dimension (blob shape is (1, C, H, W))
                np.copyto(engine_manager.h_input, blob.reshape(-1))
                
                # Run inference following exactly the same call pattern as the original working code
                start_event.record(engine_manager.stream)
                cuda.memcpy_htod_async(engine_manager.d_input, engine_manager.h_input, engine_manager.stream)
                engine_manager.context.set_tensor_address(engine_manager.input_name, int(engine_manager.d_input))
                engine_manager.context.set_tensor_address(engine_manager.output_name, int(engine_manager.d_output))
                engine_manager.context.execute_async_v3(stream_handle=engine_manager.stream.handle)
                cuda.memcpy_dtoh_async(engine_manager.h_output, engine_manager.d_output, engine_manager.stream)
                end_event.record(engine_manager.stream)
                engine_manager.stream.synchronize()
                inference_time = end_event.time_till(start_event) / 1000.0  # in seconds
                
                # Track inference performance
                inference_times.append(inference_time)
                frame_count += 1
                
                output = engine_manager.h_output.reshape(engine_manager.output_shape)
                scores_threshold = 0.5
                detection_data = output[0]
                boxes = detection_data[:4, :]
                scores = detection_data[4:, :]
                max_scores = scores.max(axis=0)
                max_indices = scores.argmax(axis=0)
                valid_mask = (max_scores > scores_threshold) & (max_indices == 0)
                valid_indices = np.where(valid_mask)[0]
                
                # Calculate frame processing time
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                # Log performance occasionally
                if time.time() - last_fps_log > 5.0 and frame_count > 10:
                    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
                    avg_frame = sum(frame_times) / len(frame_times) if frame_times else 0
                    fps = 1.0 / avg_frame if avg_frame > 0 else 0
                    
                    logger.info(f"Performance: {fps:.1f} FPS, Inference: {avg_inference*1000:.1f}ms")
                    last_fps_log = time.time()
                
                # Process detection results
                if valid_indices.size > 0:
                    best_index = valid_indices[np.argmax(max_scores[valid_indices])]
                    box_coords = boxes[:, best_index]
                    x_center, y_center, width, height = box_coords
                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)
                    cx = (x1 + x2) / 2.0
                    cy = y1 + (y2 - y1) * config.target_offsets.get(config.aim_target_choice, 0.1)
                    bbox_center_x = screen_bbox['left'] + cx
                    bbox_center_y = screen_bbox['top'] + cy
                    
                    # Global variables to track target position for smoother aim
                    target_position_x = None
                    target_position_y = None
                    target_tracking_weight = 0.8  # How much weight to give to new detections vs. previous position
                    
                    # Apply temporal smoothing to target position for more stable aiming
                    if target_position_x is None or target_position_y is None:
                        # First detection, initialize position
                        target_position_x = bbox_center_x
                        target_position_y = bbox_center_y
                    else:
                        # Smooth the target position using exponential weighted average
                        target_position_x = target_tracking_weight * bbox_center_x + (1 - target_tracking_weight) * target_position_x
                        target_position_y = target_tracking_weight * bbox_center_y + (1 - target_tracking_weight) * target_position_y
                        
                    # Calculate delta from screen center to smoothed target position
                    delta_x = target_position_x - screen_center_x
                    delta_y = target_position_y - screen_center_y
                    detection_valid = True
                    
                    # Apply aimbot movement with improved algorithm
                    if config.aimbot_enabled and aimbot_active and not panic_triggered.is_set():
                        # Calculate distance to determine if we're close enough for micro-adjustments
                        dist_to_target = np.hypot(delta_x, delta_y)
                        
                        # Use different strategies based on distance
                        if dist_to_target > 100:
                            # Far target - use faster movement
                            aimbot_controller.move_mouse_towards_target(delta_x, delta_y, max_speed=config.aimbot_speed)
                        elif dist_to_target > 30:
                            # Medium distance - use moderate speed
                            aimbot_controller.move_mouse_towards_target(delta_x, delta_y, max_speed=config.aimbot_speed * 0.8)
                        else:
                            # Close target - use precise micro-adjustments
                            aimbot_controller.move_mouse_towards_target(delta_x, delta_y, max_speed=config.aimbot_speed * 0.5)
                    
                    # Update the bounding boxes with the new detection
                    new_green_box = (
                        screen_bbox['left'] + x1,
                        screen_bbox['top'] + y1,
                        screen_bbox['left'] + x2,
                        screen_bbox['top'] + y2
                    )
                    aim_box_size = 10
                    new_red_box = (
                        target_position_x - aim_box_size / 2,  # Use smoothed position for the aim point
                        target_position_y - aim_box_size / 2,
                        target_position_x + aim_box_size / 2,
                        target_position_y + aim_box_size / 2
                    )
                    
                    # Apply smooth transition to the UI elements
                    if smoothed_green_box_coords is None:
                        smoothed_green_box_coords = new_green_box
                    else:
                        smoothed_green_box_coords = tuple(config.smoothing_factor * (n - o) + o for n, o in zip(new_green_box, smoothed_green_box_coords))
                    if smoothed_red_box_coords is None:
                        smoothed_red_box_coords = new_red_box
                    else:
                        smoothed_red_box_coords = tuple(config.smoothing_factor * (n - o) + o for n, o in zip(new_red_box, smoothed_red_box_coords))
                    with boxes_lock:
                        green_box_coords = smoothed_green_box_coords
                        red_box_coords = smoothed_red_box_coords
                    
                    screen_x1 = screen_bbox['left'] + x1
                    screen_y1 = screen_bbox['top'] + y1
                    screen_x2 = screen_bbox['left'] + x2
                    screen_y2 = screen_bbox['top'] + y2
                    
                    # Trigger bot logic
                    if (screen_x1 <= screen_center_x <= screen_x2 and 
                        screen_y1 <= screen_center_y <= screen_y2 and
                        not panic_triggered.is_set()):
                        trigger_should_run.set()
                        if not click_in_progress:
                            triggerbot_controller.should_stop.clear()
                            threading.Thread(target=triggerbot_controller.trigger_action, 
                                           args=(detection_valid,), daemon=True).start()
                    else:
                        trigger_should_run.clear()
                        triggerbot_controller.stop()
                else:
                    detection_valid = False
                    trigger_should_run.clear()
                    triggerbot_controller.stop()
                    # Keep the last target position for a short time to avoid jittery behavior
                    # when detection temporarily fails for a frame
                    # Reset target position tracking after 10 frames of no detection
                    if frame_count % 10 == 0:
                        target_position_x = None
                        target_position_y = None
                        
                    with boxes_lock:
                        green_box_coords = None
                        red_box_coords = None
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Brief pause to avoid spinning on errors
                time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception in main program: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Set exit event and clean up
        exit_event.set()
        cleanup_resources()
        # This ensures the program exits completely
        sys.exit(0)
# TensorRT_Ctypes.py
import cv2
import numpy as np
import mss
import time
from threading import Thread, Lock
from pynput import keyboard
from pynput.mouse import Listener as MouseListener, Button
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import logging
import sys
import tkinter as tk
from tkinter import font
import random
import ctypes
from ctypes import wintypes

# Define ULONG_PTR for compatibility
if not hasattr(wintypes, "ULONG_PTR"):
    if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_ulong):
        wintypes.ULONG_PTR = ctypes.c_ulong
    else:
        wintypes.ULONG_PTR = ctypes.c_ulonglong

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s')

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

# Global control variables for UI toggles and key activations
# Inference will only run when the selected key is held.
aimbot_enabled = True
trigger_enabled = True
boxes_enabled = True
scanning_box_enabled = True

aimbot_speed = 350  # Maximum speed for smoothing (pixels per update)
aim_target_choice = "neck"  # Default target

# Key customization globals with default values selected from allowed keys
allowed_keys = ["alt", "shift", "caps lock", "x", "c", "v", "b", "z"]
aimbot_key = "alt"       # Default key for aimbot activation
triggerbot_key = "alt"   # Default key for triggerbot activation

# These booleans track if the custom keys are currently pressed.
aimbot_active = False
trigger_active = False

# Additional flag to indicate if a valid detection exists
detection_valid = False

# Global variable to track if left mouse button is physically pressed by the user
left_mouse_pressed = False

# Global variables for smoothing bounding box coordinates
smoothed_green_box_coords = None
smoothed_red_box_coords = None
smoothing_factor = 0.3  # Adjusted smoothing factor (alpha)

def load_engine(trt_engine_path):
    with open(trt_engine_path, 'rb') as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)

engine = load_engine("model_fp16_320.trt")
context = engine.create_execution_context()

input_binding_idx = 0
output_binding_idx = 1
input_name = engine.get_tensor_name(input_binding_idx)
output_name = engine.get_tensor_name(output_binding_idx)
input_shape = engine.get_tensor_shape(input_name)
output_shape = engine.get_tensor_shape(output_name)

batch_size = 1
batch_input_shape = input_shape
batch_output_shape = output_shape

input_size = trt.volume(batch_input_shape) * np.dtype(np.float32).itemsize
output_size = trt.volume(batch_output_shape) * np.dtype(np.float32).itemsize

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
stream = cuda.Stream()
h_input = cuda.pagelocked_empty(trt.volume(batch_input_shape), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(batch_output_shape), dtype=np.float32)

def get_screen_info():
    with mss.mss() as sct:
        monitors = sct.monitors
        if len(monitors) > 1:
            monitor = monitors[1]
        else:
            monitor = monitors[0]
        screen_width = monitor['width']
        screen_height = monitor['height']
    return screen_width, screen_height, monitor

def get_center_bbox(input_width, input_height, monitor):
    screen_width = monitor['width']
    screen_height = monitor['height']
    center_x = screen_width // 2
    center_y = screen_height // 2
    top_left_x = center_x - (input_width // 2)
    top_left_y = center_y - (input_height // 2)
    return {
        'top': top_left_y + monitor['top'],
        'left': top_left_x + monitor['left'],
        'width': input_width,
        'height': input_height
    }

input_width = 320
input_height = 320
screen_width, screen_height, monitor = get_screen_info()
screen_bbox = get_center_bbox(input_width, input_height, monitor)
screen_center_x = monitor['left'] + screen_width // 2
screen_center_y = monitor['top'] + screen_height // 2

logging.info(f"Screen Width: {screen_width}, Screen Height: {screen_height}")
logging.info(f"Screen BBox: {screen_bbox}")
logging.info(f"Screen Center: ({screen_center_x}, {screen_center_y})")

frame_lock = Lock()
latest_frame = None
click_in_progress = False
click_lock = Lock()

boxes_lock = Lock()
green_box_coords = None
red_box_coords = None

# Define constants for mouse events
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_cursor_position():
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt

# Structures for SendInput
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR)
    ]

class INPUT_union(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_union)]

def move_mouse_relative(dx, dy):
    extra = 0
    mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, extra)
    inp = INPUT(0, INPUT_union(mi))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

offset_x = 1
offset_y = 1

def linear_ease(t):
    return t

def move_mouse_towards_target(delta_x, delta_y, max_speed=350):
    delta_x += offset_x
    delta_y += offset_y
    distance = np.hypot(delta_x, delta_y)
    if distance == 0:
        return
    direction_x = delta_x / distance
    direction_y = delta_y / distance
    normalized_distance = min(distance / max_speed, 1.0)
    ease_value = linear_ease(normalized_distance)
    move_distance = ease_value * max_speed
    move_x = int(direction_x * move_distance)
    move_y = int(direction_y * move_distance)
    logging.debug(f"Moving mouse by ({move_x}, {move_y}) towards delta ({delta_x}, {delta_y})")
    move_mouse_relative(move_x, move_y)

class MouseController:
    def __init__(self):
        pass
    def press(self, button):
        if button == 1:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    def release(self, button):
        if button == 1:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

mouse_controller = MouseController()

def overlay_window():
    global screen_bbox, screen_center_x, screen_center_y, smoothing_factor
    root = tk.Tk()
    root.title("Overlay")
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.9)
    root.attributes('-fullscreen', True)
    root.attributes("-transparentcolor", "white")
    root.configure(background='white')
    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='white', highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    control_panel = tk.Toplevel(root)
    control_panel.title("Control Panel")
    control_panel.geometry("420x750+50+50")
    control_panel.configure(background='#2d2d30')

    modern_font = font.Font(family="Segoe UI", size=10, weight="normal")

    aimbot_var = tk.BooleanVar(value=aimbot_enabled)
    trigger_var = tk.BooleanVar(value=trigger_enabled)
    boxes_var = tk.BooleanVar(value=boxes_enabled)
    scan_box_var = tk.BooleanVar(value=scanning_box_enabled)
    speed_var = tk.IntVar(value=aimbot_speed)
    target_var = tk.StringVar(value=aim_target_choice)
    aimbot_key_var = tk.StringVar(value=aimbot_key)
    triggerbot_key_var = tk.StringVar(value=triggerbot_key)
    smoothing_var = tk.DoubleVar(value=smoothing_factor)

    def update_aimbot():
        global aimbot_enabled
        aimbot_enabled = aimbot_var.get()

    def update_trigger():
        global trigger_enabled
        trigger_enabled = trigger_var.get()

    def update_boxes():
        global boxes_enabled
        boxes_enabled = boxes_var.get()

    def update_scan_box():
        global scanning_box_enabled
        scanning_box_enabled = scan_box_var.get()

    def update_speed(val):
        global aimbot_speed
        aimbot_speed = int(val)
        speed_label.config(text=f"{aimbot_speed}")

    def update_target(*args):
        global aim_target_choice
        aim_target_choice = target_var.get()

    def update_keys(*args):
        global aimbot_key, triggerbot_key
        aimbot_key = aimbot_key_var.get().lower()
        triggerbot_key = triggerbot_key_var.get().lower()

    def update_smoothing(val):
        global smoothing_factor
        smoothing_factor = float(val)
        smoothing_label.config(text=f"{smoothing_factor:.2f}")

    label_style = {"bg": "#2d2d30", "fg": "#00FF00", "font": modern_font}
    checkbox_style = {"bg": "#2d2d30", "fg": "#00FF00", "activebackground": "#2d2d30", "selectcolor": "#2d2d30", "font": modern_font}

    header = tk.Label(control_panel, text="AI AimShit & Trigger Sot", **label_style)
    header.pack(pady=10)

    aimbot_checkbox = tk.Checkbutton(control_panel, text="Enable Aimbot", variable=aimbot_var, command=update_aimbot, **checkbox_style)
    aimbot_checkbox.pack(anchor="w", padx=20, pady=5)

    trigger_checkbox = tk.Checkbutton(control_panel, text="Enable Trigger Bot", variable=trigger_var, command=update_trigger, **checkbox_style)
    trigger_checkbox.pack(anchor="w", padx=20, pady=5)

    boxes_checkbox = tk.Checkbutton(control_panel, text="Show Bounding Boxes", variable=boxes_var, command=update_boxes, **checkbox_style)
    boxes_checkbox.pack(anchor="w", padx=20, pady=5)

    scan_box_checkbox = tk.Checkbutton(control_panel, text="Show Scanning Box", variable=scan_box_var, command=update_scan_box, **checkbox_style)
    scan_box_checkbox.pack(anchor="w", padx=20, pady=5)

    speed_frame = tk.Frame(control_panel, bg="#2d2d30")
    speed_frame.pack(fill="x", padx=20, pady=15)
    speed_title = tk.Label(speed_frame, text="Aimbot Smooth Speed", **label_style)
    speed_title.pack(anchor="w")
    speed_slider = tk.Scale(speed_frame, from_=50, to=1000, orient="horizontal", variable=speed_var, command=update_speed, bg="#2d2d30", fg="#00FF00", highlightbackground="#2d2d30", font=modern_font)
    speed_slider.pack(fill="x")
    speed_label = tk.Label(speed_frame, text=f"{aimbot_speed}", **label_style)
    speed_label.pack(anchor="e")

    target_frame = tk.Frame(control_panel, bg="#2d2d30")
    target_frame.pack(fill="x", padx=20, pady=15)
    target_label = tk.Label(target_frame, text="Select Target:", **label_style)
    target_label.pack(anchor="w")
    target_options = ["head", "neck", "chest", "legs", "balls"]
    target_menu = tk.OptionMenu(target_frame, target_var, *target_options, command=update_target)
    target_menu.config(bg="#2d2d30", fg="#00FF00", activebackground="#2d2d30", font=modern_font)
    target_menu["menu"].config(bg="#2d2d30", fg="#00FF00", font=modern_font)
    target_menu.pack(fill="x")

    key_frame = tk.Frame(control_panel, bg="#2d2d30")
    key_frame.pack(fill="x", padx=20, pady=15)
    aimbot_key_label = tk.Label(key_frame, text="Aimbot Key:", **label_style)
    aimbot_key_label.pack(anchor="w")
    aimbot_key_menu = tk.OptionMenu(key_frame, aimbot_key_var, *allowed_keys, command=update_keys)
    aimbot_key_menu.config(bg="#2d2d30", fg="#00FF00", activebackground="#2d2d30", font=modern_font)
    aimbot_key_menu["menu"].config(bg="#2d2d30", fg="#00FF00", font=modern_font)
    aimbot_key_menu.pack(fill="x")
    trigger_key_label = tk.Label(key_frame, text="Trigger Bot Key:", **label_style)
    trigger_key_label.pack(anchor="w", pady=(10, 0))
    trigger_key_menu = tk.OptionMenu(key_frame, triggerbot_key_var, *allowed_keys, command=update_keys)
    trigger_key_menu.config(bg="#2d2d30", fg="#00FF00", activebackground="#2d2d30", font=modern_font)
    trigger_key_menu["menu"].config(bg="#2d2d30", fg="#00FF00", font=modern_font)
    trigger_key_menu.pack(fill="x")

    smoothing_frame = tk.Frame(control_panel, bg="#2d2d30")
    smoothing_frame.pack(fill="x", padx=20, pady=15)
    smoothing_title = tk.Label(smoothing_frame, text="Bounding Box Smoothing", **label_style)
    smoothing_title.pack(anchor="w")
    smoothing_slider = tk.Scale(smoothing_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", variable=smoothing_var, command=update_smoothing, bg="#2d2d30", fg="#00FF00", highlightbackground="#2d2d30", font=modern_font)
    smoothing_slider.pack(fill="x")
    smoothing_label = tk.Label(smoothing_frame, text=f"{smoothing_factor:.2f}", **label_style)
    smoothing_label.pack(anchor="e")

    exit_button = tk.Button(control_panel, text="Exit", command=root.destroy, bg="#2d2d30", fg="#00FF00", activebackground="#2d2d30", font=modern_font)
    exit_button.pack(pady=20)

    while True:
        if not root.winfo_exists():
            break
        try:
            if scanning_box_enabled:
                scanning_box = (
                    screen_center_x - (input_width // 2),
                    screen_center_y - (input_height // 2),
                    screen_center_x + (input_width // 2),
                    screen_center_y + (input_height // 2)
                )
                if canvas.find_withtag("scanning_box") == ():
                    canvas.create_rectangle(scanning_box, outline='yellow', width=2, tag="scanning_box")
                else:
                    canvas.coords("scanning_box", *scanning_box)
            else:
                canvas.delete("scanning_box")
            with boxes_lock:
                if boxes_enabled:
                    if green_box_coords is not None:
                        if canvas.find_withtag("green_box") == ():
                            canvas.create_rectangle(green_box_coords, outline='green', width=2, tag="green_box")
                        else:
                            canvas.coords("green_box", *green_box_coords)
                    else:
                        canvas.delete("green_box")
                    if red_box_coords is not None:
                        if canvas.find_withtag("red_box") == ():
                            canvas.create_rectangle(red_box_coords, outline='red', width=2, tag="red_box")
                        else:
                            canvas.coords("red_box", *red_box_coords)
                    else:
                        canvas.delete("red_box")
                else:
                    canvas.delete("green_box")
                    canvas.delete("red_box")
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break
        time.sleep(0.03)

overlay_thread = Thread(target=overlay_window, daemon=True)
overlay_thread.start()

def key_matches(key, target_str):
    target_str = target_str.lower()
    special_keys = {
        "alt": [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r],
        "shift": [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r],
        "caps lock": [keyboard.Key.caps_lock]
    }
    if target_str in special_keys:
        return key in special_keys[target_str]
    else:
        if hasattr(key, 'char') and key.char is not None:
            return key.char.lower() == target_str
        return False

def on_press(key):
    global aimbot_active, trigger_active
    try:
        if key_matches(key, aimbot_key):
            aimbot_active = True
            logging.info("Aimbot key pressed.")
        if key_matches(key, triggerbot_key):
            trigger_active = True
            logging.info("Trigger Bot key pressed.")
    except Exception as e:
        logging.error(f"Key press error: {e}")

def on_release(key):
    global aimbot_active, trigger_active, smoothed_green_box_coords, smoothed_red_box_coords
    try:
        if key_matches(key, aimbot_key):
            aimbot_active = False
            logging.info("Aimbot key released.")
        if key_matches(key, triggerbot_key):
            trigger_active = False
            logging.info("Trigger Bot key released.")
        if not aimbot_active and not trigger_active:
            with boxes_lock:
                smoothed_green_box_coords = None
                smoothed_red_box_coords = None
    except Exception as e:
        logging.error(f"Key release error: {e}")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def on_mouse_click(x, y, button, pressed):
    global left_mouse_pressed
    if button == Button.left:
        left_mouse_pressed = pressed

mouse_listener = MouseListener(on_click=on_mouse_click)
mouse_listener.start()

def capture_screen():
    sct = mss.mss()
    global latest_frame
    while True:
        try:
            screen_image = np.array(sct.grab(screen_bbox))
            screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)
            with frame_lock:
                latest_frame = screen_image.copy()
        except Exception as e:
            logging.error(f"Screen capture error: {e}")
        time.sleep(0.001)

capture_thread = Thread(target=capture_screen, daemon=True)
capture_thread.start()

for _ in range(5):
    context.execute_v2(bindings=[int(d_input), int(d_output)])

target_offsets = {
    "head": 0.1,
    "neck": 0.2,
    "chest": 0.3,
    "legs": 0.8,
    "balls": 0.5
}

def trigger_action():
    global click_in_progress
    with click_lock:
        if click_in_progress:
            return
        click_in_progress = True
    try:
        while trigger_enabled and trigger_active and detection_valid:
            if left_mouse_pressed:
                time.sleep(0.01)
                continue
            mouse_controller.press(1)
            time.sleep(random.uniform(0.05, 0.3))
            mouse_controller.release(1)
            time.sleep(random.uniform(0.1, 0.3))
    finally:
        with click_lock:
            click_in_progress = False

while True:
    if not (aimbot_active or trigger_active):
        with boxes_lock:
            green_box_coords = None
            red_box_coords = None
        time.sleep(0.001)
        continue
    with frame_lock:
        frame = latest_frame
        latest_frame = None
    if frame is not None:
        image_resized = cv2.resize(frame, (input_width, input_height))
        image_transposed = np.transpose(image_resized, (2, 0, 1))
        image_normalized = image_transposed.astype(np.float32) / 255.0
        np.copyto(h_input, image_normalized.ravel())
        start_time = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        elapsed_time = time.time() - start_time
        output = h_output.reshape(batch_output_shape)
        scores_threshold = 0.5
        detections = []
        for i in range(output.shape[2]):
            box_and_scores = output[0, :, i]
            box_coords = box_and_scores[:4]
            class_scores = box_and_scores[4:]
            max_class_score = np.max(class_scores)
            max_class_index = np.argmax(class_scores)
            if max_class_score > scores_threshold and max_class_index == 0:
                detections.append((box_coords, max_class_score, max_class_index))
        if detections:
            detections.sort(key=lambda x: x[1], reverse=True)
            box_coords, max_class_score, max_class_index = detections[0]
            box_coords = np.array(box_coords, dtype=np.float32)
            x_center, y_center, width, height = box_coords
            x1 = x_center - (width / 2)
            y1 = y_center - (height / 2)
            x2 = x_center + (width / 2)
            y2 = y_center + (height / 2)
            cx = (x1 + x2) / 2.0
            cy = y1 + (y2 - y1) * target_offsets.get(aim_target_choice, 0.1)
            bbox_center_x = screen_bbox['left'] + cx
            bbox_center_y = screen_bbox['top'] + cy
            detection_valid = True
            if aimbot_enabled and aimbot_active:
                # Calculate delta relative to screen center and move mouse accordingly
                delta_x = bbox_center_x - screen_center_x
                delta_y = bbox_center_y - screen_center_y
                move_mouse_towards_target(delta_x, delta_y, max_speed=aimbot_speed)
            new_green_box = (
                screen_bbox['left'] + x1,
                screen_bbox['top'] + y1,
                screen_bbox['left'] + x2,
                screen_bbox['top'] + y2
            )
            aim_box_size = 10
            new_red_box = (
                screen_bbox['left'] + cx - aim_box_size / 2,
                screen_bbox['top'] + cy - aim_box_size / 2,
                screen_bbox['left'] + cx + aim_box_size / 2,
                screen_bbox['top'] + cy + aim_box_size / 2
            )
            if smoothed_green_box_coords is None:
                smoothed_green_box_coords = new_green_box
            else:
                smoothed_green_box_coords = tuple(smoothing_factor * (n - o) + o for n, o in zip(new_green_box, smoothed_green_box_coords))
            if smoothed_red_box_coords is None:
                smoothed_red_box_coords = new_red_box
            else:
                smoothed_red_box_coords = tuple(smoothing_factor * (n - o) + o for n, o in zip(new_red_box, smoothed_red_box_coords))
            with boxes_lock:
                green_box_coords = smoothed_green_box_coords
                red_box_coords = smoothed_red_box_coords
            if trigger_enabled and trigger_active:
                screen_x1 = screen_bbox['left'] + x1
                screen_y1 = screen_bbox['top'] + y1
                screen_x2 = screen_bbox['left'] + x2
                screen_y2 = screen_bbox['top'] + y2
                if (screen_x1 <= screen_center_x <= screen_x2 and 
                    screen_y1 <= screen_center_y <= screen_y2):
                    if not click_in_progress:
                        Thread(target=trigger_action, daemon=True).start()
            logging.info(f"Detection: Detected, Inference Time: {elapsed_time:.3f} sec")
        else:
            detection_valid = False
            with boxes_lock:
                green_box_coords = None
                red_box_coords = None
            logging.info(f"Detection: Not Detected, Inference Time: {elapsed_time:.3f} sec")
    else:
        time.sleep(0.001)

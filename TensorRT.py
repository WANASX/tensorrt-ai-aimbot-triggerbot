# TensorRT.py
import os
import sys
import time
import random
import logging
import threading
import tkinter as tk
from tkinter import font
from collections import deque

import cv2
import numpy as np
import mss
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pynput import keyboard
from pynput.mouse import Listener as MouseListener, Button
from mouse_driver.ghub_mouse import LogiFck

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s')

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

aimbot_enabled = True
trigger_enabled = True
boxes_enabled = True
scanning_box_enabled = True

aimbot_speed = 300
aim_target_choice = "neck"

allowed_keys = ["alt", "shift", "caps lock", "x", "c", "v", "b", "z"]
aimbot_key = "alt"
triggerbot_key = "x"

aimbot_active = False
trigger_active = False
detection_valid = False
left_mouse_pressed = False

smoothed_green_box_coords = None
smoothed_red_box_coords = None
smoothing_factor = 0.3

exit_event = threading.Event()
new_frame_event = threading.Event()
trigger_event = threading.Event()  # For persistent trigger action thread

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

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

def get_screen_info():
    with mss.mss() as sct:
        monitors = sct.monitors
        monitor = monitors[1] if len(monitors) > 1 else monitors[0]
        screen_width = monitor['width']
        screen_height = monitor['height']
    return screen_width, screen_height, monitor

def get_center_bbox(width, height, monitor):
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

input_width = 320
input_height = 320
screen_width, screen_height, monitor = get_screen_info()
screen_bbox = get_center_bbox(input_width, input_height, monitor)
screen_center_x = monitor['left'] + screen_width // 2
screen_center_y = monitor['top'] + screen_height // 2

# Precompute constants that do not change
scanning_box_coords = (
    screen_center_x - (input_width // 2),
    screen_center_y - (input_height // 2),
    screen_center_x + (input_width // 2),
    screen_center_y + (input_height // 2)
)

logging.info(f"Screen Width: {screen_width}, Screen Height: {screen_height}")
logging.info(f"Screen BBox: {screen_bbox}")
logging.info(f"Screen Center: ({screen_center_x}, {screen_center_y})")

frame_lock = threading.Lock()
boxes_lock = threading.Lock()
click_lock = threading.Lock()
frame_queue = deque(maxlen=1)

latest_frame = None
click_in_progress = False

green_box_coords = None
red_box_coords = None

mouse_controller = LogiFck('ghub_mouse.dll')

def overlay_window():
    global screen_bbox, screen_center_x, screen_center_y, smoothing_factor, green_box_coords, red_box_coords
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
    speed_slider = tk.Scale(speed_frame, from_=1, to=1000, orient="horizontal", variable=speed_var, command=update_speed, bg="#2d2d30", fg="#00FF00", highlightbackground="#2d2d30", font=modern_font)
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

    def exit_program():
        exit_event.set()
        root.destroy()
    exit_button = tk.Button(control_panel, text="Exit", command=exit_program, bg="#2d2d30", fg="#00FF00", activebackground="#2d2d30", font=modern_font)
    exit_button.pack(pady=20)

    def update_loop():
        if scanning_box_enabled:
            if not canvas.find_withtag("scanning_box"):
                canvas.create_rectangle(scanning_box_coords, outline='yellow', width=2, tag="scanning_box")
            else:
                canvas.coords("scanning_box", *scanning_box_coords)
        else:
            canvas.delete("scanning_box")
        with boxes_lock:
            if boxes_enabled:
                if green_box_coords is not None:
                    if not canvas.find_withtag("green_box"):
                        canvas.create_rectangle(green_box_coords, outline='green', width=2, tag="green_box")
                    else:
                        canvas.coords("green_box", *green_box_coords)
                else:
                    canvas.delete("green_box")
                if red_box_coords is not None:
                    if not canvas.find_withtag("red_box"):
                        canvas.create_rectangle(red_box_coords, outline='red', width=2, tag="red_box")
                    else:
                        canvas.coords("red_box", *red_box_coords)
                else:
                    canvas.delete("red_box")
            else:
                canvas.delete("green_box")
                canvas.delete("red_box")
        root.update_idletasks()
        if not exit_event.is_set():
            root.after(30, update_loop)
    root.after(30, update_loop)
    root.mainloop()

overlay_thread = threading.Thread(target=overlay_window, daemon=True)
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
    global latest_frame
    try:
        # Create the MSS instance inside this thread to ensure proper thread-local initialization.
        with mss.mss() as sct:
            while not exit_event.is_set():
                frame_raw = sct.grab(screen_bbox)
                frame_array = np.array(frame_raw)
                frame_bgr = frame_array[:, :, :3]  # Discard alpha channel
                with frame_lock:
                    latest_frame = frame_bgr
                    frame_queue.append(frame_bgr)
                new_frame_event.set()
                time.sleep(0.001)
    except Exception as e:
        logging.error(f"Screen capture error: {e}")

capture_thread = threading.Thread(target=capture_screen, daemon=True)
capture_thread.start()

# Warm-up the inference engine
for _ in range(5):
    context.execute_v2(bindings=[int(d_input), int(d_output)])

offset_x = 1
offset_y = 1

target_offsets = {
    "head": 0.1,
    "neck": 0.2,
    "chest": 0.3,
    "legs": 0.8,
    "balls": 0.5
}

def linear_ease(t):
    return t

def move_mouse_towards_target(delta_x, delta_y, max_speed=350):
    if abs(delta_x) < 1 and abs(delta_y) < 1:
        return
    if not mouse_controller.gmok:
        logging.error("Mouse controller not initialized properly.")
        return
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
    mouse_controller.move_relative(move_x, move_y)

def trigger_action_worker():
    global click_in_progress
    while not exit_event.is_set():
        trigger_event.wait(timeout=0.005)
        if trigger_event.is_set() and trigger_enabled and trigger_active and detection_valid and not left_mouse_pressed:
            with click_lock:
                if click_in_progress:
                    continue
                click_in_progress = True
            try:
                mouse_controller.press(1)
                time.sleep(random.uniform(0.09, 0.3))
                mouse_controller.release(1)
                time.sleep(random.uniform(0.1, 0.3))
            finally:
                with click_lock:
                    click_in_progress = False
        trigger_event.clear()

trigger_thread = threading.Thread(target=trigger_action_worker, daemon=True)
trigger_thread.start()

start_event = cuda.Event()
end_event = cuda.Event()

frame_count = 0

while not exit_event.is_set():
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
        time.sleep(0.001)
        continue

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(input_width, input_height),
                                   mean=(0, 0, 0), swapRB=False, crop=False)
    np.copyto(h_input, blob.reshape(-1))

    start_event.record(stream)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    end_event.record(stream)
    stream.synchronize()
    inference_time = end_event.time_till(start_event) / 1000.0

    output = h_output.reshape(batch_output_shape)
    scores_threshold = 0.5
    detection_data = output[0]
    boxes = detection_data[:4, :]
    scores = detection_data[4:, :]
    max_scores = scores.max(axis=0)
    max_indices = scores.argmax(axis=0)
    valid_mask = (max_scores > scores_threshold) & (max_indices == 0)
    valid_indices = np.where(valid_mask)[0]

    if valid_indices.size > 0:
        best_index = valid_indices[np.argmax(max_scores[valid_indices])]
        box_coords = boxes[:, best_index]
        x_center, y_center, width, height = box_coords
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)
        cx = (x1 + x2) / 2.0
        cy = y1 + (y2 - y1) * target_offsets.get(aim_target_choice, 0.1)
        bbox_center_x = screen_bbox['left'] + cx
        bbox_center_y = screen_bbox['top'] + cy
        delta_x = bbox_center_x - screen_center_x
        delta_y = bbox_center_y - screen_center_y
        detection_valid = True
        if aimbot_enabled and aimbot_active:
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

        screen_x1 = screen_bbox['left'] + x1
        screen_y1 = screen_bbox['top'] + y1
        screen_x2 = screen_bbox['left'] + x2
        screen_y2 = screen_bbox['top'] + y2
        if (screen_x1 <= screen_center_x <= screen_x2 and 
            screen_y1 <= screen_center_y <= screen_y2):
            trigger_event.set()
        frame_count += 1
        if frame_count % 50 == 0:
            logging.info(f"Detection: Detected, Inference Time: {inference_time:.3f} sec")
    else:
        detection_valid = False
        with boxes_lock:
            green_box_coords = None
            red_box_coords = None
        frame_count += 1
        if frame_count % 50 == 0:
            logging.info(f"Detection: Not Detected, Inference Time: {inference_time:.3f} sec")
sys.exit(0)

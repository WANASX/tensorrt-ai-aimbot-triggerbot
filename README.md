# TensorRT Aimbot

An AI-powered aimbot and triggerbot using TensorRT for high-performance inference, with a modern, modular, and user-friendly design.

## ⚠️ DISCLAIMER

**This software is provided for EDUCATIONAL PURPOSES ONLY.**

- Using this software in online competitive games is likely against the Terms of Service and may result in your account being banned.
- This project is intended for research into computer vision and AI, not for cheating in games.
- The authors take NO RESPONSIBILITY for any consequences resulting from misuse of this software.

---

   - Official Website [AI Aimbot & Trigger Bot Cheat for Every Shooter Game](https://www.gamerfun.club/ai-aimbot-triggerbot-shooter-games)
   - Official Forum [AI Aimbot & Trigger Bot Cheat for Every Shooter Game GamerFun Forum](https://forum.gamerfun.club/threads/ai-aimbot-trigger-bot-cheat-for-every-shooter-game.862/)
   - UnknownCheats Forum [AI Aimbot & Trigger Bot That Outperforms Internal and External Cheats 🔥](https://www.unknowncheats.me/forum/rainbow-six-siege/685011-ai-aimbot-trigger-bot-outperforms-internal-external-cheats.html)
   - Discord Server [Damascus Discord Server](https://discord.gg/cvVvFrf)



## 🚀 How to Install & Use

### Step 1: Download & Install Dependencies

1. **TensorRT 10.7 GA for Windows & CUDA 12.x**
   - Download [NVIDIA TensorRT 10.x](https://developer.nvidia.com/tensorrt/download/10x)
   - Download and install [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)

2. **Logitech G HUB (2021-10-8013)**
   - Download from [Logitech G HUB](https://www.unknowncheats.me/forum/downloads.php?do=file&id=39879)
   - *Install even if you don't own a Logitech mouse (required for mouse driver)*

3. **Python 3.9+ (64-bit)**
   - Download and install from [python.org](https://www.python.org/downloads/)

4. **Set Up TensorRT Environment**
   - Extract the TensorRT ZIP package (e.g., to `C:\TensorRT-10.7`)
   - Add `C:\TensorRT-10.7\lib` to your system `PATH`
   - Set `TENSORRT_HOME` environment variable to `C:\TensorRT-10.7`
   - Install the TensorRT Python wheel:
     - Open CMD in `C:\TensorRT-10.7\python`
     - Run: `pip install tensorrt-*.whl` *(choose the .whl matching your Python version)*

### Step 2: Install Python Dependencies

Open CMD in the project folder and run:
```sh
pip install opencv-python numpy mss pynput pycuda tensorrt pyautogui pillow
```

### Step 3: Convert the AI Model to TensorRT

- The ONNX model (`model1_320.onnx`) is already included.
- Use the provided script to convert it to a TensorRT engine:

```sh
python convert_to_trt.py model1_320.onnx model_fp16_320.trt --fp16
```
- Wait for the process to complete. This will generate `model_fp16_320.trt`.

### Step 4: Run the Aimbot

```sh
python TensorRT.py
```
- The control panel UI will appear. Customize your settings as needed.
- Hold the configured key (default: ALT) to activate aimbot or triggerbot when a target is detected.
- Press the panic key (default: END) to instantly disable all functionality.

---

## Features

- Real-time object detection using TensorRT
- Fully modular, class-based architecture
- Advanced aimbot with velocity prediction, smoothing, and adaptive speed
- Triggerbot with configurable, randomized click/release delays
- Modern, tabbed, and auto-saving control panel UI (Tkinter + ttk)
- Overlay UI with advanced bounding box smoothing and velocity prediction
- All settings configurable via UI and `config.json` (auto-created if missing)
- Multi-monitor support (select monitor in UI)
- Panic key (default: END) to instantly disable all functionality
- Performance monitoring and logging
- Robust error handling and clean resource cleanup
- High process priority for optimal performance

---

## Project Structure

- `TensorRT.py`: Main application file (modular, all logic in classes)
- `convert_to_trt.py`: Script to convert ONNX model to TensorRT engine
- `model1_320.onnx`: Included ONNX model (ready to convert)
- `mouse_driver/`: Contains mouse control implementation (LogiFck)
- `config.json`: User configuration file (auto-created/updated)
- `CHANGELOG.md`: Detailed changelog of all updates

---

## Configuration

Settings are saved in `config.json` and can be edited directly or through the UI. The config file is auto-created with defaults if missing.

Key settings include:
- `aimbot.enabled`: Enable aimbot functionality
- `aimbot.key`: Key to hold for activating aimbot
- `aimbot.speed`: Movement speed (1-1000)
- `aimbot.target_choice`: Target aim point (head, neck, chest, etc.)
- `aimbot.smoothing_factor`: Smoothing for aimbot and UI
- `aimbot.prediction_time`: Prediction time for velocity-based aiming
- `aimbot.panic_key`: Key to instantly disable all features
- `triggerbot.enabled`: Enable triggerbot functionality
- `triggerbot.key`: Key to hold for triggerbot
- `triggerbot.min_click_delay`/`max_click_delay`: Randomized click duration
- `triggerbot.min_release_delay`/`max_release_delay`: Randomized release duration
- `display.boxes_enabled`: Show bounding boxes
- `display.scanning_box_enabled`: Show scanning box
- `display.input_width`/`input_height`: Input resolution
- `system.monitor_index`: Monitor selection

---

## Troubleshooting

- **"Failed to initialize TensorRT engine"**: Ensure your TensorRT engine file is valid and compatible with your GPU/TensorRT version
- **"Mouse controller not initialized properly"**: Check if required DLL files are present; fallback stub will be used if missing
- **Low FPS**: Try reducing resolution in config or upgrade your GPU
- **Missing dependencies**: The app will print a clear error and exit if a required package is missing

Check the console output for logs. Use `--log-level DEBUG` for more detailed logs.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- TensorRT for high-speed inference
- OpenCV for image processing
- All contributors to the open-source ecosystem

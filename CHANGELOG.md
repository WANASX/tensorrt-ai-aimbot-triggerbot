# Changelog

## Major Overhaul

### General Architecture & Structure

- **Full Modularization**:  
  The codebase is now split into multiple classes (`Config`, `TRTEngineManager`, `AimbotController`, `TriggerBotController`, `ScreenCaptureManager`, `InferenceLoop`, `OverlayWindow`, `ControlPanel`). This replaces the old monolithic, global-variable-driven script with a maintainable, extensible, and testable architecture.

- **Configuration System**:  
  - Added a `Config` class that manages all configuration, supports loading/saving from `config.json`, and provides default values and type safety.
  - All runtime parameters (aimbot, triggerbot, display, keys, aiming, system) are now configurable via file and UI, not just hardcoded globals.
  - Auto-creation of a default config file if missing.

- **Command-Line Arguments**:  
  - Added `argparse` support for log level, model path, and config path.
  - Logging level is now user-configurable.

- **Logging**:  
  - Uses the `logging` module throughout, with log levels and structured messages.
  - Logs are more informative and less spammy, with error tracebacks.

- **Dependency Handling**:  
  - All imports are wrapped in try/except with user-friendly error messages and instructions for missing dependencies.
  - Fallback stub for missing mouse driver.

### UI/UX

- **Overlay & Control Panel**:
  - Overlay and control panel are now fully class-based (`OverlayWindow`, `ControlPanel`).
  - Control panel is modernized: tabbed, scrollable, compact, and styled with `ttk` and custom fonts/colors.
  - All settings are live-editable in the UI and auto-saved.
  - Added help window and improved exit handling.
  - Overlay window supports advanced smoothing and velocity prediction for bounding boxes.

- **UI Responsiveness**:
  - Overlay update loop runs at ~120 FPS for smooth animations (was 30 FPS).
  - UI updates are more robust to errors.

### Aimbot & Triggerbot

- **Aimbot**:
  - Now encapsulated in `AimbotController` with advanced smoothing, velocity prediction, and adaptive speed/precision.
  - Uses a history of positions and velocities for more stable and human-like movement.
  - Custom easing function with lookup table for performance.
  - Offset and prediction time are now configurable.

- **Triggerbot**:
  - Now encapsulated in `TriggerBotController`.
  - Click and release delays are configurable and randomized for humanization.
  - Thread-safe, with clean start/stop logic and event-based control.

### Inference & Performance

- **TensorRT Engine Management**:
  - All TensorRT logic is encapsulated in `TRTEngineManager`.
  - Handles initialization, error reporting, and resource cleanup.
  - Warm-up runs and error handling are improved.

- **Screen Capture**:
  - Now managed by `ScreenCaptureManager` with its own thread, FPS tracking, and error recovery.
  - Uses direct numpy slicing for performance (no more `cv2.cvtColor`).
  - Can restart itself on error.

- **Inference Loop**:
  - Main inference logic is now in `InferenceLoop` class.
  - Tracks and logs FPS, inference time, and detection status.
  - Handles all detection parsing, smoothing, and aimbot/triggerbot activation.
  - More robust to errors and interruptions.

### Threading & Synchronization

- **Thread Management**:
  - All threads are tracked in a global list for clean shutdown.
  - Uses `threading.Event` for exit, panic, and frame synchronization.
  - Clean resource cleanup on exit, including mouse/keyboard listeners and all threads.

- **Synchronization**:
  - Uses locks for frame, box, and click state to avoid race conditions.

### Features & Functionality

- **Panic Key**:
  - Added a panic key (configurable) to instantly disable all functionality.

- **Multi-Monitor Support**:
  - Monitor index is now configurable.

- **Advanced Smoothing**:
  - Both overlay and aimbot use advanced smoothing and velocity prediction for more natural behavior.

- **Auto-Save**:
  - All UI changes are auto-saved to config.

- **Error Handling**:
  - All major operations are wrapped in try/except with logging and recovery.

- **Process Priority**:
  - Attempts to set process priority to high for better performance (Windows and Unix).

### Minor/Other

- **Code Quality**:
  - Type hints and docstrings throughout.
  - More comments and explanations.
  - Consistent naming and style.

- **Extensibility**:
  - All major components are now classes, making it easy to extend or replace parts of the system.

- **Exit Handling**:
  - Clean exit on user request or error, with resource cleanup and forced exit if needed.

---

## Summary Table

| Area                | Old Version                | New Version (This Release)         |
|---------------------|---------------------------|------------------------------------|
| Config              | Hardcoded globals         | Config class, file, UI, CLI        |
| UI                  | Basic Tkinter, no tabs    | Modern, tabbed, styled, auto-save  |
| Aimbot/Triggerbot   | Global functions/vars     | Encapsulated, advanced smoothing   |
| Inference           | Inline, procedural        | Class-based, robust, performant    |
| Logging             | Basic, noisy              | Structured, level-based, tracebacks|
| Error Handling      | Minimal                   | Extensive, user-friendly           |
| Threading           | Ad-hoc, no cleanup        | Managed, tracked, clean shutdown   |
| Extensibility       | Hard to extend            | Modular, class-based               |
| Performance         | Good, but less robust     | Optimized, high-FPS, error recovery|
| Features            | Basic                     | Panic key, multi-monitor, more     |

---

## Migration Notes

- **Config file**: The new version will auto-create a `config.json` if missing. You can copy your old settings into this file if needed.
- **Dependencies**: Make sure to install all required dependencies as listed in the error message or `requirements.txt`.
- **Custom Mouse Driver**: If `LogiFck` is missing, the new version will still run, but with limited mouse functionality.

---

## Breaking Changes

- The script is no longer a single-file, global-variable-driven script. All settings and state are managed via classes and config files.
- The UI and config system are not backward compatible with the old hardcoded globals.

---

## Credits

- Major refactor and modernization by [Your Name/Team].
- Original code by [Original Author]. 
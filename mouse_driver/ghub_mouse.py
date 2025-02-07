# ghub_mouse.py

from ctypes import WinDLL, c_int, windll, get_last_error
from os import path
import logging
import time as t
import sys
import ctypes

# Configure logging with detailed debug information
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler("ghub_mouse_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LogiFck:
    def __init__(self, dll_path=None):
        """
        Initialize the LogiFck mouse controller.

        Args:
        - dll_path: The absolute path to the ghub_mouse.dll library.
        """
        if dll_path is None:
            self.dlldir = path.abspath(path.join(path.dirname(__file__), 'ghub_mouse.dll'))
        else:
            self.dlldir = path.abspath(dll_path)
        logger.debug(f"Initializing LogiFck with DLL path: {self.dlldir}")

        if not path.exists(self.dlldir):
            logger.error(f"LogiFck: DLL not found at path: {self.dlldir}")
            self.gmok = 0
            return

        try:
            logger.debug("Attempting to load ghub_mouse.dll using WinDLL with last_error=True...")
            self.gm = WinDLL(self.dlldir, use_last_error=True)
            logger.info("LogiFck: DLL loaded successfully with WinDLL.")

            # Define function signatures
            # Attempt to find mouse_open, if not, try decorated names
            self.mouse_open = self.get_function('mouse_open', [])
            self.moveR = self.get_function('moveR', [c_int, c_int])
            self.press = self.get_function('press', [c_int])
            self.release = self.get_function('release', [c_int])
            self.mouse_close = self.get_function('mouse_close', [])

            # Call mouse_open
            logger.debug("Calling mouse_open() to initialize mouse controller.")
            self.gmok = self.mouse_open()
            logger.debug(f"LogiFck: mouse_open() returned: {self.gmok}")

            if self.gmok != 0:
                logger.info("LogiFck: Mouse controller opened successfully.")
            else:
                error_code = windll.kernel32.GetLastError()
                logger.error(f"LogiFck: Failed to open mouse controller. GetLastError: {error_code}")
        except OSError as e:
            logger.error(f"LogiFck: OS error loading DLL: {e}")
            self.gmok = 0
        except AttributeError as e:
            logger.error(f"LogiFck: Attribute error - possibly missing function in DLL: {e}")
            self.gmok = 0
        except Exception as e:
            logger.error(f"LogiFck: Unexpected error during initialization: {e}")
            self.gmok = 0

    def get_function(self, func_name, argtypes):
        """
        Attempts to retrieve a function from the DLL, trying both undecorated and decorated names.

        Args:
        - func_name: The base name of the function.
        - argtypes: A list of ctypes types representing the function's arguments.

        Returns:
        - The function object if found, else raises AttributeError.
        """
        try:
            func = getattr(self.gm, func_name)
            func.restype = c_int
            func.argtypes = argtypes
            logger.debug(f"LogiFck: Function '{func_name}' loaded successfully.")
            return func
        except AttributeError:
            # Try stdcall decorated name, e.g., mouse_open@0
            decorated_name = f"{func_name}@{self.get_total_arg_size(argtypes)}"
            try:
                func = getattr(self.gm, decorated_name)
                func.restype = c_int
                func.argtypes = argtypes
                logger.debug(f"LogiFck: Function '{decorated_name}' loaded successfully.")
                return func
            except AttributeError:
                logger.error(f"LogiFck: Function '{func_name}' or '{decorated_name}' not found in DLL.")
                raise

    def get_total_arg_size(self, argtypes):
        """
        Helper function to get the total size in bytes of all ctypes argument types.

        Args:
        - argtypes: A list of ctypes types.

        Returns:
        - Total size in bytes.
        """
        total_size = 0
        for arg in argtypes:
            try:
                total_size += ctypes.sizeof(arg)
            except Exception as e:
                logger.warning(f"LogiFck: Could not determine size for arg {arg}: {e}")
                total_size += 0  # Default to 0 if size can't be determined
        return total_size

    def move_relative(self, x, y):
        """
        Move the mouse by the specified relative coordinates.

        Args:
        - x: The relative x-coordinate.
        - y: The relative y-coordinate.
        """
        if not self.gmok:
            logger.error("LogiFck: Cannot move mouse because controller is not open.")
            return -1

        logger.debug(f"LogiFck: Attempting to move mouse by ({x}, {y}).")
        try:
            result = self.moveR(x, y)
            logger.debug(f"LogiFck: moveR({x}, {y}) returned: {result}")
            if result != 0:
                logger.info(f"LogiFck: Moved mouse by ({x}, {y}).")
            else:
                error_code = windll.kernel32.GetLastError()
                logger.error(f"LogiFck: moveR failed. GetLastError: {error_code}")
            return result
        except AttributeError as e:
            logger.error(f"LogiFck: moveR function not found in DLL: {e}")
            return -1
        except Exception as e:
            logger.error(f"LogiFck: Exception during move_relative: {e}")
            return -1

    def mouse_down(self, key):
        """
        Simulate pressing a mouse button.

        Args:
        - key: The integer representing the button to press.
        """
        if not self.gmok:
            logger.error("LogiFck: Cannot press mouse button because controller is not open.")
            return -1

        logger.debug(f"LogiFck: Attempting to press mouse button with key code: {hex(key)}.")
        try:
            result = self.press(key)
            logger.debug(f"LogiFck: press({hex(key)}) returned: {result}")
            if result != 0:
                logger.info(f"LogiFck: Mouse button {hex(key)} pressed.")
            else:
                error_code = windll.kernel32.GetLastError()
                logger.error(f"LogiFck: Failed to press mouse button {hex(key)}. GetLastError: {error_code}")
            return result
        except AttributeError as e:
            logger.error(f"LogiFck: press function not found in DLL: {e}")
            return -1
        except Exception as e:
            logger.error(f"LogiFck: Exception during mouse_down: {e}")
            return -1

    def mouse_up(self, key):
        """
        Simulate releasing a mouse button.

        Args:
        - key: The integer representing the button to release.
        """
        if not self.gmok:
            logger.error("LogiFck: Cannot release mouse button because controller is not open.")
            return -1

        logger.debug(f"LogiFck: Attempting to release mouse button with key code: {hex(key)}.")
        try:
            result = self.release(key)
            logger.debug(f"LogiFck: release({hex(key)}) returned: {result}")
            if result != 0:
                logger.info(f"LogiFck: Mouse button {hex(key)} released.")
            else:
                error_code = windll.kernel32.GetLastError()
                logger.error(f"LogiFck: Failed to release mouse button {hex(key)}. GetLastError: {error_code}")
            return result
        except AttributeError as e:
            logger.error(f"LogiFck: release function not found in DLL: {e}")
            return -1
        except Exception as e:
            logger.error(f"LogiFck: Exception during mouse_up: {e}")
            return -1

    def close(self):
        """
        Close the LogiFck mouse controller.
        """
        if not self.gmok:
            logger.warning("LogiFck: Mouse controller is already closed or was never opened.")
            return -1

        logger.debug("LogiFck: Attempting to close mouse controller.")
        try:
            result = self.mouse_close()
            logger.debug(f"LogiFck: mouse_close() returned: {result}")
            if result != 0:
                logger.info("LogiFck: Mouse controller closed successfully.")
            else:
                error_code = windll.kernel32.GetLastError()
                logger.error(f"LogiFck: Failed to close mouse controller. GetLastError: {error_code}")
            self.gmok = 0
            return result
        except AttributeError as e:
            logger.error(f"LogiFck: mouse_close function not found in DLL: {e}")
            return -1
        except Exception as e:
            logger.error(f"LogiFck: Exception during close: {e}")
            return -1

# Example usage:
if __name__ == "__main__":
    logger.info("Starting LogiFck test script.")


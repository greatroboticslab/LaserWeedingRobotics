import cv2
import numpy as np
import ctypes
import json
import time
import os
from scipy.spatial import KDTree
import serial
import threading
import serial.tools.list_ports


# Define Helios point structure
class HeliosPoint(ctypes.Structure):
    _fields_ = [('x', ctypes.c_uint16),
                ('y', ctypes.c_uint16),
                ('r', ctypes.c_uint8),
                ('g', ctypes.c_uint8),
                ('b', ctypes.c_uint8),
                ('i', ctypes.c_uint8)]


class LaserClickTesterDual:  # Renamed class for clarity
    def __init__(self, calibration_file_motor0='calibration_data_motor_0.json',
                 calibration_file_motor1='calibration_data_motor_1.json'):

        # --- Dual Motor Data Storage ---
        self.calibration_points_per_motor = [[] for _ in range(2)]
        self.region_corners_laser_per_motor = [[] for _ in range(2)]
        self.region_corners_camera_per_motor = [[] for _ in range(2)]
        self.kdtree_per_motor = [None, None]
        self.valid_calibration_indices_per_motor = [[] for _ in range(2)]
        self.last_match_info_per_motor = [None, None]  # Store match info separately

        # ESP32 laser control variables
        self.esp32 = None
        self.laser_enabled = False  # Laser state (ON/OFF)
        self.laser_power = 128  # Laser power (0-255)
        self.esp32_connected = False

        # Power slider variables
        self.slider_x = 0  # Will be set based on frame width
        self.slider_y = 50
        self.slider_width = 200
        self.slider_height = 20
        self.slider_handle_width = 15
        self.slider_dragging = False

        # Load data for both motors
        print("--- Loading Calibration Data ---")
        self.load_calibration_data(0, calibration_file_motor0)
        self.load_calibration_data(1, calibration_file_motor1)
        print("--------------------------------")

        # Prepare KD-trees for both motors
        self.prepare_kdtree(0)
        self.prepare_kdtree(1)

        # Transformation method (applied to both motors)
        self.transform_method = 'nearest'  # 'nearest', 'weighted', 'homography'
        self.weighted_k = 4  # Number of neighbors for weighted interpolation

        # Laser control settings (shared)
        self.LASER_MAX = 0xFFF  # 4095 (12-bit resolution)
        self.POINTS_PER_FRAME = 1000
        self.FRAME_DURATION = 30000
        self.COLOR_VALUE = 255
        self.INTENSITY = 130

        # Current laser position (DAC values for each motor)
        # Initialize to center, actual position depends on transforms
        self.current_x_per_motor = [self.LASER_MAX // 2, self.LASER_MAX // 2]
        self.current_y_per_motor = [self.LASER_MAX // 2, self.LASER_MAX // 2]

        # Last clicked position on camera view
        self.last_click_pos = None

        # Load laser device
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_dir, "HeliosLaserDAC.dll")
            if not os.path.exists(dll_path):
                dll_path = "./HeliosLaserDAC.dll"  # Fallback
            self.HeliosLib = ctypes.cdll.LoadLibrary(dll_path)
        except OSError as e:
            print(f"Error loading HeliosLaserDAC.dll: {e}")
            raise

        self.numDevices = self.HeliosLib.OpenDevices()
        print(f"Found {self.numDevices} Helios DACs")

        if self.numDevices == 0:
            raise Exception("No Helios DAC found. Please check connection.")
        if self.numDevices < 2:
            print("Warning: Fewer than 2 DAC devices found. Motor 2 control might not function.")

        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")

        # Set slider position based on frame width
        self.slider_x = self.frame_width - self.slider_width - 20

        # Initialize ESP32 connection
        self.connect_to_esp32()

        # Create window and set mouse callback
        cv2.namedWindow("Laser Click Test - Dual Motor")
        cv2.setMouseCallback("Laser Click Test - Dual Motor", self.mouse_callback)

        # Text info display
        self.show_help = True
        self.show_calibration_points = True  # Shows points for both motors if loaded

    def connect_to_esp32(self):
        """Connect to ESP32 for laser control"""
        try:
            # List available ports
            ports = serial.tools.list_ports.comports()
            print("Available serial ports:")
            for port in ports:
                print(f"  {port}")

            # Try to find ESP32 port automatically
            esp32_port = None
            for port in ports:
                if 'usbserial' in port.device.lower() or 'ch340' in port.description.lower() or 'cp210' in port.description.lower():
                    esp32_port = port.device
                    break

            if not esp32_port:
                # If auto-detection fails, try common port names
                common_ports = ['/dev/cu.usbserial-0001', 'COM3', 'COM4', 'COM5', '/dev/ttyUSB0', '/dev/ttyACM0']
                for port_name in common_ports:
                    try:
                        test_serial = serial.Serial(port_name, 115200, timeout=1)
                        test_serial.close()
                        esp32_port = port_name
                        break
                    except:
                        continue

            if esp32_port:
                self.esp32 = serial.Serial(esp32_port, 115200, timeout=1)
                print(f"Successfully connected to ESP32 on {esp32_port}")
                time.sleep(2)  # Give ESP32 time to initialize

                # Ensure laser is initially OFF
                self.send_laser_command("OFF", False)
                self.send_laser_command(f"POWER {self.laser_power}", False)

                self.esp32_connected = True
                print("ESP32 laser control initialized - Laser is OFF by default")
            else:
                print("Warning: Could not find ESP32 port. Laser control will not be available.")
                print("Available ports were:", [port.device for port in ports])
                self.esp32_connected = False

        except Exception as e:
            print(f"Error connecting to ESP32: {e}")
            print("Laser control will not be available.")
            self.esp32_connected = False

    def send_laser_command(self, cmd, read_response=True):
        """Send command to ESP32"""
        if not self.esp32_connected or not self.esp32 or not self.esp32.is_open:
            return

        try:
            self.esp32.write(f"{cmd}\n".encode())

            if not read_response:
                return

            # Read response in a separate thread to avoid blocking
            def read_response_thread():
                time.sleep(0.05)
                try:
                    if self.esp32 and self.esp32.is_open and self.esp32.in_waiting > 0:
                        response = self.esp32.readline().decode('utf-8', errors='replace').strip()
                        if response:
                            print(f"ESP32: {response}")
                except Exception as e:
                    print(f"Error reading ESP32 response: {e}")

            threading.Thread(target=read_response_thread, daemon=True).start()

        except Exception as e:
            print(f"Error sending command to ESP32: {e}")

    def toggle_laser(self):
        """Toggle laser ON/OFF"""
        if not self.esp32_connected:
            print("ESP32 not connected - cannot control laser")
            return

        self.laser_enabled = not self.laser_enabled
        if self.laser_enabled:
            self.send_laser_command("ON", False)
            print("Laser turned ON")
        else:
            self.send_laser_command("OFF", False)
            print("Laser turned OFF")

    def set_laser_power(self, power):
        """Set laser power (0-255)"""
        if not self.esp32_connected:
            return

        power = max(0, min(255, int(power)))
        self.laser_power = power
        self.send_laser_command(f"POWER {power}", False)

    def draw_power_slider(self, frame):
        """Draw power control slider on frame"""
        # Slider background
        cv2.rectangle(frame,
                      (self.slider_x, self.slider_y),
                      (self.slider_x + self.slider_width, self.slider_y + self.slider_height),
                      (60, 60, 60), -1)

        # Slider track
        track_y = self.slider_y + self.slider_height // 2
        cv2.line(frame,
                 (self.slider_x + self.slider_handle_width // 2, track_y),
                 (self.slider_x + self.slider_width - self.slider_handle_width // 2, track_y),
                 (100, 100, 100), 2)

        # Calculate handle position based on power
        handle_range = self.slider_width - self.slider_handle_width
        handle_pos = self.slider_x + (self.laser_power / 255.0) * handle_range

        # Slider handle
        cv2.rectangle(frame,
                      (int(handle_pos), self.slider_y),
                      (int(handle_pos + self.slider_handle_width), self.slider_y + self.slider_height),
                      (0, 255, 0) if self.laser_enabled else (100, 100, 100), -1)

        # Power value text
        power_text = f"Power: {self.laser_power}"
        cv2.putText(frame, power_text,
                    (self.slider_x, self.slider_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Laser status text
        status_text = "Laser: ON" if self.laser_enabled else "Laser: OFF"
        status_color = (0, 255, 0) if self.laser_enabled else (0, 0, 255)
        cv2.putText(frame, status_text,
                    (self.slider_x, self.slider_y + self.slider_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    def handle_slider_mouse(self, event, x, y):
        """Handle mouse events for power slider"""
        # Check if mouse is over slider
        if (self.slider_x <= x <= self.slider_x + self.slider_width and
                self.slider_y <= y <= self.slider_y + self.slider_height):

            if event == cv2.EVENT_LBUTTONDOWN:
                self.slider_dragging = True
                # Update power based on click position
                relative_x = x - self.slider_x
                power = int((relative_x / self.slider_width) * 255)
                self.set_laser_power(power)
                return True

        if event == cv2.EVENT_LBUTTONUP:
            self.slider_dragging = False

        if event == cv2.EVENT_MOUSEMOVE and self.slider_dragging:
            if self.slider_x <= x <= self.slider_x + self.slider_width:
                relative_x = x - self.slider_x
                power = int((relative_x / self.slider_width) * 255)
                self.set_laser_power(power)
                return True

        return False

    def load_calibration_data(self, motor_index, filename):
        """Load calibration data for a specific motor"""
        if not (0 <= motor_index <= 1):
            print(f"Error: Invalid motor_index {motor_index} for loading data.")
            return False

        print(f"Attempting to load data for Motor {motor_index + 1} from {filename}...")
        if not os.path.exists(filename):
            print(f"  File not found.")
            self.calibration_points_per_motor[motor_index] = []
            self.region_corners_laser_per_motor[motor_index] = []
            self.region_corners_camera_per_motor[motor_index] = []
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Load data into the correct motor's slot
            self.calibration_points_per_motor[motor_index] = data.get('calibration_points', [])
            self.region_corners_laser_per_motor[motor_index] = data.get('region_corners_laser', [])
            self.region_corners_camera_per_motor[motor_index] = data.get('region_corners_camera', [])

            print(
                f"  Loaded {len(self.calibration_points_per_motor[motor_index])} calibration points for Motor {motor_index + 1}.")
            return True
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"  Failed to load or parse calibration data for Motor {motor_index + 1}: {e}")
            # Clear data for this motor on error
            self.calibration_points_per_motor[motor_index] = []
            self.region_corners_laser_per_motor[motor_index] = []
            self.region_corners_camera_per_motor[motor_index] = []
            return False

    def prepare_kdtree(self, motor_index):
        """Prepare KD tree for a specific motor"""
        if not (0 <= motor_index <= 1): return False

        calibration_points = self.calibration_points_per_motor[motor_index]
        if len(calibration_points) < 1:
            print(f"Warning: Not enough calibration points for Motor {motor_index + 1} to build KD tree.")
            self.kdtree_per_motor[motor_index] = None
            self.valid_calibration_indices_per_motor[motor_index] = []
            return False

        valid_points = []
        valid_indices = []
        for i, point in enumerate(calibration_points):
            if ('camera_pixel_x' in point and 'camera_pixel_y' in point and
                    isinstance(point['camera_pixel_x'], (int, float)) and  # Check type too
                    isinstance(point['camera_pixel_y'], (int, float))):
                valid_points.append([point['camera_pixel_x'], point['camera_pixel_y']])
                valid_indices.append(i)

        if not valid_points:
            print(f"Warning: No valid calibration points with camera coordinates found for Motor {motor_index + 1}.")
            self.kdtree_per_motor[motor_index] = None
            self.valid_calibration_indices_per_motor[motor_index] = []
            return False

        try:
            camera_points_np = np.array(valid_points, dtype=np.float32)
            self.kdtree_per_motor[motor_index] = KDTree(camera_points_np)
            self.valid_calibration_indices_per_motor[motor_index] = valid_indices
            print(f"Built KD tree for Motor {motor_index + 1} based on {len(valid_points)} valid calibration points.")
            return True
        except Exception as e:
            print(f"Error building KD tree for Motor {motor_index + 1}: {e}")
            self.kdtree_per_motor[motor_index] = None
            self.valid_calibration_indices_per_motor[motor_index] = []
            return False

    def is_point_in_region(self, motor_index, point_x, point_y):
        """Check if point is in defined region for the specified motor"""
        if not (0 <= motor_index <= 1): return False  # Invalid motor index

        region_corners_camera = self.region_corners_camera_per_motor[motor_index]
        if len(region_corners_camera) < 3:
            return True  # Assume in region if not defined

        try:
            poly = np.array(region_corners_camera, dtype=np.int32).reshape((-1, 1, 2))
            point = (float(point_x), float(point_y))  # Use float for pointPolygonTest
            result = cv2.pointPolygonTest(poly, point, False)
            return result >= 0
        except Exception as e:
            print(f"Error in pointPolygonTest for Motor {motor_index + 1}: {e}")
            return False  # Treat error as outside region

    # --- Transformation Functions (modified to accept motor_index) ---

    def transform_using_nearest(self, motor_index, camera_x, camera_y):
        """Transform coordinates for a specific motor using nearest neighbor"""
        if not (0 <= motor_index <= 1): return self.LASER_MAX // 2, self.LASER_MAX // 2

        kdtree = self.kdtree_per_motor[motor_index]
        valid_indices = self.valid_calibration_indices_per_motor[motor_index]
        calibration_points = self.calibration_points_per_motor[motor_index]

        if kdtree is None or not valid_indices:
            # Fallback to simple linear mapping if KD-tree not available
            print(f"Warning: KD-Tree unavailable for Motor {motor_index + 1}. Using simple mapping.")
            return self._simple_linear_map(camera_x, camera_y)

        try:
            dist, kdtree_idx = kdtree.query([camera_x, camera_y], k=1)
            original_idx = valid_indices[kdtree_idx]  # Map KD-tree index back to original list index
            nearest_point = calibration_points[original_idx]

            laser_x = int(nearest_point['laser_x'])  # Ensure int
            laser_y = int(nearest_point['laser_y'])  # Ensure int

            self.last_match_info_per_motor[motor_index] = {  # Store match info
                'camera_x': camera_x, 'camera_y': camera_y, 'nearest_idx': original_idx,
                'distance': dist, 'nearest_point': nearest_point
            }
            # print(f"M{motor_index+1} Nearest: D={dist:.1f}, Idx={original_idx} -> Laser({laser_x},{laser_y})") # Debug
            return laser_x, laser_y
        except Exception as e:
            print(f"Error during nearest neighbor search for Motor {motor_index + 1}: {e}")
            return self._simple_linear_map(camera_x, camera_y)

    def transform_using_weighted(self, motor_index, camera_x, camera_y):
        """Transform coordinates for a specific motor using weighted K nearest neighbors"""
        if not (0 <= motor_index <= 1): return self.LASER_MAX // 2, self.LASER_MAX // 2

        kdtree = self.kdtree_per_motor[motor_index]
        valid_indices = self.valid_calibration_indices_per_motor[motor_index]
        calibration_points = self.calibration_points_per_motor[motor_index]
        k = min(self.weighted_k, len(valid_indices))  # Ensure k is not larger than available points

        if kdtree is None or len(valid_indices) < k or k < 1:
            # Fallback to nearest neighbor if weighted is not possible
            # print(f"Warning: Weighted K-NN not possible for Motor {motor_index+1}. Falling back to nearest.") # Debug
            return self.transform_using_nearest(motor_index, camera_x, camera_y)

        try:
            dists, kdtree_idxs = kdtree.query([camera_x, camera_y], k=k)

            # Handle case where query returns single value instead of array for k=1
            if k == 1 and not isinstance(dists, (np.ndarray, list)):
                dists = [dists]
                kdtree_idxs = [kdtree_idxs]

            original_idxs = [valid_indices[i] for i in kdtree_idxs]

            # Ensure distances are positive for weighting
            weights = 1.0 / (np.maximum(dists, 1e-9) ** 2)  # Inverse distance squared, avoid zero
            if np.sum(weights) == 0:  # Handle case where all distances might be huge -> weights near zero
                weights = np.ones_like(weights)  # Fallback to equal weights

            weights /= np.sum(weights)  # Normalize

            laser_x_weighted = 0.0
            laser_y_weighted = 0.0
            for i, orig_idx in enumerate(original_idxs):
                point = calibration_points[orig_idx]
                laser_x_weighted += point['laser_x'] * weights[i]
                laser_y_weighted += point['laser_y'] * weights[i]

            laser_x = int(round(laser_x_weighted))
            laser_y = int(round(laser_y_weighted))

            self.last_match_info_per_motor[motor_index] = {  # Store match info
                'camera_x': camera_x, 'camera_y': camera_y, 'nearest_idxs': original_idxs,
                'distances': dists, 'weights': weights
            }
            # print(f"M{motor_index+1} Weighted (k={k}): Nearest D={np.min(dists):.1f} -> Laser({laser_x},{laser_y})") # Debug
            return laser_x, laser_y
        except Exception as e:
            print(f"Error during weighted K-NN search for Motor {motor_index + 1}: {e}")
            return self.transform_using_nearest(motor_index, camera_x, camera_y)  # Fallback

    def _simple_linear_map(self, camera_x, camera_y):
        """Fallback: Simple linear mapping from camera to laser DAC range"""
        # Basic scaling, assumes camera top-left is laser (0, MAX) and bottom-right is (MAX, 0)
        # This is a very rough guess and likely incorrect without calibration.
        laser_x = int((camera_x / self.frame_width) * self.LASER_MAX)
        laser_y = int(((self.frame_height - camera_y) / self.frame_height) * self.LASER_MAX)
        # Constrain values
        laser_x = max(0, min(laser_x, self.LASER_MAX))
        laser_y = max(0, min(laser_y, self.LASER_MAX))
        # print(f"Simple mapping: Cam({camera_x},{camera_y}) -> Laser({laser_x},{laser_y})") # Debug
        return laser_x, laser_y

    def transform_using_homography(self, motor_index, camera_x, camera_y):
        """ DEPRECATED / PLACEHOLDER: Homography calculation should happen in calibration"""
        # This tester script should ideally load a pre-computed matrix,
        # but the original script didn't have that logic split well.
        # For now, fallback to simple mapping.
        print(f"Warning: Homography transform requested for Motor {motor_index + 1}, but not implemented "
              f"in tester. Using simple mapping.")
        return self._simple_linear_map(camera_x, camera_y)

    def transform_coordinates(self, motor_index, camera_x, camera_y):
        """Transform camera pixel coordinates to laser coordinates for a specific motor"""
        if not (0 <= motor_index <= 1):
            print(f"Error: Invalid motor_index {motor_index} for transform.")
            return self.LASER_MAX // 2, self.LASER_MAX // 2  # Default center

        if self.transform_method == 'nearest':
            return self.transform_using_nearest(motor_index, camera_x, camera_y)
        elif self.transform_method == 'weighted':
            return self.transform_using_weighted(motor_index, camera_x, camera_y)
        elif self.transform_method == 'homography':
            # Homography should ideally be loaded, using fallback for now
            return self.transform_using_homography(motor_index, camera_x, camera_y)
        else:  # Default
            return self.transform_using_nearest(motor_index, camera_x, camera_y)

    # --- Laser Control Functions (modified for specific motor) ---

    def create_frame_data(self, x_center, y_center):
        """Create laser frame data buffer (HeliosPoint array)"""
        # This function remains the same, it just creates the point data
        frame_buffer = (HeliosPoint * self.POINTS_PER_FRAME)()
        x_center_int = int(x_center)
        y_center_int = int(y_center)
        for j in range(self.POINTS_PER_FRAME):
            frame_buffer[j] = HeliosPoint(x=x_center_int, y=y_center_int,
                                          r=self.COLOR_VALUE, g=self.COLOR_VALUE, b=self.COLOR_VALUE,
                                          i=self.INTENSITY)
        return frame_buffer

    def send_frame_to_motor(self, frame_buffer, motor_idx):
        """Send frame buffer to a specific motor device"""
        if motor_idx < self.numDevices:
            statusAttempts = 0
            # Wait briefly for DAC status to become ready (status=1)
            while statusAttempts < 32 and self.HeliosLib.GetStatus(motor_idx) != 1:
                statusAttempts += 1
                time.sleep(0.001)  # Short delay for polling

            if self.HeliosLib.GetStatus(motor_idx) == 1:
                # Send the frame
                result = self.HeliosLib.WriteFrame(motor_idx, self.FRAME_DURATION, 0,  # 0 = Play flags (e.g., once)
                                                   ctypes.pointer(frame_buffer), self.POINTS_PER_FRAME)
                # Optional: Check result if WriteFrame returns status
                # if result != 0: print(f"Warning: WriteFrame for motor {motor_idx} returned {result}")
            else:
                print(f"Warning: Helios DAC device {motor_idx} not ready.")
        else:
            print(f"Warning: Motor index {motor_idx} unavailable ({self.numDevices} devices found).")

    def move_motor_to_position(self, motor_index, x_dac, y_dac):
        """Move a specific motor to specified DAC position"""
        if not (0 <= motor_index <= 1): return  # Validate index

        x_dac = max(0, min(int(x_dac), self.LASER_MAX))
        y_dac = max(0, min(int(y_dac), self.LASER_MAX))

        # Update internal record of this motor's position
        self.current_x_per_motor[motor_index] = x_dac
        self.current_y_per_motor[motor_index] = y_dac

        # Create and send frame ONLY to the specified motor
        frame_buffer = self.create_frame_data(x_dac, y_dac)
        self.send_frame_to_motor(frame_buffer, motor_index)

        # print(f"Sent Motor {motor_index + 1} to DAC: X={x_dac}, Y={y_dac}") # Debug

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event handler for dual motor control"""
        # First check if mouse is interacting with power slider
        if self.handle_slider_mouse(event, x, y):
            return

        # Original mouse callback functionality
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click_pos = (x, y)
            print(f"\nClick detected at CameraPx({x}, {y})")

            # Check if click is within regions (optional, can be noisy)
            in_region_m0 = self.is_point_in_region(0, x, y)
            in_region_m1 = self.is_point_in_region(1, x, y)
            # print(f"  In Region M1: {in_region_m0}, In Region M2: {in_region_m1}") # Debug

            # Transform coordinates for Motor 0 (if data exists)
            if self.kdtree_per_motor[0] is not None:
                laser_x_m0, laser_y_m0 = self.transform_coordinates(0, x, y)
                print(f"  Calculated for Motor 1: LaserDAC({laser_x_m0}, {laser_y_m0})")
                self.move_motor_to_position(0, laser_x_m0, laser_y_m0)
            else:
                print(f"  Skipping Motor 1: No calibration data loaded.")

            # Transform coordinates for Motor 1 (if data exists and device available)
            if self.kdtree_per_motor[1] is not None and self.numDevices > 1:
                laser_x_m1, laser_y_m1 = self.transform_coordinates(1, x, y)
                print(f"  Calculated for Motor 2: LaserDAC({laser_x_m1}, {laser_y_m1})")
                self.move_motor_to_position(1, laser_x_m1, laser_y_m1)
            elif self.numDevices <= 1:
                print(f"  Skipping Motor 2: Only one DAC device detected.")
            else:
                print(f"  Skipping Motor 2: No calibration data loaded.")

    # --- Drawing Functions ---

    def draw_region(self, frame, motor_index):
        """Draw defined region for a specific motor"""
        if not (0 <= motor_index <= 1): return frame

        region_corners_camera = self.region_corners_camera_per_motor[motor_index]
        if len(region_corners_camera) < 3: return frame

        # Choose color based on motor index
        color = (0, 255, 0) if motor_index == 0 else (255, 0, 0)  # Green M1, Blue M2
        label_prefix = "C" if motor_index == 0 else "V"

        try:
            # Draw polygon outline
            points = np.array(region_corners_camera, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=1)

            # Draw corners
            for i, corner in enumerate(region_corners_camera):
                pt = tuple(map(int, corner))
                cv2.circle(frame, pt, 4, color, -1)
                # cv2.putText(frame, f"{label_prefix}{i+1}", (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1) # Optional labels
        except Exception as e:
            print(f"Error drawing region for motor {motor_index + 1}: {e}")

        return frame

    def draw_calibration_points(self, frame, motor_index):
        """Draw calibration points for a specific motor"""
        if not self.show_calibration_points or not (0 <= motor_index <= 1): return frame

        calibration_points = self.calibration_points_per_motor[motor_index]
        if not calibration_points: return frame

        color = (150, 150, 255) if motor_index == 0 else (255, 150, 150)  # Light Red M1, Light Blue M2

        # Draw all points as small dots
        for point in calibration_points:
            if 'camera_pixel_x' in point and 'camera_pixel_y' in point:
                try:
                    px = int(point['camera_pixel_x'])
                    py = int(point['camera_pixel_y'])
                    cv2.circle(frame, (px, py), 1, color, -1)  # Very small dots
                except (ValueError, TypeError):
                    continue  # Skip if coords are invalid

        # Visualize matching info if available for this motor
        last_match_info = self.last_match_info_per_motor[motor_index]
        if last_match_info and self.last_click_pos:
            click_x, click_y = int(self.last_click_pos[0]), int(self.last_click_pos[1])

            if self.transform_method == 'nearest' and 'nearest_point' in last_match_info:
                np = last_match_info['nearest_point']
                npx, npy = int(np['camera_pixel_x']), int(np['camera_pixel_y'])
                dist = last_match_info['distance']
                cv2.line(frame, (click_x, click_y), (npx, npy), (0, 200, 200), 1)
                cv2.circle(frame, (npx, npy), 4, (0, 0, 255), -1)  # Highlight nearest
                # cv2.putText(frame, f"d:{dist:.1f}", (npx + 5, npy - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), 1)

            elif self.transform_method == 'weighted' and 'nearest_idxs' in last_match_info:
                idxs = last_match_info.get('nearest_idxs', [])
                weights = last_match_info.get('weights', [])
                for i, orig_idx in enumerate(idxs):
                    if orig_idx < len(calibration_points):
                        point = calibration_points[orig_idx]
                        px, py = int(point['camera_pixel_x']), int(point['camera_pixel_y'])
                        thickness = max(1, int(weights[i] * 5))
                        cv2.line(frame, (click_x, click_y), (px, py), (255, 0, 255), thickness)  # Magenta lines
                        # cv2.circle(frame, (px, py), 3, (255,0,255), -1) # Highlight neighbors
        return frame

    def run(self):
        """Main loop"""
        # Move both motors initially (if possible)
        if self.numDevices > 0:
            self.move_motor_to_position(0, self.current_x_per_motor[0], self.current_y_per_motor[0])
        if self.numDevices > 1:
            self.move_motor_to_position(1, self.current_x_per_motor[1], self.current_y_per_motor[1])

        print("\nDual Laser Click Test with ESP32 Control - Instructions:")
        print("-------------------------------------------------------")
        print("  Click on the video feed to aim BOTH lasers.")
        print("  Uses calibration_data_motor_0.json for Motor 1")
        print("  Uses calibration_data_motor_1.json for Motor 2")
        print("-------------------------------------------------------")
        print("  ESP32 Laser Control:")
        print("  L: Toggle laser ON/OFF")
        print("  Use power slider in top-right corner to adjust laser power")
        print("-------------------------------------------------------")
        print("  H: Toggle Help | C: Toggle Calibration Points | M: Switch Method | +/-: Adjust K")
        print("  ESC: Exit")
        print("-------------------------------------------------------\n")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Error: Failed to grab frame")
                time.sleep(0.1)
                continue

            # Draw regions and points for both motors
            display_frame = frame.copy()
            display_frame = self.draw_region(display_frame, 0)  # Draw region for motor 0
            display_frame = self.draw_region(display_frame, 1)  # Draw region for motor 1
            display_frame = self.draw_calibration_points(display_frame, 0)  # Draw points for motor 0
            display_frame = self.draw_calibration_points(display_frame, 1)  # Draw points for motor 1

            # Draw power slider
            self.draw_power_slider(display_frame)

            # Mark last clicked position (shared for both motors)
            if self.last_click_pos:
                x, y = self.last_click_pos
                cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)  # Red center dot
                cv2.circle(display_frame, (x, y), 10, (0, 255, 255), 1)  # Yellow circle outline
                # cv2.putText(display_frame, f"Click: ({x}, {y})", (x + 12, y - 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)

            # Display Status
            method_text = f"Method: {self.transform_method.capitalize()}"
            if self.transform_method == 'weighted': method_text += f" (K={self.weighted_k})"
            cv2.putText(display_frame, method_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 250), 1)

            laser_text_m0 = f"M1 DAC: ({self.current_x_per_motor[0]}, {self.current_y_per_motor[0]})"
            cv2.putText(display_frame, laser_text_m0, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 250), 1)
            if self.numDevices > 1:
                laser_text_m1 = f"M2 DAC: ({self.current_x_per_motor[1]}, {self.current_y_per_motor[1]})"
                cv2.putText(display_frame, laser_text_m1, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 250), 1)

            # Display ESP32 connection status
            esp32_status = "ESP32: Connected" if self.esp32_connected else "ESP32: Not Connected"
            esp32_color = (0, 255, 0) if self.esp32_connected else (0, 0, 255)
            cv2.putText(display_frame, esp32_status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, esp32_color, 1)

            if self.show_help:
                help_lines = ["H:Help", "C:Cal Pts", "M:Method", "+/-: K", "L:Laser", "ESC:Exit"]
                y_start = 125
                for i, line in enumerate(help_lines):
                    cv2.putText(display_frame, line, (10, y_start + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1)

            # Show the frame
            cv2.imshow("Laser Click Test - Dual Motor", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break  # ESC
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help
            elif key == ord('c') or key == ord('C'):
                self.show_calibration_points = not self.show_calibration_points
            elif key == ord('l') or key == ord('L'):
                self.toggle_laser()
            elif key == ord('m') or key == ord('M'):
                methods = ['nearest', 'weighted']  # Removed 'homography' as fallback
                current_idx = methods.index(self.transform_method) if self.transform_method in methods else 0
                next_idx = (current_idx + 1) % len(methods)
                self.transform_method = methods[next_idx]
                print(f"Switched transform method to: {self.transform_method}")
            elif key == ord('+') or key == ord('='):
                if self.transform_method == 'weighted':
                    self.weighted_k = min(self.weighted_k + 1, 10)
                    print(f"Weighted K neighbors = {self.weighted_k}")
            elif key == ord('-'):
                if self.transform_method == 'weighted':
                    self.weighted_k = max(self.weighted_k - 1, 1)
                    print(f"Weighted K neighbors = {self.weighted_k}")

        # Turn off laser before exit
        if self.esp32_connected:
            self.send_laser_command("OFF", False)
            print("Laser turned OFF before exit")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.HeliosLib: self.HeliosLib.CloseDevices()

        if self.esp32 and self.esp32.is_open:
            self.esp32.close()
            print("ESP32 connection closed")

        print("Program ended.")


if __name__ == "__main__":
    tester = None  # Initialize
    try:
        # --- Specify calibration files directly here ---
        file_m0 = 'calibration_data_motor_0.json'
        file_m1 = 'calibration_data_motor_1.json'
        tester = LaserClickTesterDual(calibration_file_motor0=file_m0,
                                      calibration_file_motor1=file_m1)
        tester.run()
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Ensuring resources are released...")
        if tester:
            if hasattr(tester, 'cap') and tester.cap and tester.cap.isOpened():
                tester.cap.release()
                print("Camera released.")
            if hasattr(tester, 'HeliosLib') and tester.HeliosLib:
                # Optional: Could call CloseDevices again, but it's in tester.run()'s finally block
                pass
            if hasattr(tester, 'esp32') and tester.esp32 and tester.esp32.is_open:
                tester.esp32.write(b'OFF\n')  # Turn off laser
                tester.esp32.close()
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")
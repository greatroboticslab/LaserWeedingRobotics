import ctypes
import cv2
import numpy as np
import time
import keyboard
import json
import os
import random
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


class RegionBasedCalibration:
    def __init__(self):
        # Laser control settings
        self.LASER_MAX = 0xFFF  # 4095 (12-bit resolution)
        self.POINTS_PER_FRAME = 1000
        self.FRAME_DURATION = 30000
        self.COLOR_VALUE = 255
        self.INTENSITY = 130
        self.SINGLE_STEP = 5

        # Current laser position
        self.current_x = self.LASER_MAX // 2
        self.current_y = self.LASER_MAX // 2

        # Calibration points list
        self.calibration_points = []

        # Region definition points (four corners) - laser coordinates
        self.region_corners_laser = []

        # Region definition points - camera coordinates
        self.region_corners_camera = []

        # Background frame (will be acquired during initialization)
        self.background_frame = None

        # Debug mode
        self.debug_mode = True

        # Calibration state
        self.calibration_state = "DEFINE_REGION"  # DEFINE_REGION, COLLECT_POINTS, COMPLETED
        self.corner_index = 0  # Current corner index to set

        # Grid calibration parameters
        self.grid_step_x = 50  # X direction step size in laser coordinate system
        self.grid_step_y = 50  # Y direction step size in laser coordinate system

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

        # Load laser device
        self.HeliosLib = ctypes.cdll.LoadLibrary("./HeliosLaserDAC.dll")
        self.numDevices = self.HeliosLib.OpenDevices()
        print(f"Found {self.numDevices} Helios DACs")

        if self.numDevices == 0:
            raise Exception("No Helios DAC found. Please check connection.")

        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # Get camera resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")

        # Set slider position based on frame width
        self.slider_x = self.frame_width - self.slider_width - 20

        # Initialize ESP32 connection
        self.connect_to_esp32()

        # Get initial background frame
        print("Getting initial background frame...")
        self.get_background_frame()

        # Create window and set mouse callback
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

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

    def get_background_frame(self):
        """Get background frame (without laser)"""
        # Store current laser position
        current_x, current_y = self.current_x, self.current_y

        # Move laser to corner (equivalent to turning off)
        self.move_to_position(0, 0)
        time.sleep(0.5)

        # Get multiple background frames and average to reduce noise
        bg_frames = []
        for i in range(10):
            ret, frame = self.cap.read()
            if ret:
                bg_frames.append(frame)
            time.sleep(0.1)

        if bg_frames:
            self.background_frame = np.mean(bg_frames, axis=0).astype(np.uint8)
            print("Background frame acquired successfully")

            # Save background frame for debugging
            if self.debug_mode:
                cv2.imwrite("background_frame.jpg", self.background_frame)
        else:
            print("Warning: Unable to get background frame")

        # Restore original laser position
        self.move_to_position(current_x, current_y)

    def create_frame(self, x_center, y_center):
        """Create a laser frame, single point"""
        frame = (HeliosPoint * self.POINTS_PER_FRAME)()

        # All points concentrated at the same position
        for j in range(self.POINTS_PER_FRAME):
            frame[j] = HeliosPoint(x=x_center, y=y_center,
                                   r=self.COLOR_VALUE, g=self.COLOR_VALUE, b=self.COLOR_VALUE,
                                   i=self.INTENSITY)
        return frame

    def send_frame(self, frame):
        """Send frame to laser device"""
        for j in range(self.numDevices):
            statusAttempts = 0
            while (statusAttempts < 32 and self.HeliosLib.GetStatus(j) != 1):
                statusAttempts += 1
            self.HeliosLib.WriteFrame(j, self.FRAME_DURATION, 0, ctypes.pointer(frame), self.POINTS_PER_FRAME)

    def move_to_position(self, x, y):
        """Move laser to specified position"""
        # Constrain coordinate range
        x = max(0, min(int(x), self.LASER_MAX))
        y = max(0, min(int(y), self.LASER_MAX))

        # Create and send frame
        frame = self.create_frame(x, y)
        self.send_frame(frame)

        # Print position information
        print(f"Laser position: X={x}, Y={y}")
        return x, y

    def calculate_grid_position(self, x, y):
        """Calculate precise grid position in camera, using original resolution instead of 100x100 grid"""
        # Use pixel coordinates directly to ensure highest precision
        # For saving and display, we still keep the grid concept for human understanding
        grid_x = int(x)
        grid_y = int(y)
        return grid_x, grid_y

    def draw_grid(self, frame):
        """Draw grid on camera view"""
        grid_frame = frame.copy()

        # Draw horizontal lines (Y axis)
        for i in range(0, self.frame_height, self.frame_height // 10):  # Only draw 10 lines, not too dense
            y_coord = i
            cv2.line(grid_frame, (0, y_coord), (self.frame_width, y_coord), (128, 128, 128), 1)

        # Draw vertical lines (X axis)
        for i in range(0, self.frame_width, self.frame_width // 10):  # Only draw 10 lines, not too dense
            cv2.line(grid_frame, (i, 0), (i, self.frame_height), (128, 128, 128), 1)

        # Draw defined region (if corner points exist)
        if len(self.region_corners_camera) >= 2:
            for i in range(len(self.region_corners_camera)):
                pt1 = tuple(map(int, self.region_corners_camera[i]))
                pt2 = tuple(map(int, self.region_corners_camera[(i + 1) % len(self.region_corners_camera)]))
                cv2.line(grid_frame, pt1, pt2, (0, 255, 255), 2)  # Yellow line

        # Mark region corner points
        for i, point in enumerate(self.region_corners_camera):
            point_int = tuple(map(int, point))
            cv2.circle(grid_frame, point_int, 8, (0, 255, 255), -1)  # Yellow corner
            cv2.putText(grid_frame, f"C{i + 1}", (point_int[0] + 15, point_int[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw power slider
        self.draw_power_slider(grid_frame)

        return grid_frame

    def detect_laser_spot(self):
        """Enhanced laser spot detection with subpixel accuracy and quality estimation"""
        # Check if background frame is acquired
        if self.background_frame is None:
            print("Warning: No background frame")
            self.get_background_frame()
            if self.background_frame is None:
                return None

        # Get current frame (with laser)
        ret, laser_frame = self.cap.read()
        if not ret:
            print("Unable to get current frame")
            return None

        # Calculate difference
        diff = cv2.absdiff(laser_frame, self.background_frame)

        # Convert to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)

        # Try adaptive thresholding first
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Also try standard thresholding as backup
        _, binary_standard = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

        # Combine both methods
        binary = cv2.bitwise_or(binary_adaptive, binary_standard)

        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Save debug images
        if self.debug_mode:
            cv2.imshow("Difference", diff)
            cv2.imshow("Blurred", blurred)
            cv2.imshow("Binary", binary)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter contours by multiple criteria
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                # Skip tiny contours
                if area < 10:
                    continue

                # Calculate circularity
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Calculate aspect ratio using bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Calculate intensity in original difference image
                mask = np.zeros_like(gray_diff)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_intensity = cv2.mean(gray_diff, mask=mask)[0]

                # Laser spots should be relatively circular, with reasonable area and high intensity
                if (0.5 < circularity < 1.2 and
                        0.5 < aspect_ratio < 2.0 and
                        10 < area < 5000 and
                        mean_intensity > 30):
                    # Calculate a quality score
                    quality = (circularity * 0.5 +
                               (1.0 - abs(aspect_ratio - 1.0)) * 0.3 +
                               min(1.0, mean_intensity / 200) * 0.2)

                    valid_contours.append((contour, area, quality))

            if valid_contours:
                # Sort by quality score and pick the best one
                best_contour, area, quality = max(valid_contours, key=lambda x: x[2])

                # Get basic centroid using moments
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Refine to subpixel accuracy
                    # Create a small region of interest around the centroid
                    roi_size = 15
                    roi_x1 = max(0, cx - roi_size)
                    roi_y1 = max(0, cy - roi_size)
                    roi_x2 = min(gray_diff.shape[1], cx + roi_size + 1)
                    roi_y2 = min(gray_diff.shape[0], cy + roi_size + 1)

                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                        roi = gray_diff[roi_y1:roi_y2, roi_x1:roi_x2].astype(float)

                        # Only proceed if we have valid intensity data
                        if np.sum(roi) > 0:
                            # Find weighted centroid for subpixel precision
                            y_indices, x_indices = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
                            total_intensity = np.sum(roi)

                            if total_intensity > 0:
                                refined_x = np.sum(x_indices * roi) / total_intensity + roi_x1
                                refined_y = np.sum(y_indices * roi) / total_intensity + roi_y1

                                # Use refined coordinates instead of basic centroid
                                cx = refined_x
                                cy = refined_y

                    # Mark detected point on image
                    marked_frame = laser_frame.copy()
                    cv2.drawContours(marked_frame, [best_contour], 0, (0, 255, 0), 2)
                    cv2.circle(marked_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                    # Draw cross for subpixel center
                    cv2.line(marked_frame,
                             (int(cx - 10), int(cy)), (int(cx + 10), int(cy)),
                             (255, 0, 0), 1)
                    cv2.line(marked_frame,
                             (int(cx), int(cy - 10)), (int(cx), int(cy + 10)),
                             (255, 0, 0), 1)

                    # Calculate grid position
                    grid_x, grid_y = self.calculate_grid_position(cx, cy)

                    # Display mark and info
                    quality_text = f"Quality: {quality:.2f}, Circularity: {circularity:.2f}"
                    position_text = f"Point: ({cx:.2f},{cy:.2f}) Area: {area:.1f}"

                    cv2.putText(marked_frame, quality_text, (10, self.frame_height - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(marked_frame, position_text, (int(cx) + 10, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Show results
                    cv2.imshow("Detected Laser Point", marked_frame)

                    print(f"Detected laser point: Pixel({cx:.2f},{cy:.2f}), Quality: {quality:.2f}, Area: {area:.1f}")
                    return {
                        'laser_x': self.current_x,
                        'laser_y': self.current_y,
                        'camera_pixel_x': cx,
                        'camera_pixel_y': cy,
                        'camera_grid_x': grid_x,
                        'camera_grid_y': grid_y,
                        'area': area,
                        'quality': quality
                    }
                else:
                    print("Zero moment encountered")
            else:
                print("No valid contours met the filtering criteria")
        else:
            print("No contours detected")

        # If detection failed, try HSV color filtering as a last resort
        hsv = cv2.cvtColor(laser_frame, cv2.COLOR_BGR2HSV)
        # Adjust these values based on your laser color
        lower_bound = np.array([0, 0, 200])
        upper_bound = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        if self.debug_mode:
            cv2.imshow("HSV Mask", mask)

        hsv_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hsv_contours:
            largest = max(hsv_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 10:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    grid_x, grid_y = self.calculate_grid_position(cx, cy)
                    print(f"HSV fallback detection: Pixel({cx},{cy}), Area: {area:.1f}")
                    return {
                        'laser_x': self.current_x,
                        'laser_y': self.current_y,
                        'camera_pixel_x': cx,
                        'camera_pixel_y': cy,
                        'camera_grid_x': grid_x,
                        'camera_grid_y': grid_y,
                        'area': area,
                        'quality': 0.1  # Low quality since it's a fallback method
                    }

        # If all methods above fail, return None
        print("Unable to detect laser spot")
        return None

    def is_point_in_region(self, point_x, point_y):
        """Check if point is in defined region"""
        if len(self.region_corners_camera) < 3:
            return False  # Region not defined

        # Prepare polygon array, ensure correct format
        poly = np.array(self.region_corners_camera, np.int32)
        poly = poly.reshape((-1, 1, 2))  # Reshape to OpenCV required format

        # Prepare point, ensure integer tuple
        point = (int(point_x), int(point_y))

        # Check if point is in polygon
        result = cv2.pointPolygonTest(poly, point, False)
        return result >= 0  # >= 0 means point is inside or on polygon boundary

    def define_region_corner(self, index):
        """Define a corner point of the region"""
        if index < 0 or index > 3:
            print("Invalid corner index, should be 0-3")
            return False

        # Detect laser point
        point = self.detect_laser_spot()
        if not point:
            print("Unable to detect laser point, cannot set corner")
            return False

        # Store laser coordinates
        laser_x = point['laser_x']
        laser_y = point['laser_y']

        # Store camera pixel coordinates
        camera_x = point['camera_pixel_x']
        camera_y = point['camera_pixel_y']

        # Update or add corner point - ensure stored as integer tuple
        if index < len(self.region_corners_laser):
            self.region_corners_laser[index] = (int(laser_x), int(laser_y))
            self.region_corners_camera[index] = (int(camera_x), int(camera_y))
        else:
            self.region_corners_laser.append((int(laser_x), int(laser_y)))
            self.region_corners_camera.append((int(camera_x), int(camera_y)))

        print(
            f"Set corner {index + 1}: Laser coordinates({laser_x},{laser_y}), Camera coordinates({camera_x},{camera_y})")
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse click callback function"""
        # First check if mouse is interacting with power slider
        if self.handle_slider_mouse(event, x, y):
            return

        # Original mouse callback functionality
        if event == cv2.EVENT_LBUTTONDOWN:
            # Process click based on current state
            if self.calibration_state == "DEFINE_REGION":
                # Click to define region corner
                # Here we should actually use laser point position, not directly mouse click position
                # We'll ignore this click, use detected laser position
                pass
            elif self.calibration_state == "COLLECT_POINTS":
                # If in collecting calibration points state, use click position as calibration point
                grid_x, grid_y = self.calculate_grid_position(x, y)

                # Check if point is in defined region
                if not self.is_point_in_region(x, y):
                    print(f"Click position({x},{y}) not in defined region, please try again")
                    return

                # Add calibration point to list
                self.calibration_points.append({
                    'laser_x': self.current_x,
                    'laser_y': self.current_y,
                    'camera_pixel_x': x,
                    'camera_pixel_y': y,
                    'camera_grid_x': grid_x,
                    'camera_grid_y': grid_y
                })

                print(f"Added calibration point: Laser({self.current_x},{self.current_y}) -> Camera({x},{y})")

    def save_calibration_data(self, filename="calibration_data.json"):
        """Save calibration data"""
        data = {
            'region_corners_laser': self.region_corners_laser,
            'region_corners_camera': self.region_corners_camera,
            'calibration_points': self.calibration_points
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Calibration data saved to {filename}")

    def load_calibration_data(self, filename="calibration_data.json"):
        """Load calibration data"""
        if not os.path.exists(filename):
            print(f"Calibration data file does not exist: {filename}")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.region_corners_laser = data.get('region_corners_laser', [])
            self.region_corners_camera = data.get('region_corners_camera', [])
            self.calibration_points = data.get('calibration_points', [])

            print(
                f"Loaded calibration data: Region corners={len(self.region_corners_laser)}, Calibration points={len(self.calibration_points)}")

            # Update state
            if len(self.region_corners_laser) == 4:
                self.calibration_state = "COLLECT_POINTS"
                if len(self.calibration_points) > 0:
                    self.calibration_state = "COMPLETED"

            return True
        except Exception as e:
            print(f"Failed to load calibration data: {e}")
            return False

    def calculate_transformation(self):
        """Calculate coordinate transformation relationship"""
        if len(self.calibration_points) < 4:
            print("Need at least 4 calibration points")
            return None

        # Extract point pairs
        src_points = np.array([[p['camera_pixel_x'], p['camera_pixel_y']] for p in self.calibration_points],
                              dtype=np.float32)
        dst_points = np.array([[p['laser_x'], p['laser_y']] for p in self.calibration_points], dtype=np.float32)

        try:
            # Calculate transformation matrix (camera pixel coordinates -> laser coordinates)
            transformation_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

            # Save transformation matrix
            np.save("transformation_matrix.npy", transformation_matrix)
            print("Transformation matrix saved")

            # Test transformation accuracy
            self.test_transformation(transformation_matrix)

            return transformation_matrix
        except Exception as e:
            print(f"Error calculating transformation matrix: {e}")
            return None

    def test_transformation(self, matrix):
        """Test transformation matrix accuracy"""
        errors = []

        for point in self.calibration_points:
            # Get actual laser coordinates
            actual_x, actual_y = point['laser_x'], point['laser_y']

            # Get camera pixel coordinates
            camera_x, camera_y = point['camera_pixel_x'], point['camera_pixel_y']

            # Use transformation matrix to calculate predicted laser coordinates
            camera_point = np.array([camera_x, camera_y, 1], dtype=np.float32).reshape((1, 3))
            transformed = np.matmul(matrix, camera_point.T).T
            transformed = transformed / transformed[0, 2]

            predicted_x = transformed[0, 0]
            predicted_y = transformed[0, 1]

            # Calculate error
            error = np.sqrt((predicted_x - actual_x) ** 2 + (predicted_y - actual_y) ** 2)
            errors.append(error)

            print(
                f"Point ({camera_x},{camera_y}) -> Actual:({actual_x},{actual_y}) Predicted:({predicted_x:.1f},{predicted_y:.1f}) Error:{error:.1f}")

        # Calculate average error
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"Average error: {mean_error:.2f}, Maximum error: {max_error:.2f}")

    def grid_calibrate_in_region(self, step_x=None, step_y=None, clear_existing=None):
        """Perform grid calibration in defined region (complete area coverage)"""
        if len(self.region_corners_laser) < 4 or len(self.region_corners_camera) < 4:
            print("Please complete region definition (set 4 corner points) first")
            return

        # Use specified step size or default values
        if step_x is not None:
            self.grid_step_x = step_x
        if step_y is not None:
            self.grid_step_y = step_y

        # Ensure we have background frame
        if self.background_frame is None:
            self.get_background_frame()

        # Store current laser position for later restoration
        orig_x, orig_y = self.current_x, self.current_y

        # Calculate bounding box of region (laser coordinates)
        min_x = min(p[0] for p in self.region_corners_laser)
        max_x = max(p[0] for p in self.region_corners_laser)
        min_y = min(p[1] for p in self.region_corners_laser)
        max_y = max(p[1] for p in self.region_corners_laser)

        # Clear existing calibration points
        if clear_existing is None:
            # Ask user whether to clear existing calibration points
            print(f"Currently have {len(self.calibration_points)} calibration points. Clear them? (Y/N):")
            print("Press Y to clear, N to keep and add new points")

            clear_confirmed = False
            wait_start = time.time()
            while time.time() - wait_start < 5:  # Wait 5 seconds
                if keyboard.is_pressed('y'):
                    self.calibration_points = []
                    print("Cleared existing calibration points")
                    clear_confirmed = True
                    time.sleep(0.5)
                    break
                if keyboard.is_pressed('n'):
                    print("Keeping existing calibration points, will add new points")
                    clear_confirmed = True
                    time.sleep(0.5)
                    break
                time.sleep(0.1)

            if not clear_confirmed:
                print("No confirmation received, keeping existing calibration points by default")
        elif clear_existing:
            self.calibration_points = []
            print("Cleared existing calibration points")

        print(f"Starting grid calibration, X step={self.grid_step_x}, Y step={self.grid_step_y}")
        print(f"Region range: X[{min_x}-{max_x}], Y[{min_y}-{max_y}]")
        print("Press ESC to interrupt calibration")
        print(f"Grid size: Camera pixel units, resolution {self.frame_width}x{self.frame_height}")

        try:
            # Record initial point count
            initial_points = len(self.calibration_points)

            # Scan region in grid pattern
            for laser_y in range(min_y, max_y + 1, self.grid_step_y):
                # Alternate scanning direction (snake pattern) for efficiency
                x_values = range(min_x, max_x + 1, self.grid_step_x)
                if (laser_y - min_y) // self.grid_step_y % 2 == 1:  # Odd rows reverse
                    x_values = reversed(list(x_values))

                for laser_x in x_values:
                    # Move laser to grid point
                    self.current_x, self.current_y = self.move_to_position(laser_x, laser_y)
                    time.sleep(0.5)  # Wait for laser to stabilize

                    # Detect laser point
                    for attempt in range(3):  # Try detection 3 times
                        point = self.detect_laser_spot()
                        if point:
                            # Check if point is in camera defined region
                            camera_x = point['camera_pixel_x']
                            camera_y = point['camera_pixel_y']

                            if self.is_point_in_region(camera_x, camera_y):
                                # Check for duplicate points
                                is_duplicate = False
                                for existing_point in self.calibration_points:
                                    if (abs(existing_point['camera_pixel_x'] - camera_x) < 5 and
                                            abs(existing_point['camera_pixel_y'] - camera_y) < 5):
                                        is_duplicate = True
                                        break

                                if not is_duplicate:
                                    self.calibration_points.append(point)
                                    print(
                                        f"Added calibration point {len(self.calibration_points)}: Laser({laser_x},{laser_y}) -> Camera({camera_x},{camera_y})")
                                    break
                                else:
                                    print(f"Skipping duplicate point: Laser({laser_x},{laser_y})")
                                    break
                            else:
                                print(f"Point outside camera region: Laser({laser_x},{laser_y})")
                                break

                        time.sleep(0.2)  # Interval

                    # Periodic saving (save every 10 points)
                    if len(self.calibration_points) % 10 == 0 and len(self.calibration_points) > initial_points:
                        self.save_calibration_data()
                        print(f"Saved {len(self.calibration_points)} calibration points")

                    # Check for ESC key to interrupt
                    if cv2.waitKey(1) == 27:  # ESC key
                        print("User interrupted calibration")
                        break

                # Check for ESC key to interrupt (row loop)
                if cv2.waitKey(1) == 27:
                    break

            # Final save
            if len(self.calibration_points) > initial_points:
                self.save_calibration_data()
                print(
                    f"Grid calibration complete, total {len(self.calibration_points)} points (added {len(self.calibration_points) - initial_points} new)")
            else:
                print("No new points added")

            # Restore original position
            self.move_to_position(orig_x, orig_y)

            # Update state
            if len(self.calibration_points) > 0:
                self.calibration_state = "COMPLETED"

        except Exception as e:
            print(f"Grid calibration error: {e}")
            # Save collected data
            if len(self.calibration_points) > initial_points:
                self.save_calibration_data()
                print(f"Saved {len(self.calibration_points)} calibration points")
            # Restore original position
            self.move_to_position(orig_x, orig_y)

    def auto_calibrate_in_region(self, num_points=20):
        """Automatically collect calibration points in defined region"""
        if len(self.region_corners_laser) < 4 or len(self.region_corners_camera) < 4:
            print("Please complete region definition (set 4 corner points) first")
            return

        # Ensure we have background frame
        if self.background_frame is None:
            self.get_background_frame()

        # Store current laser position for later restoration
        orig_x, orig_y = self.current_x, self.current_y

        print(f"Auto-collecting {num_points} calibration points in defined region...")

        # Count of valid points collected
        valid_points = 0

        # Maximum attempts
        max_attempts = num_points * 3
        attempts = 0

        while valid_points < num_points and attempts < max_attempts:
            attempts += 1

            # Generate random point in region
            random_x, random_y = self.generate_random_point_in_region()
            if random_x is None or random_y is None:
                continue

            # Move laser to random position
            self.current_x, self.current_y = self.move_to_position(random_x, random_y)
            time.sleep(0.5)  # Wait for laser to stabilize

            # Detect laser point
            for detection_attempt in range(3):
                point = self.detect_laser_spot()
                if point:
                    # Check for duplicate points (based on pixel coordinates)
                    is_duplicate = False
                    for existing_point in self.calibration_points:
                        if (abs(existing_point['camera_pixel_x'] - point['camera_pixel_x']) < 5 and
                                abs(existing_point['camera_pixel_y'] - point['camera_pixel_y']) < 5):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        self.calibration_points.append(point)
                        valid_points += 1
                        print(f"Collected {valid_points}/{num_points} calibration points")
                        break
                    else:
                        print("Skipping duplicate point")
                        break
                else:
                    print(f"Attempt {detection_attempt + 1} failed")
                    time.sleep(0.3)

            # Check if should continue
            if cv2.waitKey(1) == 27:  # ESC key
                print("User interrupted calibration")
                break

        # Restore original position
        self.move_to_position(orig_x, orig_y)

        # Save calibration data
        if len(self.calibration_points) > 0:
            self.save_calibration_data()
            print(f"Auto calibration complete, collected {valid_points} points")

            # Calculate transformation matrix
            if valid_points >= 4:
                self.calculate_transformation()
            else:
                print("Insufficient points, cannot calculate transformation matrix")
        else:
            print("No valid points collected")

    def generate_random_point_in_region(self):
        """Generate random laser coordinate point in defined region"""
        if len(self.region_corners_laser) < 4:
            print("Region not fully defined, cannot generate random point")
            return None, None

        # Calculate bounding box of region (laser coordinates)
        min_x = min(p[0] for p in self.region_corners_laser)
        max_x = max(p[0] for p in self.region_corners_laser)
        min_y = min(p[1] for p in self.region_corners_laser)
        max_y = max(p[1] for p in self.region_corners_laser)

        # Calculate bounding box of region (camera coordinates)
        min_cam_x = min(p[0] for p in self.region_corners_camera)
        max_cam_x = max(p[0] for p in self.region_corners_camera)
        min_cam_y = min(p[1] for p in self.region_corners_camera)
        max_cam_y = max(p[1] for p in self.region_corners_camera)

        # Try up to 100 times to generate valid random point
        for _ in range(100):
            # Generate random point in bounding box (laser coordinates)
            random_x = random.randint(min_x, max_x)
            random_y = random.randint(min_y, max_y)

            # Estimate this point's position on camera (simple linear interpolation)
            # This is a rough estimate to check if point is in region
            cam_x_ratio = (random_x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
            cam_y_ratio = (random_y - min_y) / (max_y - min_y) if max_y > min_y else 0.5

            estimated_cam_x = int(min_cam_x + cam_x_ratio * (max_cam_x - min_cam_x))
            estimated_cam_y = int(min_cam_y + cam_y_ratio * (max_cam_y - min_cam_y))

            # Check if estimated camera position is in region
            if self.is_point_in_region(estimated_cam_x, estimated_cam_y):
                return random_x, random_y

        # If unable to generate valid point, return rough estimate of region center
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        return center_x, center_y

    def run_calibration(self):
        """Run calibration program"""
        print("Starting region-based laser calibration program with ESP32 control")
        print("ESP32 Laser Control:")
        print("Press L to toggle laser ON/OFF")
        print("Use power slider in top-right corner to adjust laser power")
        print("")
        print("Calibration Controls:")
        print("First define calibration region's four corners:")
        print("Press W/A/S/D to move laser")
        print("Press 1/2/3/4 to set four corner points")
        print("Press R to auto-collect calibration points in defined region")
        print("Press G to perform grid calibration in defined region (will ask whether to clear existing points)")
        print("Press X to clear all calibration points")
        print("Press +/- to increase/decrease grid step size")
        print("Press Space to manually add calibration point at current position")
        print("Press B to get new background frame")
        print("Press T to calculate transformation matrix")
        print("Press L to load previous calibration data")
        print("Press F to save calibration data")
        print("Press ESC to exit")

        # Try to load previous calibration data
        self.load_calibration_data()

        # Move to initial position
        self.current_x, self.current_y = self.move_to_position(self.current_x, self.current_y)

        running = True

        while running:
            # Display camera view
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Draw grid
            grid_frame = self.draw_grid(frame)

            # Display calibration points (points in region in red)
            for i, point in enumerate(self.calibration_points):
                # Use small dots for calibration points to avoid cluttering the display
                cv2.circle(grid_frame, (point['camera_pixel_x'], point['camera_pixel_y']), 2, (0, 0, 255), -1)
                # Only show numbers for few points to avoid excessive clutter
                if i % 100 == 0:  # Show number every 100 points
                    cv2.putText(grid_frame, f"P{i + 1}", (point['camera_pixel_x'] + 10, point['camera_pixel_y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display current state
            state_text = ""
            if self.calibration_state == "DEFINE_REGION":
                state_text = "Define region corners"
                if len(self.region_corners_laser) < 4:
                    state_text += f" (Set {len(self.region_corners_laser)}/4)"
            elif self.calibration_state == "COLLECT_POINTS":
                state_text = f"Collect calibration points (Collected {len(self.calibration_points)})"
            else:
                state_text = f"Calibration complete (Total {len(self.calibration_points)} points)"

            cv2.putText(grid_frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display current laser position
            position_text = f"Laser position: X={self.current_x}, Y={self.current_y} | Grid step: X={self.grid_step_x}, Y={self.grid_step_y}"
            cv2.putText(grid_frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display ESP32 connection status
            esp32_status = "ESP32: Connected" if self.esp32_connected else "ESP32: Not Connected"
            esp32_color = (0, 255, 0) if self.esp32_connected else (0, 0, 255)
            cv2.putText(grid_frame, esp32_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, esp32_color, 2)

            cv2.imshow("Calibration", grid_frame)
            key = cv2.waitKey(1)

            # Determine basic step size
            step = self.SINGLE_STEP * 5 if keyboard.is_pressed('shift') else self.SINGLE_STEP

            # Process keyboard input
            old_x, old_y = self.current_x, self.current_y
            moved = False

            if keyboard.is_pressed('w'):
                self.current_y += step
                self.current_y = min(self.current_y, self.LASER_MAX)
                moved = True

            if keyboard.is_pressed('s'):
                self.current_y -= step
                self.current_y = max(self.current_y, 0)
                moved = True

            if keyboard.is_pressed('a'):
                self.current_x -= step
                self.current_x = max(self.current_x, 0)
                moved = True

            if keyboard.is_pressed('d'):
                self.current_x += step
                self.current_x = min(self.current_x, self.LASER_MAX)
                moved = True

            # If position changed, update laser position
            if moved and (old_x != self.current_x or old_y != self.current_y):
                self.current_x, self.current_y = self.move_to_position(self.current_x, self.current_y)

            # Toggle laser ON/OFF with L key
            if keyboard.is_pressed('l'):
                self.toggle_laser()
                time.sleep(0.5)  # Prevent multiple toggles

            # Set corner 1
            if keyboard.is_pressed('1'):
                if self.define_region_corner(0):
                    print("Corner 1 set")
                time.sleep(0.5)

            # Set corner 2
            if keyboard.is_pressed('2'):
                if self.define_region_corner(1):
                    print("Corner 2 set")
                time.sleep(0.5)

            # Set corner 3
            if keyboard.is_pressed('3'):
                if self.define_region_corner(2):
                    print("Corner 3 set")
                time.sleep(0.5)

            # Set corner 4
            if keyboard.is_pressed('4'):
                if self.define_region_corner(3):
                    print("Corner 4 set")
                    # If all 4 corners set, enter point collection phase
                    if len(self.region_corners_laser) == 4:
                        self.calibration_state = "COLLECT_POINTS"
                time.sleep(0.5)

            # Get new background frame
            if keyboard.is_pressed('b'):
                print("Getting new background frame...")
                self.get_background_frame()
                time.sleep(0.5)

            # Manually add calibration point
            if keyboard.is_pressed('space'):
                # Only if region is defined can we add calibration points
                if self.calibration_state == "COLLECT_POINTS" or self.calibration_state == "COMPLETED":
                    point = self.detect_laser_spot()
                    if point:
                        # Check if point is in defined region
                        if self.is_point_in_region(point['camera_pixel_x'], point['camera_pixel_y']):
                            # Check for duplicate points
                            is_duplicate = False
                            for existing_point in self.calibration_points:
                                if (abs(existing_point['camera_pixel_x'] - point['camera_pixel_x']) < 3 and
                                        abs(existing_point['camera_pixel_y'] - point['camera_pixel_y']) < 3):
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                self.calibration_points.append(point)
                                print(f"Added calibration point: Total {len(self.calibration_points)}")
                            else:
                                print("Skipping duplicate point")
                        else:
                            print("Detected point not in defined region, please try again")
                else:
                    print("Please complete region definition (set 4 corner points) first")
                time.sleep(0.5)

            # Clear all calibration points
            if keyboard.is_pressed('x'):
                # Second confirmation
                print("Confirm clearing all calibration points? Press Y to confirm, other keys to cancel")
                time.sleep(0.5)
                # Wait for confirmation
                wait_start = time.time()
                while time.time() - wait_start < 3:  # Wait 3 seconds
                    if keyboard.is_pressed('y'):
                        self.calibration_points = []
                        print(f"Cleared all calibration points")
                        time.sleep(0.5)
                        break
                    if keyboard.is_pressed('n') or keyboard.is_pressed('esc'):
                        print("Clearing canceled")
                        time.sleep(0.5)
                        break
                    time.sleep(0.1)

            # Auto collect calibration points
            if keyboard.is_pressed('r'):
                if self.calibration_state == "COLLECT_POINTS" or self.calibration_state == "COMPLETED":
                    self.auto_calibrate_in_region(20)  # Collect 20 points
                    self.calibration_state = "COMPLETED"
                else:
                    print("Please complete region definition (set 4 corner points) first")
                time.sleep(0.5)

            # Grid calibration (new feature)
            if keyboard.is_pressed('g'):
                if self.calibration_state == "COLLECT_POINTS" or self.calibration_state == "COMPLETED":
                    self.grid_calibrate_in_region()
                    self.calibration_state = "COMPLETED"
                else:
                    print("Please complete region definition (set 4 corner points) first")
                time.sleep(0.5)

            # Adjust grid step size
            if keyboard.is_pressed('=') or keyboard.is_pressed('+'):
                self.grid_step_x = min(self.grid_step_x + 5, 200)
                self.grid_step_y = min(self.grid_step_y + 5, 200)
                print(f"Grid step increased to: X={self.grid_step_x}, Y={self.grid_step_y}")
                time.sleep(0.2)

            if keyboard.is_pressed('-'):
                self.grid_step_x = max(self.grid_step_x - 5, 5)
                self.grid_step_y = max(self.grid_step_y - 5, 5)
                print(f"Grid step decreased to: X={self.grid_step_x}, Y={self.grid_step_y}")
                time.sleep(0.2)

            # Calculate transformation matrix
            if keyboard.is_pressed('t'):
                self.calculate_transformation()
                time.sleep(0.5)

            # Save calibration data
            if keyboard.is_pressed('f'):
                self.save_calibration_data()
                time.sleep(0.5)

            # Load calibration data (changed from 'l' to 'k' to avoid conflict with laser toggle)
            if keyboard.is_pressed('k'):
                self.load_calibration_data()
                time.sleep(0.5)

            # Exit
            if keyboard.is_pressed('esc'):
                running = False

            # Small delay
            time.sleep(0.001)

        # Save data
        self.save_calibration_data()

        # Turn off laser before exit
        if self.esp32_connected:
            self.send_laser_command("OFF", False)
            print("Laser turned OFF before exit")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.HeliosLib.CloseDevices()

        if self.esp32 and self.esp32.is_open:
            self.esp32.close()
            print("ESP32 connection closed")

        print("Calibration program ended")


if __name__ == "__main__":
    try:
        calibrator = RegionBasedCalibration()
        calibrator.run_calibration()
    except Exception as e:
        print(f"Program exception: {e}")
        import traceback

        traceback.print_exc()
        try:
            if 'calibrator' in locals() and hasattr(calibrator, 'cap'):
                calibrator.cap.release()
            if 'calibrator' in locals() and hasattr(calibrator, 'HeliosLib'):
                calibrator.HeliosLib.CloseDevices()
            if 'calibrator' in locals() and hasattr(calibrator,
                                                    'esp32') and calibrator.esp32 and calibrator.esp32.is_open:
                calibrator.esp32.write(b'OFF\n')  # Turn off laser
                calibrator.esp32.close()
            cv2.destroyAllWindows()
        except:
            pass
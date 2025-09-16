from flask import Flask, render_template, jsonify, request, Response
import paho.mqtt.client as mqtt
import json
import threading
import queue
import socket
import math
import time
import subprocess
import cv2
import numpy as np
import base64
from datetime import datetime
import os
import uuid
import ctypes

# Enhanced Weed Targeting System imports
from ultralytics import YOLO
from scipy.spatial import KDTree
from collections import deque

app = Flask(__name__)

# ------------------ Global Storage for Data ------------------ #
current_data = {
    'imu': None,  # raw IMU
    'gps': None,  # raw GPS
    'fused': None,  # fused/filtered navigation data
    'camera': None,  # camera tracking data
    'weed_detections': [],  # weed detection data
    'laser': {  # laser status and parameters
        'status': 'OFF',
        'power': 2,  # Changed from 5 to 2 for safety
        'aim_power': 2,
        'duration': 5,
        'safety_delay': 2,
        'mode': 'MANUAL'  # 'MANUAL' or 'AUTO'
    }
}

# Latest camera frame
latest_frame = None
mqtt_received_data = []  # Stores raw MQTT data
data_queue = queue.Queue()


# Simple PID parameter storage
class PIDParams:
    def __init__(self):
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0
        self.speed = 150  # default speed


pid_params = PIDParams()


# ------------------ Enhanced Weed Targeting System Classes ------------------ #

# Define Helios point structure
class HeliosPoint(ctypes.Structure):
    _fields_ = [('x', ctypes.c_uint16),
                ('y', ctypes.c_uint16),
                ('r', ctypes.c_uint8),
                ('g', ctypes.c_uint8),
                ('b', ctypes.c_uint8),
                ('i', ctypes.c_uint8)]


class AdvancedNoiseFilter:
    """Advanced noise filter for shaky hand movements and rough terrain like grass"""

    def __init__(self, filter_strength=0.3, smoothing_window=5):
        self.filter_strength = filter_strength  # 0.0 = no filtering, 1.0 = heavy filtering
        self.smoothing_window = smoothing_window  # Moving average window size
        self.position_history = {}  # {weed_id: [filtered_positions]}
        self.raw_history = {}  # {weed_id: [raw_positions]} for smoothing
        self.velocity_history = {}  # {weed_id: [velocities]} for consistency
        self.max_history = 20

        # Advanced filtering parameters
        self.movement_threshold = 3.0  # Minimum movement to consider as real motion
        self.velocity_smoothing = 0.7  # Velocity smoothing factor
        self.direction_consistency_weight = 0.8  # Weight for direction consistency
        self.outlier_threshold = 50.0  # Outlier detection threshold (pixels)

    def filter_position(self, weed_id, raw_x, raw_y):
        """Apply advanced noise filtering with multiple stages"""
        if weed_id not in self.position_history:
            self._initialize_weed_tracking(weed_id, raw_x, raw_y)
            return np.array([raw_x, raw_y])

        raw_pos = np.array([raw_x, raw_y])

        # Stage 1: Outlier detection and rejection
        filtered_pos = self._detect_and_handle_outliers(weed_id, raw_pos)

        # Stage 2: Moving average smoothing
        smoothed_pos = self._apply_moving_average(weed_id, filtered_pos)

        # Stage 3: Velocity-based filtering
        final_pos = self._apply_velocity_filtering(weed_id, smoothed_pos)

        # Store results
        self.position_history[weed_id].append(final_pos)
        self.raw_history[weed_id].append(raw_pos)

        return final_pos

    def _initialize_weed_tracking(self, weed_id, raw_x, raw_y):
        """Initialize tracking for a new weed"""
        initial_pos = np.array([raw_x, raw_y])
        self.position_history[weed_id] = deque([initial_pos], maxlen=self.max_history)
        self.raw_history[weed_id] = deque([initial_pos], maxlen=self.max_history)
        self.velocity_history[weed_id] = deque(maxlen=self.max_history)

    def _detect_and_handle_outliers(self, weed_id, raw_pos):
        """Detect and handle outlier positions (sudden jumps due to hand shake)"""
        if len(self.position_history[weed_id]) < 2:
            return raw_pos

        # Calculate distance from last filtered position
        last_filtered = self.position_history[weed_id][-1]
        distance = np.linalg.norm(raw_pos - last_filtered)

        # If movement is too large, it's likely an outlier
        if distance > self.outlier_threshold:
            # Use prediction based on recent movement
            if len(self.velocity_history[weed_id]) > 0:
                recent_velocity = np.mean(list(self.velocity_history[weed_id])[-3:], axis=0)
                predicted_pos = last_filtered + recent_velocity * 0.1  # Assume 0.1s time step
                return predicted_pos
            else:
                return last_filtered  # No movement history, stay at last position

        return raw_pos

    def _apply_moving_average(self, weed_id, position):
        """Apply moving average smoothing to reduce jitter"""
        window_size = min(self.smoothing_window, len(self.raw_history[weed_id]))
        if window_size <= 1:
            return position

        # Get recent positions for averaging
        recent_positions = list(self.raw_history[weed_id])[-window_size:]
        recent_positions.append(position)

        # Weighted moving average (more weight to recent positions)
        weights = np.exp(np.linspace(-1, 0, len(recent_positions)))
        weights /= np.sum(weights)

        smoothed = np.average(recent_positions, axis=0, weights=weights)
        return smoothed

    def _apply_velocity_filtering(self, weed_id, position):
        """Apply velocity-based filtering for consistent movement"""
        if len(self.position_history[weed_id]) < 2:
            return position

        last_pos = self.position_history[weed_id][-1]
        current_velocity = position - last_pos

        # Calculate movement magnitude
        movement_magnitude = np.linalg.norm(current_velocity)

        # If movement is too small, apply heavy filtering (likely noise)
        if movement_magnitude < self.movement_threshold:
            # Heavy filtering for small movements
            filtered_pos = (1 - self.filter_strength * 1.5) * position + self.filter_strength * 1.5 * last_pos
        else:
            # Check velocity consistency for larger movements
            if len(self.velocity_history[weed_id]) > 0:
                # Get recent velocity trend
                recent_velocities = list(self.velocity_history[weed_id])[-3:]
                if len(recent_velocities) > 0:
                    avg_velocity = np.mean(recent_velocities, axis=0)
                    velocity_consistency = self._calculate_velocity_consistency(current_velocity, avg_velocity)

                    # Adjust filtering strength based on consistency
                    dynamic_filter_strength = self.filter_strength * (
                            1 - velocity_consistency * self.direction_consistency_weight)
                    filtered_pos = (1 - dynamic_filter_strength) * position + dynamic_filter_strength * last_pos
                else:
                    filtered_pos = (1 - self.filter_strength) * position + self.filter_strength * last_pos
            else:
                filtered_pos = (1 - self.filter_strength) * position + self.filter_strength * last_pos

        # Store velocity for future reference
        if movement_magnitude > self.movement_threshold:
            smoothed_velocity = current_velocity
            if len(self.velocity_history[weed_id]) > 0:
                last_velocity = self.velocity_history[weed_id][-1]
                smoothed_velocity = (
                                            1 - self.velocity_smoothing) * current_velocity + self.velocity_smoothing * last_velocity

            self.velocity_history[weed_id].append(smoothed_velocity)

        return filtered_pos

    def _calculate_velocity_consistency(self, current_velocity, average_velocity):
        """Calculate how consistent current velocity is with recent trend"""
        if np.linalg.norm(average_velocity) < 0.1:
            return 0.0

        # Normalize velocities
        current_norm = np.linalg.norm(current_velocity)
        avg_norm = np.linalg.norm(average_velocity)

        if current_norm < 0.1 or avg_norm < 0.1:
            return 0.0

        current_dir = current_velocity / current_norm
        avg_dir = average_velocity / avg_norm

        # Calculate dot product (direction similarity)
        dot_product = np.clip(np.dot(current_dir, avg_dir), -1, 1)
        angle_diff = np.arccos(dot_product)

        # Convert to consistency score (0 = inconsistent, 1 = consistent)
        consistency = 1.0 - (angle_diff / np.pi)

        return max(0.0, consistency)

    def set_filter_strength(self, strength):
        """Set filter strength (0.0 to 1.0)"""
        self.filter_strength = max(0.0, min(1.0, strength))

    def set_smoothing_window(self, window_size):
        """Set moving average window size"""
        self.smoothing_window = max(1, min(10, window_size))

    def set_movement_threshold(self, threshold):
        """Set minimum movement threshold"""
        self.movement_threshold = max(1.0, threshold)

    def set_outlier_threshold(self, threshold):
        """Set outlier detection threshold"""
        self.outlier_threshold = max(10.0, threshold)

    def get_filter_stats(self, weed_id):
        """Get filtering statistics for debugging"""
        if weed_id not in self.position_history:
            return None

        return {
            'positions_tracked': len(self.position_history[weed_id]),
            'velocities_tracked': len(self.velocity_history.get(weed_id, [])),
            'filter_strength': self.filter_strength,
            'smoothing_window': self.smoothing_window
        }


class LaserShapeGenerator:
    """Generate laser shapes for weed elimination"""

    def __init__(self, points_per_frame=1000):
        self.points_per_frame = points_per_frame
        self.current_shape = "zigzag"
        self.shape_size = 100
        self.shape_density = 0.5

    def generate_zigzag_pattern(self, center_x, center_y, size, density):
        """Generate zigzag pattern around center point"""
        points = []
        half_size = size // 2
        num_lines = max(3, int(10 * density))

        for line in range(num_lines):
            y_offset = -half_size + (line * size) // (num_lines - 1)
            y_pos = center_y + y_offset
            points_per_line = self.points_per_frame // num_lines

            for point in range(points_per_line):
                progress = point / points_per_line
                if line % 2 == 0:
                    x_pos = center_x - half_size + int(progress * size)
                else:
                    x_pos = center_x + half_size - int(progress * size)
                points.append((x_pos, y_pos))

        return points

    def generate_shape_points(self, center_x, center_y):
        """Generate points for current shape"""
        if self.current_shape == "zigzag":
            return self.generate_zigzag_pattern(center_x, center_y, self.shape_size, self.shape_density)
        return [(center_x, center_y)] * self.points_per_frame


class WeedTrajectoryPredictor:
    """Enhanced trajectory prediction system with YOLO processing delay compensation"""

    def __init__(self, max_history_length=20):
        self.max_history_length = max_history_length
        self.weed_trajectories = {}
        self.min_movement_threshold = 2.0

        # YOLO processing delay compensation
        self.yolo_processing_delay = 1.0  # Default YOLO processing delay in seconds
        self.max_yolo_delay = 3.0  # Maximum expected YOLO delay
        self.min_yolo_delay = 0.2  # Minimum YOLO delay

    def set_yolo_delay(self, delay_seconds):
        """Set YOLO processing delay for prediction compensation"""
        self.yolo_processing_delay = max(self.min_yolo_delay, min(delay_seconds, self.max_yolo_delay))

    def update_weed_position(self, weed_id, pixel_x, pixel_y, timestamp):
        """Update weed position and calculate trajectory with delay compensation"""
        # Compensate for YOLO processing delay
        compensated_timestamp = timestamp + self.yolo_processing_delay

        if weed_id not in self.weed_trajectories:
            self.weed_trajectories[weed_id] = {
                'positions': deque(maxlen=self.max_history_length),
                'timestamps': deque(maxlen=self.max_history_length),
                'velocities': deque(maxlen=self.max_history_length),
                'last_update': compensated_timestamp,
                'trajectory_confidence': 0.0,
                'movement_detected': False,
                'average_velocity': np.array([0.0, 0.0]),
                'velocity_consistency': 0.0,
                'observation_duration': 0.0,
                'processing_delay': self.yolo_processing_delay
            }

        trajectory = self.weed_trajectories[weed_id]
        current_pos = np.array([pixel_x, pixel_y])

        trajectory['positions'].append(current_pos)
        trajectory['timestamps'].append(compensated_timestamp)
        trajectory['last_update'] = compensated_timestamp
        trajectory['processing_delay'] = self.yolo_processing_delay

        # Calculate observation duration
        if len(trajectory['timestamps']) > 1:
            trajectory['observation_duration'] = compensated_timestamp - trajectory['timestamps'][0]

        if len(trajectory['positions']) >= 2:
            self._calculate_motion_parameters(trajectory)

    def _calculate_motion_parameters(self, trajectory):
        """Calculate velocity and detect meaningful movement"""
        positions = list(trajectory['positions'])
        timestamps = list(trajectory['timestamps'])

        if len(positions) < 2:
            return

        pos_current = positions[-1]
        pos_previous = positions[-2]
        time_current = timestamps[-1]
        time_previous = timestamps[-2]

        dt = time_current - time_previous
        if dt > 0:
            displacement = pos_current - pos_previous
            velocity = displacement / dt
            movement_distance = np.linalg.norm(displacement)

            if movement_distance > self.min_movement_threshold:
                trajectory['velocities'].append(velocity)
                trajectory['movement_detected'] = True
                self._update_velocity_statistics(trajectory)
            else:
                trajectory['velocities'].append(np.array([0.0, 0.0]))

    def _update_velocity_statistics(self, trajectory):
        """Update velocity statistics for better prediction"""
        velocities = list(trajectory['velocities'])
        if len(velocities) < 2:
            return

        non_zero_velocities = [v for v in velocities if np.linalg.norm(v) > 0.5]

        if len(non_zero_velocities) >= 2:
            trajectory['average_velocity'] = np.mean(non_zero_velocities, axis=0)

            if len(non_zero_velocities) >= 3:
                directions = []
                for vel in non_zero_velocities:
                    vel_norm = np.linalg.norm(vel)
                    if vel_norm > 0:
                        directions.append(vel / vel_norm)

                if len(directions) >= 2:
                    angle_diffs = []
                    for i in range(1, len(directions)):
                        dot_product = np.clip(np.dot(directions[i - 1], directions[i]), -1, 1)
                        angle_diff = np.arccos(dot_product)
                        angle_diffs.append(angle_diff)

                    if angle_diffs:
                        avg_angle_diff = np.mean(angle_diffs)
                        trajectory['velocity_consistency'] = max(0.0, 1.0 - (avg_angle_diff / np.pi))

        trajectory['trajectory_confidence'] = self._calculate_trajectory_confidence(trajectory)

    def _calculate_trajectory_confidence(self, trajectory):
        """Calculate confidence level of trajectory prediction with improved algorithm"""
        if len(trajectory['positions']) < 2:
            return 0.1

        positions = list(trajectory['positions'])
        timestamps = list(trajectory['timestamps'])

        # Fast trajectory method: use first and last position for quick trajectory
        if len(positions) >= 2:
            observation_time = trajectory.get('observation_duration', 0)

            # Quick confidence boost for consistent movement
            if observation_time >= 0.8:  # After 0.8 seconds
                start_pos = positions[0]
                end_pos = positions[-1]
                total_displacement = np.linalg.norm(end_pos - start_pos)

                # Check if movement is significant
                if total_displacement > 10.0:  # 10 pixels minimum movement
                    # Calculate average velocity from start to end
                    avg_velocity = (end_pos - start_pos) / observation_time

                    # Check consistency with intermediate positions
                    consistency_score = self._check_trajectory_consistency(positions, timestamps, start_pos,
                                                                           avg_velocity)

                    # Fast confidence calculation
                    time_factor = min(1.0, observation_time / 1.0)
                    movement_factor = min(1.0, total_displacement / 50.0)
                    consistency_factor = consistency_score

                    # Delay compensation factor
                    delay_factor = max(0.7, 1.0 - (trajectory.get('processing_delay', 1.0) / 5.0))

                    fast_confidence = time_factor * movement_factor * consistency_factor * delay_factor

                    if fast_confidence > 0.4:  # Good enough for execution
                        trajectory['fast_trajectory_ready'] = True
                        trajectory['start_position'] = start_pos
                        trajectory['end_position'] = end_pos
                        trajectory['trajectory_velocity'] = avg_velocity
                        return min(0.9, fast_confidence)

        # Fallback to original method
        time_confidence = min(1.0, observation_time / 1.0)
        position_confidence = min(1.0, len(trajectory['positions']) / 8.0)
        movement_confidence = 1.0 if trajectory.get('movement_detected', False) else 0.1
        consistency_confidence = trajectory.get('velocity_consistency', 0.0)
        delay_factor = max(0.7, 1.0 - (trajectory.get('processing_delay', 1.0) / 5.0))

        overall_confidence = time_confidence * position_confidence * movement_confidence * (
                0.5 + 0.5 * consistency_confidence) * delay_factor
        return max(0.1, min(1.0, overall_confidence))

    def _check_trajectory_consistency(self, positions, timestamps, start_pos, avg_velocity):
        """Check if intermediate positions are consistent with start-end trajectory"""
        if len(positions) < 3:
            return 1.0

        consistency_scores = []
        start_time = timestamps[0]

        # Check each intermediate position against predicted trajectory
        for i in range(1, len(positions) - 1):
            actual_pos = positions[i]
            time_elapsed = timestamps[i] - start_time
            predicted_pos = start_pos + avg_velocity * time_elapsed

            # Calculate deviation
            deviation = np.linalg.norm(actual_pos - predicted_pos)

            # Convert deviation to consistency score (smaller deviation = higher consistency)
            max_allowed_deviation = 30.0  # pixels
            consistency = max(0.0, 1.0 - (deviation / max_allowed_deviation))
            consistency_scores.append(consistency)

        # Return average consistency
        return np.mean(consistency_scores) if consistency_scores else 1.0

    def predict_complete_trajectory(self, weed_id, prediction_duration_seconds, speed_scaling_factor=1.0,
                                    time_step=0.1):
        """Predict complete future trajectory path with fast trajectory method"""
        if weed_id not in self.weed_trajectories:
            return None

        trajectory = self.weed_trajectories[weed_id]
        positions = list(trajectory['positions'])

        if len(positions) < 2:
            return None

        current_position = positions[-1]

        if not trajectory.get('movement_detected', False):
            # Static weed, return single point trajectory
            return {
                'trajectory_points': [current_position],
                'timestamps': [0.0],
                'confidence': 0.1,
                'is_moving': False,
                'delay_compensated': True
            }

        # Use fast trajectory method if available (better for quick response)
        if trajectory.get('fast_trajectory_ready', False):
            trajectory_velocity = trajectory.get('trajectory_velocity', np.array([0.0, 0.0]))
        else:
            # Fallback to average velocity method
            trajectory_velocity = trajectory.get('average_velocity', np.array([0.0, 0.0]))

        # Apply speed scaling factor to match real movement speed
        trajectory_velocity = trajectory_velocity * speed_scaling_factor

        if np.linalg.norm(trajectory_velocity) < 1.0:
            return None

        # Generate trajectory points with delay compensation
        trajectory_points = []
        timestamps = []

        # Start prediction from current position plus delay compensation
        delay_compensation = trajectory.get('processing_delay', self.yolo_processing_delay)
        start_position = current_position + trajectory_velocity * delay_compensation

        num_points = int(prediction_duration_seconds / time_step)

        for i in range(num_points + 1):
            time_offset = i * time_step
            predicted_position = start_position + trajectory_velocity * time_offset

            trajectory_points.append(predicted_position)
            timestamps.append(time_offset)

        confidence = trajectory['trajectory_confidence']

        return {
            'trajectory_points': trajectory_points,
            'timestamps': timestamps,
            'confidence': confidence,
            'is_moving': True,
            'velocity': trajectory_velocity,
            'speed': np.linalg.norm(trajectory_velocity),
            'delay_compensated': True,
            'compensation_applied': delay_compensation,
            'method': 'fast' if trajectory.get('fast_trajectory_ready', False) else 'average'
        }

    def get_movement_info(self, weed_id):
        """Get detailed movement information for visualization"""
        if weed_id not in self.weed_trajectories:
            return None

        trajectory = self.weed_trajectories[weed_id]
        avg_vel = trajectory.get('average_velocity', np.array([0.0, 0.0]))
        speed = np.linalg.norm(avg_vel)

        if speed > 0:
            direction = avg_vel / speed
        else:
            direction = np.array([0.0, 0.0])

        return {
            'has_movement': trajectory.get('movement_detected', False),
            'speed': speed,
            'direction': direction,
            'consistency': trajectory.get('velocity_consistency', 0.0),
            'confidence': trajectory.get('trajectory_confidence', 0.0),
            'observation_duration': trajectory.get('observation_duration', 0.0),
            'processing_delay': trajectory.get('processing_delay', 0.0)
        }

    def cleanup_old_trajectories(self, current_time, max_age_seconds=15.0):
        """Remove old trajectory data"""
        expired_ids = []
        for weed_id, trajectory in self.weed_trajectories.items():
            if current_time - trajectory['last_update'] > max_age_seconds:
                expired_ids.append(weed_id)

        for weed_id in expired_ids:
            del self.weed_trajectories[weed_id]


class AutonomousTrajectoryFollower:
    """Autonomous trajectory follower - laser follows predicted path"""

    def __init__(self, parent_system):
        self.parent = parent_system
        self.active_execution = None
        self.execution_thread = None
        self.execution_lock = threading.Lock()

    def start_trajectory_execution(self, weed_id, trajectory_data):
        """Start trajectory following execution"""
        with self.execution_lock:
            # Stop previous execution
            if self.active_execution:
                self.stop_current_execution()

            if not trajectory_data or not trajectory_data['trajectory_points']:
                print(f"Invalid trajectory data for Weed #{weed_id}")
                return False

            self.active_execution = {
                'weed_id': weed_id,
                'trajectory_data': trajectory_data,
                'start_time': time.time(),
                'current_point_index': 0,
                'is_running': True
            }

            # Start execution thread
            self.execution_thread = threading.Thread(
                target=self._trajectory_execution_thread,
                daemon=True
            )
            self.execution_thread.start()

            delay_info = ""
            if trajectory_data.get('delay_compensated', False):
                compensation = trajectory_data.get('compensation_applied', 0)
                delay_info = f" (Delay compensated: +{compensation:.2f}s)"

            print(f"Started autonomous trajectory execution for Weed #{weed_id}")

            return True

    def stop_current_execution(self):
        """Stop current trajectory execution"""
        if self.active_execution:
            self.active_execution['is_running'] = False

        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=1.0)

    def _trajectory_execution_thread(self):
        """Trajectory execution thread - core autonomous movement logic"""
        execution = self.active_execution
        if not execution:
            return

        weed_id = execution['weed_id']
        trajectory_data = execution['trajectory_data']
        trajectory_points = trajectory_data['trajectory_points']
        timestamps = trajectory_data['timestamps']

        start_time = time.time()

        try:
            while execution['is_running'] and self.parent.running:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Find current target trajectory point
                target_point_index = 0
                for i, timestamp in enumerate(timestamps):
                    if elapsed_time >= timestamp:
                        target_point_index = i
                    else:
                        break

                # Check if trajectory execution is complete
                if target_point_index >= len(trajectory_points) - 1:
                    break

                # Get target position
                target_position = trajectory_points[target_point_index]

                # Apply noise filtering
                filtered_pos = self.parent.noise_filter.filter_position(
                    f"exec_{weed_id}", target_position[0], target_position[1]
                )

                # Transform to laser coordinates
                laser_x, laser_y = self.parent.transform_coordinates(
                    filtered_pos[0], filtered_pos[1]
                )

                # Send laser command
                if self.parent.pattern_enabled:
                    frame_buffer = self.parent.create_pattern_frame(laser_x, laser_y)
                else:
                    frame_buffer = self.parent.create_frame_data(laser_x, laser_y)

                self.parent.send_frame_to_laser(frame_buffer)

                # Update current laser position
                self.parent.current_x = laser_x
                self.parent.current_y = laser_y

                # Control update frequency
                time.sleep(1.0 / self.parent.pattern_update_rate)

        except Exception as e:
            print(f"Error in trajectory execution thread: {e}")
        finally:
            with self.execution_lock:
                if self.active_execution and self.active_execution['weed_id'] == weed_id:
                    self.active_execution = None

    def get_execution_status(self):
        """Get current execution status"""
        if not self.active_execution:
            return None

        execution = self.active_execution
        current_time = time.time()
        elapsed_time = current_time - execution['start_time']
        trajectory_data = execution['trajectory_data']
        total_duration = trajectory_data['timestamps'][-1]

        return {
            'weed_id': execution['weed_id'],
            'elapsed_time': elapsed_time,
            'total_duration': total_duration,
            'progress': min(1.0, elapsed_time / total_duration),
            'is_running': execution['is_running']
        }


class EnhancedWeedTargeting:
    def __init__(self, model_path='weed3.pt', calibration_file='calibration_data.json'):
        print("Initializing Enhanced Predictive Weed Targeting System...")

        # --- YOLO Model ---
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded YOLO model from: {model_path}")
            else:
                print(f"Warning: YOLO model file not found: {model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading YOLO model '{model_path}': {e}")
            self.model = None

        # --- Load Calibration Data ---
        self.calibration_points = []
        self.region_corners_laser = []
        self.region_corners_camera = []
        self.kdtree = None
        self.valid_calibration_indices = []
        self.load_calibration_data(calibration_file)
        self.prepare_kdtree()

        # --- Laser Control Settings ---
        self.LASER_MAX = 0xFFF
        self.POINTS_PER_FRAME = 1000
        self.FRAME_DURATION = 30000
        self.COLOR_VALUE = 255
        self.INTENSITY = 130

        self.current_x = self.LASER_MAX // 2
        self.current_y = self.LASER_MAX // 2

        # --- Helios DAC Setup ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_dir, "HeliosLaserDAC.dll")
            if not os.path.exists(dll_path):
                dll_path = "./HeliosLaserDAC.dll"

            if os.path.exists(dll_path):
                self.HeliosLib = ctypes.cdll.LoadLibrary(dll_path)
                self.numDevices = self.HeliosLib.OpenDevices()
                print(f"Found {self.numDevices} Helios DACs")
            else:
                print("Warning: HeliosLaserDAC.dll not found, running in simulation mode")
                self.HeliosLib = None
                self.numDevices = 0
        except OSError as e:
            print(f"Warning: Error loading HeliosLaserDAC.dll: {e}, running in simulation mode")
            self.HeliosLib = None
            self.numDevices = 0

        # --- Camera Setup ---
        self.cap = None
        self.frame_width = 1920
        self.frame_height = 1080

        # --- Enhanced Prediction Systems ---
        self.trajectory_predictor = WeedTrajectoryPredictor()
        self.noise_filter = AdvancedNoiseFilter(filter_strength=0.3, smoothing_window=5)
        self.laser_shape_generator = LaserShapeGenerator()
        self.autonomous_follower = AutonomousTrajectoryFollower(self)

        # --- Enhanced Control Parameters ---
        self.observation_time = 1.0  # Observation time: 1 second as requested
        self.prediction_duration = 8.0  # Prediction duration
        self.prediction_delay = 1.5  # Additional prediction delay (adjustable)
        self.execution_mode = "AUTONOMOUS"

        # Parameter ranges
        self.min_observation_time = 0.3
        self.max_observation_time = 3.0
        self.min_prediction_duration = 2.0
        self.max_prediction_duration = 20.0
        self.min_prediction_delay = 0.0
        self.max_prediction_delay = 5.0

        # YOLO processing delay compensation
        self.yolo_processing_delay = 1.0  # Default 1 second YOLO delay
        self.min_yolo_delay = 0.5
        self.max_yolo_delay = 2.0

        # Speed scaling for prediction accuracy
        self.speed_scaling_factor = 0.85  # Scale down predicted speed (85% of calculated speed)
        self.min_speed_scaling = 0.3
        self.max_speed_scaling = 1.5

        # Noise filtering parameters (enhanced controls)
        self.noise_filter_strength = 0.3
        self.noise_smoothing_window = 5
        self.noise_movement_threshold = 3.0
        self.noise_outlier_threshold = 50.0

        self.min_confidence_for_execution = 0.25  # Lowered for better responsiveness

        # --- Pattern Parameters ---
        self.pattern_enabled = True
        self.pattern_type = "zigzag"
        self.pattern_size = 80
        self.pattern_density = 0.7
        self.pattern_update_rate = 40

        # --- System State ---
        self.targeting_enabled = False
        self.running = False
        self.debug_mode = True

        # Weed tracking
        self.weed_counter = 0
        self.detected_weeds = []
        self.current_target = None
        self.target_start_time = None
        self.target_phase = "OBSERVATION"
        self.min_confidence = 0.4
        self.targeting_lock = threading.Lock()

        # Coordinate transformation
        self.transform_method = 'weighted'
        self.weighted_k = 5

        # Display options
        self.show_region = True
        self.show_trajectories = True
        self.show_predictions = True
        self.show_movement_vectors = True
        self.show_noise_stats = True

        # Threads
        self.detection_thread = None
        self.targeting_thread = None

        print("Enhanced Predictive Weed Targeting System initialized successfully")

    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            time.sleep(1.0)

            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def stop_camera(self):
        """Stop camera capture"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def load_calibration_data(self, filename):
        """Load calibration data from JSON file"""
        if not os.path.exists(filename):
            print(f"Calibration file not found: {filename}, using default transformation")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.calibration_points = data.get('calibration_points', [])
            self.region_corners_laser = data.get('region_corners_laser', [])
            self.region_corners_camera = data.get('region_corners_camera', [])

            print(f"Loaded {len(self.calibration_points)} calibration points")
            return True
        except Exception as e:
            print(f"Failed to load calibration data: {e}")
            return False

    def prepare_kdtree(self):
        """Prepare KD tree for coordinate transformation"""
        if len(self.calibration_points) < 1:
            print("Warning: No calibration points available")
            return False

        valid_points = []
        valid_indices = []

        for i, point in enumerate(self.calibration_points):
            if ('camera_pixel_x' in point and 'camera_pixel_y' in point and
                    isinstance(point['camera_pixel_x'], (int, float)) and
                    isinstance(point['camera_pixel_y'], (int, float))):
                valid_points.append([point['camera_pixel_x'], point['camera_pixel_y']])
                valid_indices.append(i)

        if not valid_points:
            print("Warning: No valid calibration points found")
            return False

        try:
            camera_points = np.array(valid_points, dtype=np.float32)
            self.kdtree = KDTree(camera_points)
            self.valid_calibration_indices = valid_indices
            print(f"Built KD tree with {len(valid_points)} valid points")
            return True
        except Exception as e:
            print(f"Error building KD tree: {e}")
            return False

    def transform_coordinates(self, camera_x, camera_y):
        """Transform camera coordinates to laser coordinates"""
        if self.kdtree is None or not self.valid_calibration_indices:
            laser_x = int((camera_x / self.frame_width) * self.LASER_MAX)
            laser_y = int((camera_y / self.frame_height) * self.LASER_MAX)
            return max(0, min(laser_x, self.LASER_MAX)), max(0, min(laser_y, self.LASER_MAX))

        try:
            k = min(self.weighted_k, len(self.valid_calibration_indices))
            if k < 1:
                return self.LASER_MAX // 2, self.LASER_MAX // 2

            dists, kdtree_idxs = self.kdtree.query([camera_x, camera_y], k=k)

            if k == 1:
                dists, kdtree_idxs = [dists], [kdtree_idxs]

            original_idxs = [self.valid_calibration_indices[i] for i in kdtree_idxs]
            weights = 1.0 / (np.maximum(dists, 1e-9) ** 2)
            weights /= np.sum(weights)

            laser_x, laser_y = 0.0, 0.0
            for i, orig_idx in enumerate(original_idxs):
                point = self.calibration_points[orig_idx]
                laser_x += point['laser_x'] * weights[i]
                laser_y += point['laser_y'] * weights[i]

            return int(round(laser_x)), int(round(laser_y))

        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return self.LASER_MAX // 2, self.LASER_MAX // 2

    def create_frame_data(self, x_dac, y_dac):
        """Create HeliosPoint array data"""
        frame_buffer = (HeliosPoint * self.POINTS_PER_FRAME)()
        x_int, y_int = int(x_dac), int(y_dac)

        point = HeliosPoint(x=x_int, y=y_int,
                            r=self.COLOR_VALUE, g=self.COLOR_VALUE, b=self.COLOR_VALUE,
                            i=self.INTENSITY)

        for j in range(self.POINTS_PER_FRAME):
            frame_buffer[j] = point

        return frame_buffer

    def create_pattern_frame(self, center_x, center_y):
        """Create a frame with laser pattern around center position"""
        if not self.pattern_enabled:
            return self.create_frame_data(center_x, center_y)

        pattern_points = self.laser_shape_generator.generate_shape_points(center_x, center_y)
        frame_buffer = (HeliosPoint * self.POINTS_PER_FRAME)()
        points_per_pattern_point = max(1, self.POINTS_PER_FRAME // len(pattern_points))
        point_index = 0

        for pattern_x, pattern_y in pattern_points:
            x_clamped = max(0, min(int(pattern_x), self.LASER_MAX))
            y_clamped = max(0, min(int(pattern_y), self.LASER_MAX))

            for _ in range(points_per_pattern_point):
                if point_index < self.POINTS_PER_FRAME:
                    frame_buffer[point_index] = HeliosPoint(
                        x=x_clamped, y=y_clamped,
                        r=self.COLOR_VALUE, g=self.COLOR_VALUE, b=self.COLOR_VALUE,
                        i=self.INTENSITY
                    )
                    point_index += 1

        while point_index < self.POINTS_PER_FRAME:
            frame_buffer[point_index] = HeliosPoint(
                x=int(center_x), y=int(center_y),
                r=self.COLOR_VALUE, g=self.COLOR_VALUE, b=self.COLOR_VALUE,
                i=self.INTENSITY
            )
            point_index += 1

        return frame_buffer

    def send_frame_to_laser(self, frame_buffer):
        """Send frame data to laser"""
        if not self.HeliosLib:
            return False  # Simulation mode

        try:
            statusAttempts = 0
            while statusAttempts < 32:
                if self.HeliosLib.GetStatus(0) == 1:
                    break
                statusAttempts += 1
                time.sleep(0.001)

            self.HeliosLib.WriteFrame(0, self.FRAME_DURATION, 0,
                                      ctypes.pointer(frame_buffer), self.POINTS_PER_FRAME)
            return True
        except Exception as e:
            print(f"Error sending frame to laser: {e}")
            return False

    def is_point_in_region(self, x, y):
        """Check if point is within the defined region"""
        if len(self.region_corners_camera) < 3:
            return True

        try:
            points = np.array(self.region_corners_camera, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            result = cv2.pointPolygonTest(points, (int(x), int(y)), False)
            return result >= 0
        except Exception:
            return True

    def start_system(self):
        """Start the targeting system"""
        if self.running:
            return False

        self.running = True

        # Start camera
        if not self.start_camera():
            self.running = False
            return False

        # Start threads
        self.detection_thread = threading.Thread(target=self._detection_thread, daemon=True)
        self.targeting_thread = threading.Thread(target=self._targeting_thread, daemon=True)

        self.detection_thread.start()
        self.targeting_thread.start()

        return True

    def stop_system(self):
        """Stop the targeting system"""
        self.running = False
        self.targeting_enabled = False

        # Stop autonomous execution
        self.autonomous_follower.stop_current_execution()

        # Stop camera
        self.stop_camera()

    def _detection_thread(self):
        """Main detection and tracking thread"""
        frame_counter = 0
        last_cleanup_time = time.time()

        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Store frame for web interface
            global latest_frame
            latest_frame = frame.copy()

            frame_counter += 1
            current_time = time.time()

            # Run YOLO detection if model is available
            detected_this_frame = []
            if self.model:
                try:
                    yolo_start_time = time.time()
                    results = self.model.predict(frame, conf=self.min_confidence, verbose=False)
                    actual_yolo_time = time.time() - yolo_start_time

                    # Update YOLO delay compensation
                    self.yolo_processing_delay = 0.8 * self.yolo_processing_delay + 0.2 * actual_yolo_time
                    self.yolo_processing_delay = max(self.min_yolo_delay,
                                                     min(self.yolo_processing_delay, self.max_yolo_delay))

                    # Process detection results
                    if results and results[0].boxes is not None:
                        for box in results[0].boxes:
                            try:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                conf = float(box.conf.cpu().numpy()[0])
                                cls = int(box.cls.cpu().numpy()[0])

                                if cls == 0:  # Assuming class 0 is weed
                                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                                    if 0 <= cx < self.frame_width and 0 <= cy < self.frame_height:
                                        in_laser_region = self.is_point_in_region(cx, cy)

                                        detected_this_frame.append({
                                            'pixel_x': cx,
                                            'pixel_y': cy,
                                            'confidence': conf,
                                            'box': (x1, y1, x2, y2),
                                            'in_laser_region': in_laser_region
                                        })
                            except Exception as e:
                                continue
                except Exception as e:
                    print(f"YOLO detection error: {e}")
                    continue

            # Update weed tracking
            with self.targeting_lock:
                self._update_weed_tracking(detected_this_frame, current_time)

            # Update trajectory predictions
            for weed in self.detected_weeds:
                if weed.get('visible_this_frame', False):
                    filtered_pos = self.noise_filter.filter_position(
                        weed['weed_id'], weed['pixel_x'], weed['pixel_y']
                    )

                    self.trajectory_predictor.update_weed_position(
                        weed['weed_id'], filtered_pos[0], filtered_pos[1], current_time
                    )

                    weed['filtered_x'] = filtered_pos[0]
                    weed['filtered_y'] = filtered_pos[1]

            # Cleanup old data
            if current_time - last_cleanup_time > 5.0:
                self.trajectory_predictor.cleanup_old_trajectories(current_time)
                self._cleanup_old_weeds(current_time)
                last_cleanup_time = current_time

            time.sleep(0.03)  # ~30 FPS

    def _update_weed_tracking(self, detected_this_frame, current_time):
        """Update weed tracking with new detections"""
        for weed in self.detected_weeds:
            weed['visible_this_frame'] = False

        used_detections = set()

        if self.detected_weeds and detected_this_frame:
            existing_positions = np.array([[w['pixel_x'], w['pixel_y']] for w in self.detected_weeds])
            new_positions = np.array([[d['pixel_x'], d['pixel_y']] for d in detected_this_frame])

            try:
                from scipy.spatial.distance import cdist
                distances = cdist(existing_positions, new_positions)
                match_threshold = 50

                for i, weed in enumerate(self.detected_weeds):
                    best_match_idx = -1
                    min_distance = match_threshold

                    for j in range(len(detected_this_frame)):
                        if j not in used_detections and distances[i, j] < min_distance:
                            min_distance = distances[i, j]
                            best_match_idx = j

                    if best_match_idx != -1:
                        matched_detection = detected_this_frame[best_match_idx]
                        weed.update({
                            'pixel_x': matched_detection['pixel_x'],
                            'pixel_y': matched_detection['pixel_y'],
                            'confidence': matched_detection['confidence'],
                            'box': matched_detection['box'],
                            'last_seen': current_time,
                            'visible_this_frame': True,
                            'in_laser_region': matched_detection.get('in_laser_region', True)
                        })
                        used_detections.add(best_match_idx)
            except ImportError:
                # Fallback if scipy is not available
                pass

        unmatched_detections = [d for i, d in enumerate(detected_this_frame) if i not in used_detections]
        for detection in unmatched_detections:
            self.weed_counter += 1
            new_weed = detection.copy()
            new_weed.update({
                'weed_id': self.weed_counter,
                'first_seen': current_time,
                'last_seen': current_time,
                'visible_this_frame': True,
                'targeted': False,
                'in_laser_region': detection.get('in_laser_region', True)
            })
            self.detected_weeds.append(new_weed)

    def _cleanup_old_weeds(self, current_time):
        """Remove weeds not seen for a while"""
        max_age = 10.0
        self.detected_weeds = [w for w in self.detected_weeds
                               if current_time - w.get('last_seen', 0) < max_age]

    def _targeting_thread(self):
        """Enhanced targeting thread with observation-prediction-execution phases"""
        while self.running:
            if not self.targeting_enabled:
                time.sleep(0.2)
                continue

            current_time = time.time()

            with self.targeting_lock:
                if self.current_target is None:
                    self._select_new_target(current_time)
                else:
                    self._process_target_phases(current_time)

            time.sleep(0.05)

    def _select_new_target(self, current_time):
        """Select a new target with enhanced priority system"""
        execution_status = self.autonomous_follower.get_execution_status()
        executing_weed_id = execution_status['weed_id'] if execution_status else None

        available_weeds = [w for w in self.detected_weeds
                           if w.get('visible_this_frame', False)
                           and not w.get('targeted', False)
                           and w['weed_id'] != executing_weed_id
                           and w.get('in_laser_region', False)]

        if not available_weeds:
            return

        def weed_priority(weed):
            weed_id = weed['weed_id']
            movement_info = self.trajectory_predictor.get_movement_info(weed_id)
            observation_time = current_time - weed.get('first_seen', current_time)

            if movement_info and movement_info['has_movement']:
                speed_score = min(1.0, movement_info['speed'] / 10.0)
                consistency_score = movement_info['consistency']
                confidence_score = movement_info['confidence']
                time_score = min(1.0, observation_time / 0.8)

                return speed_score * consistency_score * confidence_score * time_score * 10.0
            else:
                return observation_time

        best_weed = max(available_weeds, key=weed_priority)
        observation_time = current_time - best_weed.get('first_seen', current_time)

        if observation_time >= self.observation_time:
            self.current_target = best_weed
            self.target_start_time = current_time
            self.target_phase = "PREDICTION"

    def _process_target_phases(self, current_time):
        """Process target through observation-prediction-execution phases"""
        if not self.current_target:
            return

        target_id = self.current_target['weed_id']

        # Check if target is still visible
        current_target_data = next((w for w in self.detected_weeds if w['weed_id'] == target_id), None)

        if self.target_phase == "PREDICTION":
            if current_target_data is None or not current_target_data.get('visible_this_frame', False):
                self.current_target = None
                self.target_phase = "OBSERVATION"
                return

            # Update target data
            self.current_target = current_target_data

            # Generate complete trajectory prediction with delay compensation
            total_prediction_time = self.prediction_duration + self.prediction_delay
            trajectory_data = self.trajectory_predictor.predict_complete_trajectory(
                target_id, total_prediction_time, self.speed_scaling_factor
            )

            if trajectory_data and trajectory_data['confidence'] >= self.min_confidence_for_execution:
                # Start autonomous trajectory execution
                success = self.autonomous_follower.start_trajectory_execution(target_id, trajectory_data)

                if success:
                    self.target_phase = "EXECUTION"
                else:
                    self.current_target = None
                    self.target_phase = "OBSERVATION"
            else:
                self.current_target = None
                self.target_phase = "OBSERVATION"

        elif self.target_phase == "EXECUTION":
            # Check execution status
            execution_status = self.autonomous_follower.get_execution_status()

            if not execution_status or execution_status['weed_id'] != target_id:
                # Execution completed
                if current_target_data:
                    current_target_data['targeted'] = True
                self.current_target = None
                self.target_phase = "OBSERVATION"

    def get_system_status(self):
        """Get current system status"""
        execution_status = self.autonomous_follower.get_execution_status()

        all_visible_weeds = [w for w in self.detected_weeds if w.get('visible_this_frame', False)]
        in_laser_weeds = [w for w in all_visible_weeds if w.get('in_laser_region', True)]
        out_laser_weeds = [w for w in all_visible_weeds if not w.get('in_laser_region', True)]

        moving_in_laser = sum(1 for w in in_laser_weeds
                              if self.trajectory_predictor.get_movement_info(w['weed_id']) and
                              self.trajectory_predictor.get_movement_info(w['weed_id'])['has_movement'])
        static_in_laser = len(in_laser_weeds) - moving_in_laser

        moving_out_laser = sum(1 for w in out_laser_weeds
                               if self.trajectory_predictor.get_movement_info(w['weed_id']) and
                               self.trajectory_predictor.get_movement_info(w['weed_id'])['has_movement'])
        static_out_laser = len(out_laser_weeds) - moving_out_laser

        return {
            'running': self.running,
            'targeting_enabled': self.targeting_enabled,
            'execution_mode': self.execution_mode,
            'target_phase': self.target_phase,
            'current_target_id': self.current_target['weed_id'] if self.current_target else None,
            'executing_weed_id': execution_status['weed_id'] if execution_status else None,
            'execution_progress': execution_status['progress'] if execution_status else 0,
            'weeds_in_laser': {'moving': moving_in_laser, 'static': static_in_laser},
            'weeds_out_laser': {'moving': moving_out_laser, 'static': static_out_laser},
            'observation_time': self.observation_time,
            'prediction_duration': self.prediction_duration,
            'prediction_delay': self.prediction_delay,
            'yolo_processing_delay': self.yolo_processing_delay,
            'speed_scaling_factor': self.speed_scaling_factor,
            'noise_filter_strength': self.noise_filter_strength,
            'min_confidence_for_execution': self.min_confidence_for_execution,
            'pattern_enabled': self.pattern_enabled,
            'laser_position': {'x': self.current_x, 'y': self.current_y}
        }

    def update_parameters(self, params):
        """Update system parameters"""
        if 'observation_time' in params:
            self.observation_time = max(self.min_observation_time,
                                        min(params['observation_time'], self.max_observation_time))

        if 'prediction_duration' in params:
            self.prediction_duration = max(self.min_prediction_duration,
                                           min(params['prediction_duration'], self.max_prediction_duration))

        if 'prediction_delay' in params:
            self.prediction_delay = max(self.min_prediction_delay,
                                        min(params['prediction_delay'], self.max_prediction_delay))

        if 'yolo_processing_delay' in params:
            self.yolo_processing_delay = max(self.min_yolo_delay,
                                             min(params['yolo_processing_delay'], self.max_yolo_delay))
            self.trajectory_predictor.set_yolo_delay(self.yolo_processing_delay)

        if 'speed_scaling_factor' in params:
            self.speed_scaling_factor = max(self.min_speed_scaling,
                                            min(params['speed_scaling_factor'], self.max_speed_scaling))

        if 'noise_filter_strength' in params:
            self.noise_filter_strength = max(0.0, min(params['noise_filter_strength'], 1.0))
            self.noise_filter.set_filter_strength(self.noise_filter_strength)

        if 'noise_smoothing_window' in params:
            self.noise_smoothing_window = max(1, min(params['noise_smoothing_window'], 10))
            self.noise_filter.set_smoothing_window(self.noise_smoothing_window)

        if 'noise_movement_threshold' in params:
            self.noise_movement_threshold = max(1.0, min(params['noise_movement_threshold'], 20.0))
            self.noise_filter.set_movement_threshold(self.noise_movement_threshold)

        if 'noise_outlier_threshold' in params:
            self.noise_outlier_threshold = max(10.0, min(params['noise_outlier_threshold'], 200.0))
            self.noise_filter.set_outlier_threshold(self.noise_outlier_threshold)

        if 'min_confidence_for_execution' in params:
            self.min_confidence_for_execution = max(0.1, min(params['min_confidence_for_execution'], 1.0))

        if 'pattern_enabled' in params:
            self.pattern_enabled = bool(params['pattern_enabled'])

        if 'targeting_enabled' in params:
            self.targeting_enabled = bool(params['targeting_enabled'])

    def shutdown(self):
        """Clean shutdown"""
        self.stop_system()

        if hasattr(self, 'HeliosLib') and self.HeliosLib:
            try:
                self.HeliosLib.CloseDevices()
            except Exception as e:
                print(f"Error closing Helios devices: {e}")


# Initialize the enhanced weed targeting system
enhanced_targeting_system = None


# ------------------ Extended FusionState ------------------ #
class FusionState:
    """
    A lightweight structure to track fused heading and lat/lon.
    Weighted approach:
      - Weighted heading (GPS-based vs IMU-based)
      - Weighted position (GPS vs short-range IMU)
      - Basic friction logic: if rotating but heading not changing => up speed
      - Supports static detection if movement < threshold => zero out acc
      - Dynamic offset correction if traveling > 5m in same direction => heading_offset
    """

    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0

        self.last_gps_lat = None
        self.last_gps_lon = None
        self.last_update_time = time.time()
        self.last_heading = 0.0

        self.turn_stuck_counter = 0
        self.turn_override_speed = None

        # IMU accel bias if needed
        self.imu_accX_bias = 0.0
        self.imu_accY_bias = 0.0

        # heading offset for dynamic correction
        self.heading_offset = 0.0

        # For dynamic correction
        self.last_gps_heading = None
        self.dist_same_dir = 0.0


fusion_state = FusionState()

# Thresholds
GPS_NOISE_THRESHOLD = 2  # if distance <2 => consider ~static
STATIC_ACC_THRESHOLD = 100.0  # if |accX|,|accY|<100 => consider static
DIRECTION_DIFF_MAX = 10.0  # if gps heading differs <10°, consider same direction
MIN_DISTANCE_CORRECT = 5.0  # 5m => begin offset correction
MAX_TRUST_DISTANCE = 100.0  # up to 100m => 100% trust

# Laser control system parameters
LASER_PORT = '/dev/ttyUSB0'  # Modify as needed
DEFAULT_POWER_LEVEL = 2  # Changed from 5 to 2 for safety
MAX_SAFE_POWER = 50
tracking_enabled = False


# ------------------ MQTT Communication ------------------ #
def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print("Connected to MQTT broker with result code", rc)
        mqtt_connected = True
        client.subscribe([
            ("IMU/data", 0),
            ("CAMERA/tracking", 0),
            ("camera/detections", 0),
            ("camera/frame", 0)  # Subscribe to camera frames
        ])
    else:
        print(f"Failed to connect to MQTT broker with result code {rc}")
        mqtt_connected = False


def on_message(client, userdata, msg):
    global latest_frame
    try:
        if msg.topic == "CAMERA/tracking":
            tracking_data = json.loads(msg.payload.decode())
            current_data['camera'] = tracking_data
            if tracking_enabled:
                adjust_robot_movement(tracking_data)
        elif msg.topic == 'camera/detections':
            data = json.loads(msg.payload.decode())
            current_data['weed_detections'] = data.get('detections', [])
            current_data['camera'] = data
        elif msg.topic == 'camera/frame':
            # Decode base64 image
            try:
                img_data = base64.b64decode(msg.payload)
                nparr = np.frombuffer(img_data, np.uint8)
                latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decoding camera frame: {e}")
        else:  # IMU/data
            data = json.loads(msg.payload.decode())
            mqtt_received_data.append(data)
            if len(mqtt_received_data) > 40:
                mqtt_received_data.pop(0)
            data_queue.put(data)
    except Exception as e:
        print(f"Error processing message: {e}")


def distance_meters(lat1, lon1, lat2, lon2):
    # basic haversine
    R = 6371000.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def bearing_degs(lat1, lon1, lat2, lon2):
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    if abs(dLat) > 1e-7 or abs(dLon) > 1e-7:
        # original approach
        bearing = math.degrees(math.atan2(math.radians(dLon), math.radians(dLat)))
        return (bearing + 360) % 360
    else:
        return 0.0


def process_data():
    while True:
        try:
            raw = data_queue.get()

            imu_heading = float(raw.get('Heading', 0.0))
            imu_accX = float(raw.get('accX', 0.0))
            imu_accY = float(raw.get('accY', 0.0))
            gps_lat = float(raw.get('gps:Lat', 0.0))
            gps_lon = float(raw.get('Lon', 0.0))

            now_t = time.time()
            dt = now_t - fusion_state.last_update_time
            fusion_state.last_update_time = now_t

            dist = 0.0
            gps_heading = None
            if fusion_state.last_gps_lat is not None and fusion_state.last_gps_lon is not None:
                dist = distance_meters(fusion_state.last_gps_lat, fusion_state.last_gps_lon, gps_lat, gps_lon)
                dLat = gps_lat - fusion_state.last_gps_lat
                dLon = gps_lon - fusion_state.last_gps_lon
                if abs(dLat) > 1e-7 or abs(dLon) > 1e-7:
                    gps_heading = bearing_degs(fusion_state.last_gps_lat, fusion_state.last_gps_lon, gps_lat, gps_lon)

            # apply heading_offset
            corrected_imu_heading = (imu_heading + fusion_state.heading_offset) % 360

            # Weighted heading
            alpha_imu = 0.4
            alpha_gps = 0.6
            fused_heading = corrected_imu_heading
            if gps_heading is not None:
                if dist < 3.0:
                    alpha_imu = 0.6
                    alpha_gps = 0.4
                fused_heading = alpha_imu * fusion_state.heading + alpha_gps * gps_heading
            else:
                fused_heading = corrected_imu_heading

            # Weighted lat/lon
            alpha_pos_imu = 0.4
            alpha_pos_gps = 0.6
            lat_fused = alpha_pos_imu * fusion_state.lat + alpha_pos_gps * gps_lat
            lon_fused = alpha_pos_imu * fusion_state.lon + alpha_pos_gps * gps_lon

            # If "not moving", set acc=0
            if dist < GPS_NOISE_THRESHOLD or (
                    abs(imu_accX) < STATIC_ACC_THRESHOLD and abs(imu_accY) < STATIC_ACC_THRESHOLD):
                imu_accX = 0.0
                imu_accY = 0.0

            # friction logic
            dHeading = abs(fused_heading - fusion_state.last_heading)
            if dHeading < 1.0 and dt > 0:
                fusion_state.turn_stuck_counter += 1
                if fusion_state.turn_stuck_counter > 5:
                    fusion_state.turn_override_speed = pid_params.speed + 50
            else:
                fusion_state.turn_stuck_counter = 0
                fusion_state.turn_override_speed = None

            # dynamic offset correction
            if gps_heading is not None:
                if fusion_state.last_gps_heading is None:
                    fusion_state.last_gps_heading = gps_heading
                    fusion_state.dist_same_dir = 0.0
                else:
                    dir_diff = abs(gps_heading - fusion_state.last_gps_heading)
                    # normalize to [-180..180]
                    dir_diff = ((dir_diff + 180) % 360) - 180
                    if abs(dir_diff) < DIRECTION_DIFF_MAX:
                        fusion_state.dist_same_dir += dist
                    else:
                        fusion_state.dist_same_dir = 0.0
                        fusion_state.last_gps_heading = gps_heading

                    if fusion_state.dist_same_dir > MIN_DISTANCE_CORRECT:
                        offset_suggestion = gps_heading - imu_heading
                        offset_suggestion = ((offset_suggestion + 180) % 360) - 180
                        trust = min(fusion_state.dist_same_dir, MAX_TRUST_DISTANCE)
                        ratio = trust / MAX_TRUST_DISTANCE
                        old_off = fusion_state.heading_offset
                        new_off = old_off * (1 - ratio) + offset_suggestion * ratio
                        fusion_state.heading_offset = new_off

            fusion_state.last_gps_lat = gps_lat
            fusion_state.last_gps_lon = gps_lon
            fusion_state.last_heading = fused_heading

            fusion_state.lat = lat_fused
            fusion_state.lon = lon_fused
            fusion_state.heading = fused_heading

            fused_dict = {
                'lat': fusion_state.lat,
                'lon': fusion_state.lon,
                'heading': fusion_state.heading
            }
            global current_data
            current_data['imu'] = {
                'heading': imu_heading,
                'accX': imu_accX,
                'accY': imu_accY
            }
            current_data['gps'] = {
                'lat': gps_lat,
                'lon': gps_lon
            }
            current_data['fused'] = fused_dict

        except Exception as e:
            print("Error in data processing:", e)


# ------------------ Robot & Laser Control ------------------ #
class RobotSocket:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        self.last_ping_time = 0
        self.ping_interval = 0.5  # Send ping every 0.5 second

    def connect(self):
        try:
            if self.sock:
                self.sock.close()
                time.sleep(0.1)  # Wait for socket to fully close

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)  # Add timeout setting
            print(f"Connecting to robot at {self.ip}:{self.port}...")
            self.sock.connect((self.ip, self.port))

            # Test connection with initialization commands
            init_resp = self.send_command("SYS CAL")
            print(f"Robot init response: {init_resp}")

            if init_resp and "OK" in init_resp:
                self.connected = True
                self.last_ping_time = time.time()
                # Start ping thread
                ping_thread = threading.Thread(target=self._ping_thread, daemon=True)
                ping_thread.start()
                print(f"Successfully connected to robot at {self.ip}:{self.port}")
                return True
            else:
                print("Connected but received invalid response")
                self.sock.close()
                self.sock = None
                return False

        except Exception as e:
            print(f"Error connecting to robot: {e}")
            if self.sock:
                self.sock.close()
            self.sock = None
            self.connected = False
            return False

    def disconnect(self):
        try:
            if self.sock:
                # Send explicit disconnect command
                self.send_command("SYS DISCONNECT")
                time.sleep(0.1)
                self.sock.close()
                self.sock = None
                self.connected = False
                print("Robot disconnected successfully")
                return True
        except Exception as e:
            print(f"Error disconnecting robot: {e}")
            self.sock = None
            self.connected = False
        return False

    def _ping_thread(self):
        """Thread to send regular pings to keep connection alive"""
        while self.connected and self.sock:
            try:
                current_time = time.time()
                if current_time - self.last_ping_time >= self.ping_interval:
                    self.send_command("SYS PING", read_response=False)
                    self.last_ping_time = current_time
                time.sleep(0.1)  # Sleep for a short time
            except Exception as e:
                print(f"Error in ping thread: {e}")
                self.connected = False
                break

    def send_command(self, cmd, read_response=True):
        if not self.sock:
            print("Robot socket is not connected.")
            return None

        try:
            if not cmd.endswith('\r\n'):
                cmd += '\r\n'

            # Print if not a ping command (to reduce log noise)
            if not cmd.startswith("SYS PING"):
                print(f"Sending command: {cmd.strip()}")

            self.sock.sendall(cmd.encode())

            if read_response:
                # Wait for response with a short timeout
                self.sock.settimeout(0.5)  # Short timeout for better responsiveness
                try:
                    resp = self.sock.recv(1024).decode()
                    if not cmd.startswith("SYS PING"):
                        print(f"Response: {resp.strip()}")
                    return resp
                except socket.timeout:
                    # For stop commands, don't wait for response
                    if cmd.startswith("STOP") or "!M 0 0" in cmd:
                        return "OK"
                    print(f"Timeout waiting for response to: {cmd.strip()}")
                    return None
                except Exception as e:
                    print(f"Error receiving response: {e}")
                    return None
            return "OK"  # For non-response commands

        except Exception as e:
            print(f"Error sending command: {e}")
            # Only disconnect on non-ping errors
            if not cmd.startswith("SYS PING"):
                self.sock = None
                self.connected = False
            return None


class LaserController:
    def __init__(self, port=LASER_PORT):
        self.port = port
        self.serial = None
        self.connected = False
        self.power_level = 2  # Changed from 5 to 2 for safety
        self.aim_power = 2
        self.duration = 5
        self.safety_delay = 2
        self.status = "OFF"
        self.mode = "MANUAL"  # 'MANUAL' or 'AUTO'
        self.high_power_confirmed = False

        # Directory for captures
        if not os.path.exists("static/captures"):
            os.makedirs("static/captures")

    def connect(self):
        try:
            import serial
            self.serial = serial.Serial(self.port, 115200, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            self.connected = True
            print(f"Connected to laser controller on {self.port}")
            return True
        except Exception as e:
            print(f"Error connecting to laser controller: {e}")
            return False

    def send_command(self, cmd, read_response=True):
        if not self.serial or not self.connected:
            print("Laser controller not connected")
            return False

        try:
            self.serial.write(f"{cmd}\n".encode())
            if read_response:
                time.sleep(0.1)
                if self.serial.in_waiting > 0:
                    response = self.serial.readline().decode('utf-8', errors='replace').strip()
                    print(f"Laser controller response: {response}")
            return True
        except Exception as e:
            print(f"Error sending command to laser: {e}")
            return False

    def set_power(self, power_level):
        self.power_level = power_level
        if power_level <= MAX_SAFE_POWER or self.high_power_confirmed:
            self.send_command(f"POWER {power_level}")
            self.high_power_confirmed = False
            current_data['laser']['power'] = power_level
            return True
        return False

    def set_aim_power(self, aim_power):
        self.aim_power = max(1, min(20, aim_power))
        current_data['laser']['aim_power'] = self.aim_power
        return True

    def set_duration(self, duration):
        self.duration = max(1, duration)
        current_data['laser']['duration'] = self.duration
        return True

    def set_safety_delay(self, delay):
        self.safety_delay = max(1, delay)
        current_data['laser']['safety_delay'] = self.safety_delay
        return True

    def confirm_high_power(self):
        self.high_power_confirmed = True
        if self.power_level > MAX_SAFE_POWER:
            self.send_command(f"POWER {self.power_level}")
            return True
        return False

    def toggle_laser(self):
        if self.status == "OFF":
            self.send_command("ON")
            self.status = "ON"
        else:
            self.send_command("OFF")
            self.status = "OFF"
        current_data['laser']['status'] = self.status
        return self.status

    def laser_on(self):
        self.send_command("ON")
        self.status = "ON"
        current_data['laser']['status'] = self.status
        return True

    def laser_off(self):
        self.send_command("OFF")
        self.status = "OFF"
        current_data['laser']['status'] = self.status
        return True

    def toggle_mode(self):
        if self.mode == "MANUAL":
            self.mode = "AUTO"
        else:
            self.mode = "MANUAL"
        current_data['laser']['mode'] = self.mode
        return self.mode

    def capture_image(self, prefix="capture"):
        global latest_frame
        if latest_frame is None:
            print("No camera frame available for capture")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"static/captures/{prefix}_{timestamp}.jpg"

            # Add metadata to the image
            info_text = f"Power: {self.power_level}, Duration: {self.duration}s, Mode: {self.mode}"
            cv2.putText(latest_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Save the image
            cv2.imwrite(filename, latest_frame)
            print(f"Image saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def run_experiment(self):
        if self.mode != "AUTO":
            print("Experiment can only run in AUTO mode")
            return False

        try:
            # Capture before image
            print("Capturing before image...")
            self.capture_image(prefix="before")

            # Safety delay
            print(f"Safety delay: {self.safety_delay} seconds...")
            time.sleep(self.safety_delay)

            # Set aiming power
            print(f"Setting aim power to {self.aim_power}...")
            self.send_command(f"POWER {self.aim_power}")
            time.sleep(0.5)

            # Turn on laser for aiming
            print("Turning on laser for aiming...")
            self.send_command("ON")
            self.status = "ON"
            current_data['laser']['status'] = self.status
            time.sleep(2)  # Give time to aim

            # Set treatment power
            print(f"Setting treatment power to {self.power_level}...")
            self.send_command(f"POWER {self.power_level}")
            time.sleep(0.5)

            # Run laser for duration
            print(f"Running laser for {self.duration} seconds...")
            time.sleep(self.duration)

            # Turn off laser
            print("Turning off laser...")
            self.send_command("OFF")
            self.status = "OFF"
            current_data['laser']['status'] = self.status

            # Capture after image
            print("Capturing after image...")
            time.sleep(1)  # Give time for any smoke to clear
            self.capture_image(prefix="after")

            print("Experiment completed successfully")
            return True
        except Exception as e:
            print(f"Error running experiment: {e}")
            # Make sure laser is off
            self.send_command("OFF")
            self.status = "OFF"
            current_data['laser']['status'] = self.status
            return False

    def close(self):
        if self.serial and self.connected:
            self.send_command("OFF")
            self.serial.close()
            self.connected = False
            print("Laser controller disconnected")
            return True
        return False


class EnhancedMotorController:
    def __init__(self):
        # Motor positions for multiple devices
        self.motor_positions = {
            0: {'x': 2048, 'y': 2048},  # Device 0 (Motor 1)
            1: {'x': 2048, 'y': 2048}  # Device 1 (Motor 2)
        }

        # Motor control parameters
        self.motor_min = 80
        self.motor_max = 4040
        self.move_step = 50  # Step size for keyboard movement
        self.connected = False

        # Helios DAC setup
        self.HeliosLib = None
        self.numDevices = 0
        self.frameType = None
        self.frames = []

        # Current active device
        self.current_device = 0

        print("Enhanced Motor Controller initialized")

    def get_device_count(self):
        """Get the number of connected Helios DAC devices"""
        return self.numDevices

    def set_current_device(self, device_index):
        """Set the currently active device"""
        if 0 <= device_index < self.numDevices:
            self.current_device = device_index
            print(f"Switched to Device {device_index + 1}")
            return True
        else:
            print(f"Invalid device index: {device_index}. Only {self.numDevices} devices available.")
            return False

    def get_current_device(self):
        """Get the currently active device index"""
        return self.current_device

    def set_position(self, device_index, x, y):
        """Set the position of the specified device"""
        # Ensure values are within range
        x = max(self.motor_min, min(self.motor_max, int(x)))
        y = max(self.motor_min, min(self.motor_max, int(y)))

        # Update stored positions
        if device_index not in self.motor_positions:
            self.motor_positions[device_index] = {'x': 2048, 'y': 2048}

        self.motor_positions[device_index]['x'] = x
        self.motor_positions[device_index]['y'] = y

        # Control the actual hardware if connected
        if self.connected and self.HeliosLib and device_index < self.numDevices:
            try:
                # Create frame for motor control
                frame = self.frameType()

                # Different orientation handling for different devices
                if device_index == 0:
                    # Device 0: Standard orientation
                    frame[0] = HeliosPoint(x=x, y=y, r=255, g=255, b=255, i=130)
                else:
                    # Device 1: Reversed orientation (if needed)
                    frame[0] = HeliosPoint(x=y, y=x, r=255, g=255, b=255, i=130)

                # Send to specific device
                statusAttempts = 0
                while (statusAttempts < 32 and self.HeliosLib.GetStatus(device_index) != 1):
                    statusAttempts += 1

                self.HeliosLib.WriteFrame(device_index, 30000, 0, ctypes.pointer(frame), 1)
                print(f"Device {device_index + 1} position set to X={x}, Y={y}")

            except Exception as e:
                print(f"Error controlling Device {device_index + 1}: {str(e)}")
                return False
        else:
            print(f"Setting Device {device_index + 1} position to X={x}, Y={y} (simulation)")

        return True

    def move_relative(self, device_index, dx, dy):
        """Move the device relative to its current position"""
        if device_index not in self.motor_positions:
            self.motor_positions[device_index] = {'x': 2048, 'y': 2048}

        current_pos = self.motor_positions[device_index]
        new_x = current_pos['x'] + dx
        new_y = current_pos['y'] + dy

        return self.set_position(device_index, new_x, new_y)

    def move_current_device(self, direction):
        """Move the currently active device in the specified direction"""
        dx, dy = 0, 0

        if direction == 'up':
            dy = -self.move_step
        elif direction == 'down':
            dy = self.move_step
        elif direction == 'left':
            dx = -self.move_step
        elif direction == 'right':
            dx = self.move_step
        else:
            print(f"Invalid direction: {direction}")
            return False

        return self.move_relative(self.current_device, dx, dy)

    def get_position(self, device_index):
        """Get the current position of the specified device"""
        if device_index in self.motor_positions:
            return self.motor_positions[device_index]
        else:
            return {'x': 2048, 'y': 2048}

    def get_current_position(self):
        """Get the position of the currently active device"""
        return self.get_position(self.current_device)

    def center_device(self, device_index):
        """Center the specified device"""
        return self.set_position(device_index, 2048, 2048)

    def center_current_device(self):
        """Center the currently active device"""
        return self.center_device(self.current_device)

    def get_device_info(self):
        """Get information about all connected devices"""
        devices_info = []
        for i in range(self.numDevices):
            pos = self.get_position(i)
            devices_info.append({
                'device_id': i,
                'name': f"Device {i + 1}",
                'position': pos,
                'is_current': i == self.current_device
            })
        return devices_info

    def connect(self):
        """Connect to the motor hardware"""
        try:
            print("Connecting to Enhanced Motor Controllers...")

            # Try to load the Helios DAC library
            try:
                self.HeliosLib = ctypes.cdll.LoadLibrary(r".\HeliosLaserDAC.dll")
                self.numDevices = self.HeliosLib.OpenDevices()
                print(f"Found {self.numDevices} Helios DAC devices")

                if self.numDevices > 0:
                    # Initialize frame type for sending data to DAC
                    self.frameType = HeliosPoint * 1
                    self.frames = [self.frameType() for _ in range(30)]

                    # Initialize all devices to center position
                    for i in range(self.numDevices):
                        self.set_position(i, 2048, 2048)

                    print(f"All {self.numDevices} devices initialized successfully")
                    self.connected = True

                    # Set device 0 as default if available
                    if self.numDevices > 0:
                        self.current_device = 0

                else:
                    print("No Helios DAC devices found - running in simulation mode")
                    self.numDevices = 2  # Simulate 2 devices for testing
                    self.connected = True  # Set to true for simulation mode

            except Exception as e:
                print(f"Error loading Helios DAC library: {str(e)}")
                print("Running in simulation mode with 2 virtual devices")
                self.numDevices = 2  # Simulate 2 devices
                self.connected = True

        except Exception as e:
            print(f"Error connecting to motor controllers: {str(e)}")
            self.connected = False

        return self.connected

    def disconnect(self):
        """Disconnect from the motor hardware"""
        try:
            if self.HeliosLib:
                # Close Helios DAC connections
                self.HeliosLib.CloseDevices()
                print("Helios DACs disconnected")

            print("Enhanced Motor Controllers disconnected")
            self.connected = False
            return True

        except Exception as e:
            print(f"Error disconnecting motor controllers: {str(e)}")
            self.connected = False
            return False


# Initialize controllers
robot = RobotSocket('192.168.1.104', 10001)  # Using fixed IP address
laser_controller = LaserController(LASER_PORT)
motor_controller = EnhancedMotorController()  # Add motor controller


def robot_forward():
    # Simplified commands for better response time
    robot.send_command("MMW !M {0} {0}".format(pid_params.speed), read_response=False)


def robot_backward():
    robot.send_command("MMW !M -{0} -{0}".format(pid_params.speed), read_response=False)


def robot_left():
    turn_speed = fusion_state.turn_override_speed if fusion_state.turn_override_speed else pid_params.speed
    robot.send_command("MMW !M -{0} {0}".format(turn_speed / 2), read_response=False)


def robot_right():
    turn_speed = fusion_state.turn_override_speed if fusion_state.turn_override_speed else pid_params.speed
    robot.send_command("MMW !M {0} -{0}".format(turn_speed / 2), read_response=False)


def robot_stop():
    # Send STOP command first for highest priority
    robot.send_command("STOP", read_response=False)
    # Then send explicit stop command
    robot.send_command("MMW !M 0 0", read_response=False)


def adjust_robot_movement(tracking_data):
    if not tracking_enabled:
        return

    # Use tracking data to adjust robot movement
    # This will need to be customized based on your specific tracking data format
    pass


def generate_camera_frames():
    while True:
        if latest_frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Return empty image if no frame is available
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n\r\n')
        time.sleep(0.03)  # ~30 FPS


# ------------------ Flask Routes ------------------ #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset')
def dataset_collection():
    """Dataset collection interface"""
    return render_template('dataset/index.html')


@app.route('/save-dataset-image', methods=['POST'])
def save_dataset_image():
    """Save an image for the dataset"""
    try:
        # Check if dataset directory exists, create if not
        dataset_dir = 'static/dataset'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Get the uploaded image
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        # Generate a unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{dataset_dir}/{timestamp}_{unique_id}.jpg"

        # Save the image to disk
        image_file.save(filename)

        return jsonify({'success': True, 'filename': filename}), 200
    except Exception as e:
        print(f"Error saving dataset image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/data')
def get_data():
    return jsonify(current_data)


@app.route('/mqtt-data')
def get_mqtt_data():
    return jsonify(mqtt_received_data)


@app.route('/video_feed')
def video_feed():
    """Route for streaming video from camera"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tracking-mode', methods=['POST'])
def tracking_mode():
    global tracking_enabled
    data = request.json
    tracking_enabled = data.get('enabled', False)
    return jsonify({'success': True, 'tracking': tracking_enabled})


@app.route('/robot-cmd', methods=['POST'])
def robot_cmd():
    data = request.json
    cmd = data.get('cmd', '').lower()

    if not robot.connected:
        return jsonify({'success': False, 'error': 'Robot not connected'})

    # For "key up" events (releasing keys), immediately stop
    if cmd == 'stop' or cmd == 'keyup':
        robot_stop()
        return jsonify({'success': True, 'command': 'stop'})

    # For key press events, send without waiting for response
    if cmd == 'w':
        robot_forward()
    elif cmd == 's':
        robot_backward()
    elif cmd == 'a':
        robot_left()
    elif cmd == 'd':
        robot_right()
    else:
        print("Unknown command:", cmd)
        return jsonify({'success': False, 'error': 'Unknown command'})

    return jsonify({'success': True, 'command': cmd})


@app.route('/connect-robot', methods=['POST'])
def connect_robot():
    # Check if already connected - if so, disconnect instead
    if robot.connected:
        robot.disconnect()
        time.sleep(0.2)  # Brief wait
        return jsonify({
            'success': True,
            'connected': False,
            'message': 'Robot disconnected successfully'
        })

    # Not connected, so connect
    ok = robot.connect()
    return jsonify({
        'success': ok,
        'connected': ok,
        'message': 'Robot connected successfully' if ok else 'Failed to connect to robot'
    })


@app.route('/disconnect-robot', methods=['POST'])
def disconnect_robot():
    ok = robot.disconnect()
    return jsonify({'success': ok, 'connected': False,
                    'message': 'Robot disconnected successfully' if ok else 'Failed to disconnect robot'})


@app.route('/robot-status', methods=['GET'])
def robot_status():
    return jsonify({'connected': robot.connected})


# ------------------ Enhanced Weed Targeting Routes ------------------ #

@app.route('/weed-targeting-start', methods=['POST'])
def start_weed_targeting():
    """Start the enhanced weed targeting system"""
    global enhanced_targeting_system

    try:
        if enhanced_targeting_system is None:
            enhanced_targeting_system = EnhancedWeedTargeting(
                model_path='weed3.pt',
                calibration_file='calibration_data.json'
            )

        success = enhanced_targeting_system.start_system()

        if success:
            return jsonify({
                'success': True,
                'message': 'Weed targeting system started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start weed targeting system'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting weed targeting system: {str(e)}'
        })


@app.route('/weed-targeting-stop', methods=['POST'])
def stop_weed_targeting():
    """Stop the enhanced weed targeting system"""
    global enhanced_targeting_system

    try:
        if enhanced_targeting_system:
            enhanced_targeting_system.stop_system()
            return jsonify({
                'success': True,
                'message': 'Weed targeting system stopped successfully'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Weed targeting system was not running'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping weed targeting system: {str(e)}'
        })


@app.route('/weed-targeting-status', methods=['GET'])
def get_weed_targeting_status():
    """Get the current status of the weed targeting system"""
    global enhanced_targeting_system

    try:
        if enhanced_targeting_system:
            status = enhanced_targeting_system.get_system_status()
            return jsonify({
                'success': True,
                'status': status
            })
        else:
            return jsonify({
                'success': True,
                'status': {
                    'running': False,
                    'targeting_enabled': False,
                    'message': 'System not initialized'
                }
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting status: {str(e)}'
        })


@app.route('/weed-targeting-params', methods=['POST'])
def update_weed_targeting_params():
    """Update weed targeting parameters"""
    global enhanced_targeting_system

    try:
        if not enhanced_targeting_system:
            return jsonify({
                'success': False,
                'message': 'Weed targeting system not initialized'
            })

        params = request.json
        enhanced_targeting_system.update_parameters(params)

        return jsonify({
            'success': True,
            'message': 'Parameters updated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error updating parameters: {str(e)}'
        })


@app.route('/weed-targeting-toggle', methods=['POST'])
def toggle_weed_targeting():
    """Toggle weed targeting on/off"""
    global enhanced_targeting_system

    try:
        if not enhanced_targeting_system:
            return jsonify({
                'success': False,
                'message': 'Weed targeting system not initialized'
            })

        data = request.json
        enabled = data.get('enabled', False)
        enhanced_targeting_system.targeting_enabled = enabled

        return jsonify({
            'success': True,
            'targeting_enabled': enhanced_targeting_system.targeting_enabled,
            'message': f'Targeting {"enabled" if enabled else "disabled"}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error toggling targeting: {str(e)}'
        })


@app.route('/weed-targeting-stop-execution', methods=['POST'])
def stop_weed_targeting_execution():
    """Stop current weed targeting execution"""
    global enhanced_targeting_system

    try:
        if enhanced_targeting_system:
            enhanced_targeting_system.autonomous_follower.stop_current_execution()
            return jsonify({
                'success': True,
                'message': 'Execution stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Weed targeting system not initialized'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping execution: {str(e)}'
        })


# ------------------ Motor Control Routes ------------------ #

@app.route('/motor-cmd', methods=['POST'])
def motor_cmd():
    """Handle motor control commands"""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    motor_index = data.get('motor', 0)  # 0 or 1
    cmd = data.get('cmd', '')

    if not motor_controller.connected:
        # Try to connect first
        motor_controller.connect()

    try:
        if cmd == 'position':
            # Set absolute position
            x = int(data.get('x', 2048))
            y = int(data.get('y', 2048))
            ok = motor_controller.set_position(motor_index, x, y)
            if ok:
                return jsonify({
                    'success': True,
                    'motor': motor_index,
                    'position': motor_controller.get_position(motor_index)
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to set position'})

        elif cmd == 'move':
            # Relative movement
            dx = int(data.get('dx', 0))
            dy = int(data.get('dy', 0))
            ok = motor_controller.move_relative(motor_index, dx, dy)
            if ok:
                return jsonify({
                    'success': True,
                    'motor': motor_index,
                    'position': motor_controller.get_position(motor_index)
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to move motor'})

        elif cmd == 'get':
            # Get current position
            return jsonify({
                'success': True,
                'motor': motor_index,
                'position': motor_controller.get_position(motor_index)
            })

        else:
            return jsonify({'success': False, 'error': f'Unknown command: {cmd}'})

    except Exception as e:
        print(f"Error processing motor command: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/motor-enhanced', methods=['POST'])
def motor_enhanced():
    """Handle enhanced motor control commands"""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    cmd = data.get('cmd', '')

    if not motor_controller.connected:
        motor_controller.connect()

    try:
        if cmd == 'move_direction':
            # Move current device in specified direction
            direction = data.get('direction', '')
            ok = motor_controller.move_current_device(direction)
            return jsonify({
                'success': ok,
                'device': motor_controller.get_current_device(),
                'position': motor_controller.get_current_position(),
                'direction': direction
            })

        elif cmd == 'switch_device':
            # Switch to specified device
            device_index = int(data.get('device', 0))
            ok = motor_controller.set_current_device(device_index)
            return jsonify({
                'success': ok,
                'current_device': motor_controller.get_current_device(),
                'device_count': motor_controller.get_device_count(),
                'position': motor_controller.get_current_position()
            })

        elif cmd == 'center_current':
            # Center current device
            ok = motor_controller.center_current_device()
            return jsonify({
                'success': ok,
                'device': motor_controller.get_current_device(),
                'position': motor_controller.get_current_position()
            })

        elif cmd == 'get_status':
            # Get status of all devices
            return jsonify({
                'success': True,
                'connected': motor_controller.connected,
                'device_count': motor_controller.get_device_count(),
                'current_device': motor_controller.get_current_device(),
                'devices_info': motor_controller.get_device_info()
            })

        elif cmd == 'set_step_size':
            # Set movement step size
            step_size = int(data.get('step_size', 50))
            motor_controller.move_step = max(1, min(500, step_size))
            return jsonify({
                'success': True,
                'step_size': motor_controller.move_step
            })

        else:
            return jsonify({'success': False, 'error': f'Unknown command: {cmd}'})

    except Exception as e:
        print(f"Error processing enhanced motor command: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/motor-status', methods=['GET'])
def motor_status():
    """Get current motor system status"""
    return jsonify({
        'success': True,
        'connected': motor_controller.connected,
        'device_count': motor_controller.get_device_count(),
        'current_device': motor_controller.get_current_device(),
        'current_position': motor_controller.get_current_position(),
        'devices_info': motor_controller.get_device_info(),
        'step_size': motor_controller.move_step
    })


# ------------------ Laser Control Routes ------------------ #

@app.route('/connect-laser', methods=['POST'])
def connect_laser():
    ok = False
    try:
        # First attempt to connect
        ok = laser_controller.connect()
        if ok:
            # Automatically initialize laser after successful connection
            time.sleep(2)  # Wait for connection to stabilize
            laser_controller.send_command("POWER 2", False)
            time.sleep(0.1)
            laser_controller.send_command("ON", False)
            laser_controller.status = "ON"
            current_data['laser']['status'] = "ON"
            print("Laser initialized and turned ON with power level 2")
    except Exception as e:
        print(f"Error connecting to laser: {e}")
        ok = False
    return jsonify({'success': ok})


@app.route('/laser-cmd', methods=['POST'])
def laser_cmd():
    data = request.json
    cmd = data.get('cmd', '')

    result = {'success': False}

    if cmd == 'toggle':
        status = laser_controller.toggle_laser()
        result = {'success': True, 'status': status}
    elif cmd == 'on':
        ok = laser_controller.laser_on()
        result = {'success': ok, 'status': 'ON'}
    elif cmd == 'off':
        ok = laser_controller.laser_off()
        result = {'success': ok, 'status': 'OFF'}
    elif cmd == 'set_power':
        power = int(data.get('value', DEFAULT_POWER_LEVEL))
        ok = laser_controller.set_power(power)
        result = {'success': ok, 'power': power}
    elif cmd == 'set_aim_power':
        power = int(data.get('value', 2))
        ok = laser_controller.set_aim_power(power)
        result = {'success': ok, 'aim_power': power}
    elif cmd == 'set_duration':
        duration = int(data.get('value', 5))
        ok = laser_controller.set_duration(duration)
        result = {'success': ok, 'duration': duration}
    elif cmd == 'set_safety_delay':
        delay = int(data.get('value', 2))
        ok = laser_controller.set_safety_delay(delay)
        result = {'success': ok, 'safety_delay': delay}
    elif cmd == 'confirm_high_power':
        ok = laser_controller.confirm_high_power()
        result = {'success': ok}
    elif cmd == 'toggle_mode':
        mode = laser_controller.toggle_mode()
        result = {'success': True, 'mode': mode}
    elif cmd == 'run_experiment':
        ok = laser_controller.run_experiment()
        result = {'success': ok}
    elif cmd == 'capture':
        prefix = data.get('prefix', 'capture')
        filename = laser_controller.capture_image(prefix)
        result = {'success': filename is not None, 'filename': filename}

    return jsonify(result)


# ------------------ PID Control Routes ------------------ #

@app.route('/pid-update', methods=['POST'])
def pid_update():
    data = request.get_json() or {}
    try:
        pid_params.kp = float(data.get('kp', pid_params.kp))
        pid_params.ki = float(data.get('ki', pid_params.ki))
        pid_params.kd = float(data.get('kd', pid_params.kd))
        pid_params.speed = int(data.get('speed', pid_params.speed))
    except Exception as e:
        print("Error updating PID:", e)
    return jsonify({
        "kp": pid_params.kp,
        "ki": pid_params.ki,
        "kd": pid_params.kd,
        "speed": pid_params.speed
    })


# ------------------ System Status Routes ------------------ #

@app.route('/check-mqtt')
def check_mqtt():
    global mqtt_connected
    return jsonify({
        'connected': mqtt_connected,
        'last_message_time': time.time()
    })


@app.route('/check-connections')
def check_connections():
    """Check all connection statuses"""
    global mqtt_connected, enhanced_targeting_system

    weed_system_status = False
    if enhanced_targeting_system:
        weed_system_status = enhanced_targeting_system.running

    return jsonify({
        'mqtt_connected': mqtt_connected,
        'robot_connected': robot.connected,
        'laser_connected': laser_controller.connected,
        'motor_connected': motor_controller.connected,
        'weed_targeting_running': weed_system_status,
        'camera_available': latest_frame is not None
    })


# ------------------ Gamepad Control Routes ------------------ #

@app.route('/gamepad-control', methods=['POST'])
def gamepad_control():
    """Handle gamepad control commands"""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    cmd_type = data.get('type', '')
    cmd_value = data.get('value', {})

    try:
        if cmd_type == 'motor':
            # Motor control
            motor_index = cmd_value.get('motor', 0)  # 0 or 1

            # Check if motor controller is connected
            if not motor_controller.connected:
                motor_controller.connect()

            # Check command type
            cmd = cmd_value.get('cmd', '')

            if cmd == 'move':
                # Relative movement
                dx = int(cmd_value.get('dx', 0))
                dy = int(cmd_value.get('dy', 0))
                ok = motor_controller.move_relative(motor_index, dx, dy)
                if ok:
                    return jsonify({
                        'success': True,
                        'motor': motor_index,
                        'position': motor_controller.get_position(motor_index)
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to move motor'})

            elif cmd == 'position':
                # Absolute position
                x = int(cmd_value.get('x', 2048))
                y = int(cmd_value.get('y', 2048))
                ok = motor_controller.set_position(motor_index, x, y)
                if ok:
                    return jsonify({
                        'success': True,
                        'motor': motor_index,
                        'position': motor_controller.get_position(motor_index)
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to set position'})

            elif cmd == 'get':
                # Get current position
                return jsonify({
                    'success': True,
                    'motor': motor_index,
                    'position': motor_controller.get_position(motor_index)
                })

            else:
                # For any other or missing command
                x_pos = cmd_value.get('x', 0)
                y_pos = cmd_value.get('y', 0)
                print(f"Motor {motor_index} control without specific command: X={x_pos}, Y={y_pos}")
                return jsonify({'success': True, 'motor': motor_index, 'position': {'x': x_pos, 'y': y_pos}})

        elif cmd_type == 'laser':
            # Laser control
            action = cmd_value.get('action', '')

            if action == 'toggle':
                status = laser_controller.toggle_laser()
                result = {'success': True, 'status': status}
            elif action == 'power':
                power = int(cmd_value.get('power', DEFAULT_POWER_LEVEL))
                ok = laser_controller.set_power(power)
                result = {'success': ok, 'power': power}
            elif action == 'mode':
                mode = laser_controller.toggle_mode()
                result = {'success': True, 'mode': mode}
            elif action == 'experiment':
                if laser_controller.mode != 'AUTO':
                    result = {'success': False, 'error': 'Experiment can only be run in AUTO mode'}
                else:
                    ok = laser_controller.run_experiment()
                    result = {'success': ok}
            else:
                result = {'success': False, 'error': f'Unknown laser action: {action}'}

        elif cmd_type == 'slider':
            # Slider control
            slider = cmd_value.get('slider', '')
            value = cmd_value.get('value', 0)

            if slider == 'power':
                ok = laser_controller.set_power(int(value))
                result = {'success': ok, 'power': value}
            elif slider == 'aim_power':
                ok = laser_controller.set_aim_power(int(value))
                result = {'success': ok, 'aim_power': value}
            elif slider == 'duration':
                ok = laser_controller.set_duration(int(value))
                result = {'success': ok, 'duration': value}
            elif slider == 'safety_delay':
                ok = laser_controller.set_safety_delay(int(value))
                result = {'success': ok, 'safety_delay': value}
            else:
                result = {'success': False, 'error': f'Unknown slider: {slider}'}

        else:
            result = {'success': False, 'error': f'Unknown command type: {cmd_type}'}

        return jsonify(result)

    except Exception as e:
        print(f"Error processing gamepad command: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/list-ports')
def list_ports():
    """List all available serial ports"""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        port_list = [{"device": p.device, "description": p.description} for p in ports]
        return jsonify({"success": True, "ports": port_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_reconnect_delay = 1  # Start with 1 second delay
mqtt_reconnect_max_delay = 60  # Maximum delay of 60 seconds
mqtt_broker_ip = "192.168.1.103"
mqtt_broker_port = 1883
mqtt_connected = False


def mqtt_connect_with_retry():
    global mqtt_reconnect_delay, mqtt_connected

    def on_mqtt_connect_fail(client, userdata, flags, rc):
        global mqtt_reconnect_delay
        print(f"MQTT connection failed with code {rc}, retrying in {mqtt_reconnect_delay} seconds...")
        time.sleep(mqtt_reconnect_delay)
        # Exponential backoff with cap
        mqtt_reconnect_delay = min(mqtt_reconnect_delay * 2, mqtt_reconnect_max_delay)
        mqtt_thread = threading.Thread(target=mqtt_connect_with_retry, daemon=True)
        mqtt_thread.start()

    try:
        # Set temporary failure callback
        mqtt_client.on_connect = on_mqtt_connect_fail
        mqtt_client.connect(mqtt_broker_ip, mqtt_broker_port, 60)

        # If we get here, connection succeeded, restore normal callback
        mqtt_client.on_connect = on_connect
        print(f"Connected to MQTT broker at {mqtt_broker_ip}:{mqtt_broker_port}")
        mqtt_reconnect_delay = 1  # Reset delay on successful connection
        mqtt_connected = True
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"MQTT connection error: {e}, retrying in {mqtt_reconnect_delay} seconds...")
        time.sleep(mqtt_reconnect_delay)
        # Exponential backoff with cap
        mqtt_reconnect_delay = min(mqtt_reconnect_delay * 2, mqtt_reconnect_max_delay)
        mqtt_thread = threading.Thread(target=mqtt_connect_with_retry, daemon=True)
        mqtt_thread.start()


# Start the data processing thread
process_thread = threading.Thread(target=process_data, daemon=True)
process_thread.start()

# Start the MQTT client with retry capability
mqtt_thread = threading.Thread(target=mqtt_connect_with_retry, daemon=True)
mqtt_thread.start()

# Register disconnect handler for clean shutdown
import atexit


@atexit.register
def shutdown():
    print("Application shutting down, cleaning up connections...")
    if robot.connected:
        robot.disconnect()
    if laser_controller.connected:
        laser_controller.close()
    if motor_controller.connected:
        motor_controller.disconnect()
    if enhanced_targeting_system:
        enhanced_targeting_system.shutdown()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5049, debug=True)
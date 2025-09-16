# -*- coding: utf-8 -*-
import ctypes
import cv2
import numpy as np
import time
from ultralytics import YOLO
import keyboard
import threading
import json
import os
from scipy.spatial import KDTree
from collections import deque
import math
import serial
import serial.tools.list_ports


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
        observation_time = trajectory.get('observation_duration', 0)
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
            print(f"Using FAST trajectory method for Weed #{weed_id} (start-end based)")
        else:
            # Fallback to average velocity method
            trajectory_velocity = trajectory.get('average_velocity', np.array([0.0, 0.0]))
            print(f"Using average velocity method for Weed #{weed_id}")

        # Apply speed scaling factor to match real movement speed
        trajectory_velocity = trajectory_velocity * speed_scaling_factor
        print(
            f"Applied speed scaling: {speed_scaling_factor:.2f} -> Speed: {np.linalg.norm(trajectory_velocity):.1f}px/s")

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


class DualMotorAutonomousTrajectoryFollower:
    """Dual motor autonomous trajectory follower - both lasers follow same predicted path"""

    def __init__(self, parent_system):
        self.parent = parent_system
        self.active_execution = None
        self.execution_threads = []  # List for multiple threads
        self.execution_lock = threading.Lock()

    def start_dual_trajectory_execution(self, weed_id, trajectory_data):
        """Start trajectory following execution for both motors simultaneously"""
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
                'is_running': True,
                'motors_active': []  # Track which motors are active
            }

            # Start execution threads for both motors
            self.execution_threads = []

            # Start thread for motor 0
            thread_m0 = threading.Thread(
                target=self._trajectory_execution_thread_for_motor,
                args=(0,),
                daemon=True
            )
            thread_m0.start()
            self.execution_threads.append(thread_m0)
            self.active_execution['motors_active'].append(0)

            # Start thread for motor 1 if available
            if self.parent.numDevices > 1:
                thread_m1 = threading.Thread(
                    target=self._trajectory_execution_thread_for_motor,
                    args=(1,),
                    daemon=True
                )
                thread_m1.start()
                self.execution_threads.append(thread_m1)
                self.active_execution['motors_active'].append(1)

            delay_info = ""
            if trajectory_data.get('delay_compensated', False):
                compensation = trajectory_data.get('compensation_applied', 0)
                delay_info = f" (Delay compensated: +{compensation:.2f}s)"

            motors_str = "Motor 1" if self.parent.numDevices == 1 else "Motors 1&2"
            print(f"Started DUAL autonomous trajectory execution for Weed #{weed_id} on {motors_str}")
            print(f"Trajectory: {len(trajectory_data['trajectory_points'])} points, "
                  f"Duration: {trajectory_data['timestamps'][-1]:.1f}s, "
                  f"Confidence: {trajectory_data['confidence']:.2f}{delay_info}")

            # Turn on laser when starting execution
            if self.parent.targeting_enabled and self.parent.esp32_connected:
                if not self.parent.laser_enabled:
                    self.parent.laser_enabled = True
                    self.parent.send_laser_command("ON", False)
                    print("Laser automatically turned ON for tracking")

            return True

    def stop_current_execution(self):
        """Stop current trajectory execution"""
        if self.active_execution:
            self.active_execution['is_running'] = False
            print(f"Stopping trajectory execution for Weed #{self.active_execution['weed_id']}")

        for thread in self.execution_threads:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)

        self.execution_threads = []

        # Turn off laser when stopping execution in auto mode
        if self.parent.targeting_enabled and self.parent.esp32_connected:
            if self.parent.laser_enabled:
                self.parent.laser_enabled = False
                self.parent.send_laser_command("OFF", False)
                print("Laser automatically turned OFF (no target)")

    def _trajectory_execution_thread_for_motor(self, motor_index):
        """Trajectory execution thread for a specific motor"""
        execution = self.active_execution
        if not execution:
            return

        weed_id = execution['weed_id']
        trajectory_data = execution['trajectory_data']
        trajectory_points = trajectory_data['trajectory_points']
        timestamps = trajectory_data['timestamps']

        print(f"Trajectory execution thread started for Weed #{weed_id} on Motor {motor_index + 1}")

        start_time = execution['start_time']  # Use shared start time
        last_update_time = start_time

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
                    print(f"Trajectory execution completed for Weed #{weed_id} on Motor {motor_index + 1}")
                    break

                # Get target position
                target_position = trajectory_points[target_point_index]

                # Apply noise filtering (use different filter ID for each motor)
                filtered_pos = self.parent.noise_filter.filter_position(
                    f"exec_{weed_id}_m{motor_index}", target_position[0], target_position[1]
                )

                # Transform to laser coordinates for specific motor
                laser_x, laser_y = self.parent.transform_coordinates(
                    motor_index, filtered_pos[0], filtered_pos[1]
                )

                # Send laser command to specific motor
                if self.parent.pattern_enabled:
                    frame_buffer = self.parent.create_pattern_frame(laser_x, laser_y)
                else:
                    frame_buffer = self.parent.create_frame_data(laser_x, laser_y)

                self.parent.send_frame_to_motor(frame_buffer, motor_index)

                # Update current laser position for specific motor
                self.parent.current_x_per_motor[motor_index] = laser_x
                self.parent.current_y_per_motor[motor_index] = laser_y

                # Debug output (every 0.5 seconds, only from motor 0 to avoid duplicate messages)
                if motor_index == 0 and current_time - last_update_time >= 0.5:
                    remaining_time = timestamps[-1] - elapsed_time
                    print(f"Executing Weed #{weed_id}: Point {target_point_index}/{len(trajectory_points) - 1}, "
                          f"Remaining: {remaining_time:.1f}s")
                    last_update_time = current_time

                # Control update frequency
                time.sleep(1.0 / self.parent.pattern_update_rate)

        except Exception as e:
            print(f"Error in trajectory execution thread for Motor {motor_index + 1}: {e}")
        finally:
            # Only clear active execution when both threads are done
            with self.execution_lock:
                if self.active_execution and self.active_execution['weed_id'] == weed_id:
                    # Remove this motor from active list
                    if motor_index in self.active_execution['motors_active']:
                        self.active_execution['motors_active'].remove(motor_index)

                    # If no motors active, clear execution
                    if not self.active_execution['motors_active']:
                        self.active_execution = None
                        # Turn off laser when all motors done
                        if self.parent.targeting_enabled and self.parent.esp32_connected:
                            if self.parent.laser_enabled:
                                self.parent.laser_enabled = False
                                self.parent.send_laser_command("OFF", False)
                                print("Laser automatically turned OFF (tracking complete)")

            print(f"Trajectory execution thread ended for Motor {motor_index + 1}")

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
            'motors_active': execution['motors_active'],
            'elapsed_time': elapsed_time,
            'total_duration': total_duration,
            'progress': min(1.0, elapsed_time / total_duration),
            'is_running': execution['is_running']
        }


class EnhancedWeedTargeting:
    def __init__(self, model_path='weed4.pt',
                 calibration_file_motor0='calibration_data_motor_0.json',
                 calibration_file_motor1='calibration_data_motor_1.json'):
        print("Initializing Enhanced Predictive Weed Targeting System (Dual Motor)...")

        # --- YOLO Model ---
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model from: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model '{model_path}': {e}")
            raise

        # --- Dual Motor Calibration Data ---
        self.calibration_points_per_motor = [[] for _ in range(2)]
        self.region_corners_laser_per_motor = [[] for _ in range(2)]
        self.region_corners_camera_per_motor = [[] for _ in range(2)]
        self.kdtree_per_motor = [None, None]
        self.valid_calibration_indices_per_motor = [[] for _ in range(2)]

        # Load calibration data for both motors
        self.load_calibration_data(0, calibration_file_motor0)
        self.load_calibration_data(1, calibration_file_motor1)
        self.prepare_kdtree(0)
        self.prepare_kdtree(1)

        # --- Laser Control Settings ---
        self.LASER_MAX = 0xFFF
        self.POINTS_PER_FRAME = 1000
        self.FRAME_DURATION = 30000
        self.COLOR_VALUE = 255
        self.INTENSITY = 130

        # Current laser position for each motor
        self.current_x_per_motor = [self.LASER_MAX // 2, self.LASER_MAX // 2]
        self.current_y_per_motor = [self.LASER_MAX // 2, self.LASER_MAX // 2]

        # --- Helios DAC Setup ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_dir, "HeliosLaserDAC.dll")
            if not os.path.exists(dll_path):
                dll_path = "./HeliosLaserDAC.dll"
            self.HeliosLib = ctypes.cdll.LoadLibrary(dll_path)
        except OSError as e:
            print(f"Error loading HeliosLaserDAC.dll: {e}")
            raise

        self.numDevices = self.HeliosLib.OpenDevices()
        print(f"Found {self.numDevices} Helios DACs")
        if self.numDevices == 0:
            raise Exception("No Helios DAC found")
        if self.numDevices < 2:
            print("Warning: Only 1 DAC found. Motor 2 control will not function.")

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(1.0)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")

        # --- ESP32 Laser Control ---
        self.esp32 = None
        self.laser_enabled = False
        self.laser_power = 128  # Default power (0-255)
        self.esp32_connected = False

        # Power slider variables
        self.slider_x = self.frame_width - 220  # Position on right side
        self.slider_y = 50
        self.slider_width = 200
        self.slider_height = 20
        self.slider_handle_width = 15
        self.slider_dragging = False

        # Initialize ESP32 connection
        self.connect_to_esp32()

        # --- Enhanced Prediction Systems ---
        self.trajectory_predictor = WeedTrajectoryPredictor()
        self.noise_filter = AdvancedNoiseFilter(filter_strength=0.3, smoothing_window=5)
        self.laser_shape_generator = LaserShapeGenerator()
        self.autonomous_follower = DualMotorAutonomousTrajectoryFollower(self)  # Use dual motor follower

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
        self.max_speed_scaling = 3.0

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
        self.running = True
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

        print("Enhanced Predictive Weed Targeting System (Dual Motor) initialized successfully")
        print("Mode: OBSERVE (1s) → PREDICT → DUAL MOTOR AUTONOMOUS EXECUTION")
        print("Features: Dual motor synchronized tracking, Auto laser control, ESP32 power control")

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
        """Toggle laser ON/OFF (manual control only)"""
        if not self.esp32_connected:
            print("ESP32 not connected - cannot control laser")
            return

        # In auto targeting mode, don't allow manual laser control
        if self.targeting_enabled:
            print("Manual laser control disabled in AUTO mode")
            return

        self.laser_enabled = not self.laser_enabled
        if self.laser_enabled:
            self.send_laser_command("ON", False)
            print("Laser manually turned ON")
        else:
            self.send_laser_command("OFF", False)
            print("Laser manually turned OFF")

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

        # Laser status text with auto mode indicator
        if self.targeting_enabled:
            status_text = "Laser: AUTO"
            status_color = (255, 165, 0)  # Orange for auto mode
        else:
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
            print(f"Error: Invalid motor_index {motor_index}")
            return False

        print(f"Loading calibration data for Motor {motor_index + 1} from {filename}...")
        if not os.path.exists(filename):
            print(f"  Calibration file not found: {filename}")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.calibration_points_per_motor[motor_index] = data.get('calibration_points', [])
            self.region_corners_laser_per_motor[motor_index] = data.get('region_corners_laser', [])
            self.region_corners_camera_per_motor[motor_index] = data.get('region_corners_camera', [])

            print(
                f"  Loaded {len(self.calibration_points_per_motor[motor_index])} calibration points for Motor {motor_index + 1}")
            return True
        except Exception as e:
            print(f"  Failed to load calibration data for Motor {motor_index + 1}: {e}")
            return False

    def prepare_kdtree(self, motor_index):
        """Prepare KD tree for a specific motor"""
        if not (0 <= motor_index <= 1):
            return False

        calibration_points = self.calibration_points_per_motor[motor_index]
        if len(calibration_points) < 1:
            print(f"Warning: No calibration points available for Motor {motor_index + 1}")
            return False

        valid_points = []
        valid_indices = []

        for i, point in enumerate(calibration_points):
            if ('camera_pixel_x' in point and 'camera_pixel_y' in point and
                    isinstance(point['camera_pixel_x'], (int, float)) and
                    isinstance(point['camera_pixel_y'], (int, float))):
                valid_points.append([point['camera_pixel_x'], point['camera_pixel_y']])
                valid_indices.append(i)

        if not valid_points:
            print(f"Warning: No valid calibration points found for Motor {motor_index + 1}")
            return False

        try:
            camera_points = np.array(valid_points, dtype=np.float32)
            self.kdtree_per_motor[motor_index] = KDTree(camera_points)
            self.valid_calibration_indices_per_motor[motor_index] = valid_indices
            print(f"Built KD tree for Motor {motor_index + 1} with {len(valid_points)} valid points")
            return True
        except Exception as e:
            print(f"Error building KD tree for Motor {motor_index + 1}: {e}")
            return False

    def transform_coordinates(self, motor_index, camera_x, camera_y):
        """Transform camera coordinates to laser coordinates for specific motor"""
        if not (0 <= motor_index <= 1):
            return self.LASER_MAX // 2, self.LASER_MAX // 2

        kdtree = self.kdtree_per_motor[motor_index]
        valid_indices = self.valid_calibration_indices_per_motor[motor_index]
        calibration_points = self.calibration_points_per_motor[motor_index]

        if kdtree is None or not valid_indices:
            # Fallback to simple linear mapping
            laser_x = int((camera_x / self.frame_width) * self.LASER_MAX)
            laser_y = int((camera_y / self.frame_height) * self.LASER_MAX)
            return max(0, min(laser_x, self.LASER_MAX)), max(0, min(laser_y, self.LASER_MAX))

        try:
            k = min(self.weighted_k, len(valid_indices))
            if k < 1:
                return self.LASER_MAX // 2, self.LASER_MAX // 2

            dists, kdtree_idxs = kdtree.query([camera_x, camera_y], k=k)

            if k == 1:
                dists, kdtree_idxs = [dists], [kdtree_idxs]

            original_idxs = [valid_indices[i] for i in kdtree_idxs]
            weights = 1.0 / (np.maximum(dists, 1e-9) ** 2)
            weights /= np.sum(weights)

            laser_x, laser_y = 0.0, 0.0
            for i, orig_idx in enumerate(original_idxs):
                point = calibration_points[orig_idx]
                laser_x += point['laser_x'] * weights[i]
                laser_y += point['laser_y'] * weights[i]

            return int(round(laser_x)), int(round(laser_y))

        except Exception as e:
            print(f"Error in coordinate transformation for Motor {motor_index + 1}: {e}")
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

    def send_frame_to_motor(self, frame_buffer, motor_idx):
        """Send frame buffer to a specific motor device"""
        if motor_idx >= self.numDevices:
            return False

        try:
            statusAttempts = 0
            while statusAttempts < 32:
                if self.HeliosLib.GetStatus(motor_idx) == 1:
                    break
                statusAttempts += 1
                time.sleep(0.001)

            self.HeliosLib.WriteFrame(motor_idx, self.FRAME_DURATION, 0,
                                      ctypes.pointer(frame_buffer), self.POINTS_PER_FRAME)
            return True
        except Exception as e:
            print(f"Error sending frame to motor {motor_idx}: {e}")
            return False

    def is_point_in_region(self, motor_index, x, y):
        """Check if point is within the defined region for specific motor"""
        if not (0 <= motor_index <= 1):
            return True

        region_corners_camera = self.region_corners_camera_per_motor[motor_index]
        if len(region_corners_camera) < 3:
            return True

        try:
            points = np.array(region_corners_camera, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            result = cv2.pointPolygonTest(points, (int(x), int(y)), False)
            return result >= 0
        except Exception:
            return True

    def update_noise_filter_settings(self):
        """Update noise filter with current settings"""
        self.noise_filter.set_filter_strength(self.noise_filter_strength)
        self.noise_filter.set_smoothing_window(self.noise_smoothing_window)
        self.noise_filter.set_movement_threshold(self.noise_movement_threshold)
        self.noise_filter.set_outlier_threshold(self.noise_outlier_threshold)

    def update_yolo_delay_compensation(self):
        """Update YOLO delay compensation in trajectory predictor"""
        self.trajectory_predictor.set_yolo_delay(self.yolo_processing_delay)

    def detection_thread(self):
        """Main detection and tracking thread"""
        frame_counter = 0
        last_cleanup_time = time.time()
        last_yolo_time = time.time()

        # Set mouse callback for slider
        cv2.namedWindow("Autonomous Predictive Weed Targeting - Dual Motor")
        cv2.setMouseCallback("Autonomous Predictive Weed Targeting - Dual Motor", self.mouse_callback)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_counter += 1
            current_time = time.time()

            # Measure YOLO processing time
            yolo_start_time = time.time()

            # Run YOLO detection
            try:
                results = self.model.predict(frame, conf=self.min_confidence, verbose=False)
            except Exception as e:
                print(f"YOLO detection error: {e}")
                continue

            # Calculate actual YOLO processing time
            actual_yolo_time = time.time() - yolo_start_time

            # Update YOLO delay compensation (moving average)
            self.yolo_processing_delay = 0.8 * self.yolo_processing_delay + 0.2 * actual_yolo_time
            self.yolo_processing_delay = max(self.min_yolo_delay, min(self.yolo_processing_delay, self.max_yolo_delay))

            # Process detection results - detect all weeds in camera view
            detected_this_frame = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf.cpu().numpy()[0])
                        cls = int(box.cls.cpu().numpy()[0])

                        if cls == 0:  # Assuming class 0 is weed
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                            # Detect all weeds in camera view (not just laser region)
                            if 0 <= cx < self.frame_width and 0 <= cy < self.frame_height:
                                # Check if weed is in laser targeting region for each motor
                                in_laser_region_m0 = self.is_point_in_region(0, cx, cy)
                                in_laser_region_m1 = self.is_point_in_region(1, cx,
                                                                             cy) if self.numDevices > 1 else False

                                detected_this_frame.append({
                                    'pixel_x': cx,
                                    'pixel_y': cy,
                                    'confidence': conf,
                                    'box': (x1, y1, x2, y2),
                                    'in_laser_region_m0': in_laser_region_m0,
                                    'in_laser_region_m1': in_laser_region_m1
                                })
                    except Exception as e:
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

            # Display frame
            self._display_frame(frame, current_time)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                self.running = False
                break

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event handler for slider interaction"""
        self.handle_slider_mouse(event, x, y)

    def _update_weed_tracking(self, detected_this_frame, current_time):
        """Update weed tracking with new detections"""
        for weed in self.detected_weeds:
            weed['visible_this_frame'] = False

        used_detections = set()

        if self.detected_weeds and detected_this_frame:
            existing_positions = np.array([[w['pixel_x'], w['pixel_y']] for w in self.detected_weeds])
            new_positions = np.array([[d['pixel_x'], d['pixel_y']] for d in detected_this_frame])

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
                        'in_laser_region_m0': matched_detection.get('in_laser_region_m0', True),
                        'in_laser_region_m1': matched_detection.get('in_laser_region_m1', False)
                    })
                    used_detections.add(best_match_idx)

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
                'in_laser_region_m0': detection.get('in_laser_region_m0', True),
                'in_laser_region_m1': detection.get('in_laser_region_m1', False)
            })
            self.detected_weeds.append(new_weed)

    def _cleanup_old_weeds(self, current_time):
        """Remove weeds not seen for a while"""
        max_age = 10.0
        self.detected_weeds = [w for w in self.detected_weeds
                               if current_time - w.get('last_seen', 0) < max_age]

    def targeting_thread(self):
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

        # Find weeds that are in either motor region
        available_weeds = [w for w in self.detected_weeds
                           if w.get('visible_this_frame', False)
                           and not w.get('targeted', False)
                           and w['weed_id'] != executing_weed_id
                           and (w.get('in_laser_region_m0', False) or w.get('in_laser_region_m1', False))]

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

            movement_info = self.trajectory_predictor.get_movement_info(best_weed['weed_id'])
            motors_str = "Motors 1&2" if self.numDevices > 1 else "Motor 1"

            if movement_info and movement_info['has_movement']:
                print(f"Selected MOVING target for {motors_str}: Weed #{best_weed['weed_id']} "
                      f"(Speed: {movement_info['speed']:.1f}px/s, "
                      f"Obs: {observation_time:.1f}s, Delay: {movement_info['processing_delay']:.2f}s)")
            else:
                print(f"Selected STATIC target for {motors_str}: Weed #{best_weed['weed_id']} "
                      f"(Obs: {observation_time:.1f}s)")

    def _process_target_phases(self, current_time):
        """Process target through observation-prediction-execution phases"""
        if not self.current_target:
            return

        target_id = self.current_target['weed_id']

        # Check if target is still visible
        current_target_data = next((w for w in self.detected_weeds if w['weed_id'] == target_id), None)

        if self.target_phase == "PREDICTION":
            if current_target_data is None or not current_target_data.get('visible_this_frame', False):
                print(f"Target lost during prediction phase: Weed #{target_id}")
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
                # Start dual motor autonomous trajectory execution
                success = self.autonomous_follower.start_dual_trajectory_execution(
                    target_id, trajectory_data
                )

                if success:
                    self.target_phase = "EXECUTION"
                    motors_str = "Motors 1&2" if self.numDevices > 1 else "Motor 1"
                    print(f"Started DUAL AUTONOMOUS execution for Weed #{target_id} on {motors_str} "
                          f"(Prediction delay: +{self.prediction_delay:.1f}s)")
                else:
                    print(f"Failed to start execution for Weed #{target_id}")
                    self.current_target = None
                    self.target_phase = "OBSERVATION"
            else:
                confidence = trajectory_data['confidence'] if trajectory_data else 0.0
                print(f"Insufficient confidence for Weed #{target_id} "
                      f"(confidence: {confidence:.2f}, required: {self.min_confidence_for_execution:.2f})")
                self.current_target = None
                self.target_phase = "OBSERVATION"

        elif self.target_phase == "EXECUTION":
            # Check execution status
            execution_status = self.autonomous_follower.get_execution_status()

            if not execution_status or execution_status['weed_id'] != target_id:
                # Execution completed
                if current_target_data:
                    current_target_data['targeted'] = True
                motors_str = "Motors 1&2" if self.numDevices > 1 else "Motor 1"
                print(f"Execution completed for Weed #{target_id} on {motors_str}")
                self.current_target = None
                self.target_phase = "OBSERVATION"

    def control_thread(self):
        """Enhanced keyboard controls for dual motor system"""
        print("\n=== DUAL MOTOR PREDICTIVE WEED TARGETING CONTROLS ===")
        print("T: Toggle Auto Targeting (with auto laser control)")
        print("L: Toggle Laser ON/OFF (manual mode only)")
        print("UP/DOWN: Adjust Observation Time (current: {:.1f}s)".format(self.observation_time))
        print("LEFT/RIGHT: Adjust Prediction Duration (current: {:.1f}s)".format(self.prediction_duration))
        print("PgUp/PgDn: Adjust Prediction Delay (current: {:.1f}s)".format(self.prediction_delay))
        print("HOME/END: Adjust YOLO Delay Compensation (current: {:.2f}s)".format(self.yolo_processing_delay))
        print("O/K: Adjust Speed Scaling Factor (current: {:.2f})".format(self.speed_scaling_factor))
        print("")
        print("NOISE FILTERING CONTROLS:")
        print("Q/A: Filter Strength (current: {:.2f})".format(self.noise_filter_strength))
        print("W/S: Smoothing Window (current: {})".format(self.noise_smoothing_window))
        print("E/D: Movement Threshold (current: {:.1f})".format(self.noise_movement_threshold))
        print("R/F: Outlier Threshold (current: {:.1f})".format(self.noise_outlier_threshold))
        print("")
        print("U/I: Adjust Min Execution Confidence (current: {:.2f})".format(self.min_confidence_for_execution))
        print("Y: Toggle Laser Patterns")
        print("C: Toggle Execution Mode")
        print("Z: Stop Current Execution")
        print("X: Toggle Noise Stats Display")
        print("V: Toggle Region Display")
        print("P: Toggle Trajectory Predictions Display")
        print("ESC: Exit")
        print("=====================================================\n")

        while self.running:
            try:
                if keyboard.is_pressed('t'):
                    self.targeting_enabled = not self.targeting_enabled
                    print(f"Auto Targeting: {'ENABLED' if self.targeting_enabled else 'DISABLED'}")

                    # Turn off laser when disabling auto mode
                    if not self.targeting_enabled and self.esp32_connected:
                        if self.laser_enabled:
                            self.laser_enabled = False
                            self.send_laser_command("OFF", False)
                            print("Laser turned OFF (auto mode disabled)")

                    time.sleep(0.3)

                elif keyboard.is_pressed('l'):
                    self.toggle_laser()
                    time.sleep(0.3)

                elif keyboard.is_pressed('up'):
                    self.observation_time = min(self.observation_time + 0.1, self.max_observation_time)
                    print(f"Observation Time: {self.observation_time:.1f}s")
                    time.sleep(0.1)

                elif keyboard.is_pressed('down'):
                    self.observation_time = max(self.observation_time - 0.1, self.min_observation_time)
                    print(f"Observation Time: {self.observation_time:.1f}s")
                    time.sleep(0.1)

                elif keyboard.is_pressed('right'):
                    self.prediction_duration = min(self.prediction_duration + 0.5, self.max_prediction_duration)
                    print(f"Prediction Duration: {self.prediction_duration:.1f}s")
                    time.sleep(0.1)

                elif keyboard.is_pressed('left'):
                    self.prediction_duration = max(self.prediction_duration - 0.5, self.min_prediction_duration)
                    print(f"Prediction Duration: {self.prediction_duration:.1f}s")
                    time.sleep(0.1)

                elif keyboard.is_pressed('page up'):
                    self.prediction_delay = min(self.prediction_delay + 0.1, self.max_prediction_delay)
                    print(f"Prediction Delay: {self.prediction_delay:.1f}s (laser aims further ahead)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('page down'):
                    self.prediction_delay = max(self.prediction_delay - 0.1, self.min_prediction_delay)
                    print(f"Prediction Delay: {self.prediction_delay:.1f}s (laser aims closer)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('home'):
                    self.yolo_processing_delay = min(self.yolo_processing_delay + 0.1, self.max_yolo_delay)
                    self.update_yolo_delay_compensation()
                    print(f"YOLO Delay Compensation: {self.yolo_processing_delay:.2f}s")
                    time.sleep(0.1)

                elif keyboard.is_pressed('end'):
                    self.yolo_processing_delay = max(self.yolo_processing_delay - 0.1, self.min_yolo_delay)
                    self.update_yolo_delay_compensation()
                    print(f"YOLO Delay Compensation: {self.yolo_processing_delay:.2f}s")
                    time.sleep(0.1)

                elif keyboard.is_pressed('o'):
                    self.speed_scaling_factor = min(self.speed_scaling_factor + 0.05, self.max_speed_scaling)
                    print(f"Speed Scaling Factor: {self.speed_scaling_factor:.2f} (Higher = Faster laser movement)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('k'):
                    self.speed_scaling_factor = max(self.speed_scaling_factor - 0.05, self.min_speed_scaling)
                    print(f"Speed Scaling Factor: {self.speed_scaling_factor:.2f} (Lower = Slower laser movement)")
                    time.sleep(0.1)

                # Noise filtering controls
                elif keyboard.is_pressed('q'):
                    self.noise_filter_strength = min(self.noise_filter_strength + 0.05, 1.0)
                    self.update_noise_filter_settings()
                    print(f"Noise Filter Strength: {self.noise_filter_strength:.2f} (Higher = More Filtering)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('a'):
                    self.noise_filter_strength = max(self.noise_filter_strength - 0.05, 0.0)
                    self.update_noise_filter_settings()
                    print(f"Noise Filter Strength: {self.noise_filter_strength:.2f} (Lower = Less Filtering)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('w'):
                    self.noise_smoothing_window = min(self.noise_smoothing_window + 1, 10)
                    self.update_noise_filter_settings()
                    print(f"Smoothing Window: {self.noise_smoothing_window} (Larger = More Smoothing)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('s'):
                    self.noise_smoothing_window = max(self.noise_smoothing_window - 1, 1)
                    self.update_noise_filter_settings()
                    print(f"Smoothing Window: {self.noise_smoothing_window} (Smaller = Less Smoothing)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('e'):
                    self.noise_movement_threshold = min(self.noise_movement_threshold + 0.5, 20.0)
                    self.update_noise_filter_settings()
                    print(f"Movement Threshold: {self.noise_movement_threshold:.1f}px (Higher = Less Sensitive)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('d'):
                    self.noise_movement_threshold = max(self.noise_movement_threshold - 0.5, 1.0)
                    self.update_noise_filter_settings()
                    print(f"Movement Threshold: {self.noise_movement_threshold:.1f}px (Lower = More Sensitive)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('r'):
                    self.noise_outlier_threshold = min(self.noise_outlier_threshold + 5.0, 200.0)
                    self.update_noise_filter_settings()
                    print(f"Outlier Threshold: {self.noise_outlier_threshold:.1f}px (Higher = More Tolerant)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('f'):
                    self.noise_outlier_threshold = max(self.noise_outlier_threshold - 5.0, 10.0)
                    self.update_noise_filter_settings()
                    print(f"Outlier Threshold: {self.noise_outlier_threshold:.1f}px (Lower = More Strict)")
                    time.sleep(0.1)

                elif keyboard.is_pressed('u'):
                    self.min_confidence_for_execution = min(self.min_confidence_for_execution + 0.05, 1.0)
                    print(f"Min Execution Confidence: {self.min_confidence_for_execution:.2f}")
                    time.sleep(0.1)

                elif keyboard.is_pressed('i'):
                    self.min_confidence_for_execution = max(self.min_confidence_for_execution - 0.05, 0.1)
                    print(f"Min Execution Confidence: {self.min_confidence_for_execution:.2f}")
                    time.sleep(0.1)

                elif keyboard.is_pressed('y'):
                    self.pattern_enabled = not self.pattern_enabled
                    print(f"Laser Patterns: {'ENABLED' if self.pattern_enabled else 'DISABLED'}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('c'):
                    modes = ["AUTONOMOUS", "REALTIME"]
                    current_idx = modes.index(self.execution_mode)
                    self.execution_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"Execution Mode: {self.execution_mode}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('z'):
                    self.autonomous_follower.stop_current_execution()
                    print("Stopped current execution")
                    time.sleep(0.3)

                elif keyboard.is_pressed('x'):
                    self.show_noise_stats = not self.show_noise_stats
                    print(f"Noise Stats Display: {'ON' if self.show_noise_stats else 'OFF'}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('v'):
                    self.show_region = not self.show_region
                    print(f"Region Display: {'ON' if self.show_region else 'OFF'}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('p'):
                    self.show_predictions = not self.show_predictions
                    print(f"Prediction Display: {'ON' if self.show_predictions else 'OFF'}")
                    time.sleep(0.3)

                elif keyboard.is_pressed('esc'):
                    print("ESC pressed, exiting...")
                    self.running = False
                    break

            except Exception as e:
                print(f"Control thread error: {e}")
                time.sleep(1)

            time.sleep(0.02)

    def _display_frame(self, frame, current_time):
        """Enhanced display with dual motor status and power slider"""
        display_frame = frame.copy()

        # Draw regions for both motors
        if self.show_region:
            # Motor 0 region in green
            if len(self.region_corners_camera_per_motor[0]) >= 3:
                points = np.array(self.region_corners_camera_per_motor[0], dtype=np.int32)
                cv2.polylines(display_frame, [points], True, (0, 255, 0), 2)

            # Motor 1 region in blue (if available)
            if self.numDevices > 1 and len(self.region_corners_camera_per_motor[1]) >= 3:
                points = np.array(self.region_corners_camera_per_motor[1], dtype=np.int32)
                cv2.polylines(display_frame, [points], True, (255, 0, 0), 2)

        # Draw power slider
        self.draw_power_slider(display_frame)

        # Get execution status
        execution_status = self.autonomous_follower.get_execution_status()
        executing_weed_id = execution_status['weed_id'] if execution_status else None
        executing_motors = execution_status['motors_active'] if execution_status else []

        # Draw weeds and trajectories
        with self.targeting_lock:
            current_target_id = self.current_target['weed_id'] if self.current_target else None

            for weed in self.detected_weeds:
                if not weed.get('visible_this_frame', False):
                    continue

                weed_id = weed['weed_id']
                x1, y1, x2, y2 = weed['box']
                cx, cy = int(weed['pixel_x']), int(weed['pixel_y'])

                movement_info = self.trajectory_predictor.get_movement_info(weed_id)

                # Determine which motor region the weed is in
                in_m0 = weed.get('in_laser_region_m0', False)
                in_m1 = weed.get('in_laser_region_m1', False)

                # Enhanced color coding based on location and status
                if weed_id == executing_weed_id:
                    color = (0, 0, 255)  # Red for executing
                    motors_str = "M1&2" if len(
                        executing_motors) > 1 else f"M{executing_motors[0] + 1}" if executing_motors else "M?"
                    tag = f"[EXEC {motors_str}]"
                elif weed_id == current_target_id:
                    if self.target_phase == "PREDICTION":
                        color = (255, 165, 0)  # Orange for predicting
                        tag = "[PREDICTING]"
                    else:
                        color = (255, 255, 0)  # Yellow for target
                        tag = "[TARGET]"
                elif weed.get('targeted', False):
                    color = (128, 128, 128)  # Gray for completed
                    tag = "[DONE]"
                elif not in_m0 and not in_m1:  # Outside both regions
                    color = (180, 180, 180)  # Light gray for out of range
                    if movement_info and movement_info['has_movement']:
                        tag = f"[OUT-MOVING {movement_info['speed']:.1f}px/s]"
                    else:
                        tag = "[OUT-STATIC]"
                elif in_m0 and in_m1:  # In both regions
                    color = (255, 0, 255)  # Magenta for in both
                    tag = "[M1+M2]"
                elif in_m0:  # In motor 0 region only
                    color = (0, 255, 0)  # Green
                    tag = "[M1]"
                elif in_m1:  # In motor 1 region only
                    color = (255, 0, 0)  # Blue
                    tag = "[M2]"
                else:
                    color = (0, 255, 255)  # Cyan for moving
                    tag = "[UNKNOWN]"

                thickness = 3 if weed_id in [current_target_id, executing_weed_id] else 2
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(display_frame, (cx, cy), 5, color, -1)

                # Draw trajectory
                if self.show_trajectories and weed_id in self.trajectory_predictor.weed_trajectories:
                    trajectory = self.trajectory_predictor.weed_trajectories[weed_id]
                    positions = list(trajectory['positions'])

                    if len(positions) > 1:
                        for i in range(1, len(positions)):
                            pt1 = tuple(map(int, positions[i - 1]))
                            pt2 = tuple(map(int, positions[i]))
                            cv2.line(display_frame, pt1, pt2, (255, 255, 0), 2)

                # Draw filtered position (blue dot)
                if 'filtered_x' in weed and 'filtered_y' in weed:
                    filtered_x = int(weed['filtered_x'])
                    filtered_y = int(weed['filtered_y'])
                    cv2.circle(display_frame, (filtered_x, filtered_y), 3, (255, 0, 0), -1)

                    # Draw line from raw to filtered position
                    if abs(filtered_x - cx) > 2 or abs(filtered_y - cy) > 2:
                        cv2.line(display_frame, (cx, cy), (filtered_x, filtered_y), (255, 0, 0), 1)

                # Draw movement vector
                if (self.show_movement_vectors and movement_info and
                        movement_info['has_movement'] and movement_info['speed'] > 1.0):
                    direction = movement_info['direction']
                    speed = movement_info['speed']

                    arrow_length = min(100, speed * 3)
                    end_x = int(cx + direction[0] * arrow_length)
                    end_y = int(cy + direction[1] * arrow_length)

                    cv2.arrowedLine(display_frame, (cx, cy), (end_x, end_y),
                                    (0, 255, 0), 3, tipLength=0.3)
                    cv2.putText(display_frame, f"{speed:.1f}px/s",
                                (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw predicted trajectory for target
                if (self.show_predictions and weed_id == current_target_id and
                        self.target_phase == "PREDICTION"):
                    total_prediction_time = self.prediction_duration + self.prediction_delay
                    trajectory_data = self.trajectory_predictor.predict_complete_trajectory(
                        weed_id, total_prediction_time, self.speed_scaling_factor
                    )

                    if trajectory_data and trajectory_data['trajectory_points']:
                        # Draw complete predicted trajectory
                        points = trajectory_data['trajectory_points']
                        for i in range(1, min(len(points), 50)):
                            pt1 = tuple(map(int, points[i - 1]))
                            pt2 = tuple(map(int, points[i]))
                            cv2.line(display_frame, pt1, pt2, (255, 0, 255), 2)

                        # Mark end point
                        if len(points) > 1:
                            end_point = tuple(map(int, points[-1]))
                            cv2.circle(display_frame, end_point, 8, (255, 0, 255), 3)
                            cv2.putText(display_frame, f"END {total_prediction_time:.1f}s",
                                        (end_point[0] + 10, end_point[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                label = f"W{weed_id}{tag}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show noise filter stats if enabled
                if self.show_noise_stats and weed_id == current_target_id:
                    stats = self.noise_filter.get_filter_stats(weed_id)
                    if stats:
                        stats_text = f"Filter: Pos={stats['positions_tracked']}, Vel={stats['velocities_tracked']}"
                        cv2.putText(display_frame, stats_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Enhanced status overlay
        status_y = 30
        status_color = (0, 255, 255)

        mode = "AUTO" if self.targeting_enabled else "MANUAL"
        phase = self.target_phase if self.current_target else "IDLE"

        cv2.putText(display_frame,
                    f"Mode: {mode} | Phase: {phase} | Motors: {self.numDevices} | Exec: {self.execution_mode}",
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        status_y += 25

        cv2.putText(display_frame,
                    f"Obs: {self.observation_time:.1f}s | Pred: {self.prediction_duration:.1f}s | "
                    f"Delay: +{self.prediction_delay:.1f}s | YOLO: {self.yolo_processing_delay:.2f}s | "
                    f"Speed: x{self.speed_scaling_factor:.2f}",
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        status_y += 20

        # Noise filtering status
        cv2.putText(display_frame,
                    f"Noise Filter: Strength={self.noise_filter_strength:.2f}, "
                    f"Window={self.noise_smoothing_window}, MoveTh={self.noise_movement_threshold:.1f}, "
                    f"OutlierTh={self.noise_outlier_threshold:.1f}",
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        status_y += 18

        # Motor positions
        cv2.putText(display_frame,
                    f"M1 DAC: ({self.current_x_per_motor[0]}, {self.current_y_per_motor[0]}) | "
                    f"M2 DAC: ({self.current_x_per_motor[1]}, {self.current_y_per_motor[1]})",
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        status_y += 20

        # ESP32 status
        esp32_status = "ESP32: Connected" if self.esp32_connected else "ESP32: Not Connected"
        esp32_color = (0, 255, 0) if self.esp32_connected else (0, 0, 255)
        cv2.putText(display_frame, esp32_status, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, esp32_color, 1)
        status_y += 20

        # Execution status
        if execution_status:
            progress = execution_status['progress'] * 100
            motors_str = "Motors 1&2" if len(
                execution_status['motors_active']) > 1 else f"Motor {execution_status['motors_active'][0] + 1}" if \
            execution_status['motors_active'] else "Motor ?"
            cv2.putText(display_frame,
                        f"EXECUTING Weed #{execution_status['weed_id']} on {motors_str}: "
                        f"{progress:.1f}% ({execution_status['elapsed_time']:.1f}s/"
                        f"{execution_status['total_duration']:.1f}s)",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_y += 25

        # Weed counts - separate by motor regions
        all_visible_weeds = [w for w in self.detected_weeds if w.get('visible_this_frame', False)]

        in_m0_weeds = [w for w in all_visible_weeds if w.get('in_laser_region_m0', False)]
        in_m1_weeds = [w for w in all_visible_weeds if w.get('in_laser_region_m1', False)]
        out_both_weeds = [w for w in all_visible_weeds if
                          not w.get('in_laser_region_m0', False) and not w.get('in_laser_region_m1', False)]

        moving_m0 = sum(1 for w in in_m0_weeds
                        if self.trajectory_predictor.get_movement_info(w['weed_id']) and
                        self.trajectory_predictor.get_movement_info(w['weed_id'])['has_movement'])
        static_m0 = len(in_m0_weeds) - moving_m0

        moving_m1 = sum(1 for w in in_m1_weeds
                        if self.trajectory_predictor.get_movement_info(w['weed_id']) and
                        self.trajectory_predictor.get_movement_info(w['weed_id'])['has_movement'])
        static_m1 = len(in_m1_weeds) - moving_m1

        cv2.putText(display_frame,
                    f"M1: {moving_m0}M, {static_m0}S | M2: {moving_m1}M, {static_m1}S | OUT: {len(out_both_weeds)} | "
                    f"Min Conf: {self.min_confidence_for_execution:.2f}",
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        cv2.imshow("Autonomous Predictive Weed Targeting - Dual Motor", display_frame)

    def run(self):
        """Start the enhanced dual motor targeting system"""
        print("Starting Autonomous Predictive Weed Targeting System (Dual Motor)...")
        print("Operation Mode: OBSERVE (1s) → PREDICT → DUAL MOTOR AUTONOMOUS EXECUTE")
        print("Features:")
        print("- Dual motor synchronized tracking (both lasers track same target)")
        print("- Automatic laser control (ON during tracking, OFF when idle)")
        print("- ESP32 laser power control with slider")
        print("- Fast trajectory prediction using start-end positions")
        print("- Adjustable speed scaling to match real movement")
        print("- Full camera detection with dual laser region targeting")
        print("- Advanced multi-stage noise filtering for shaky movements")
        print("- YOLO processing delay compensation (auto-measured)")
        print("- Adjustable prediction delay for hardware latency")
        print("- Real-time noise filter parameter adjustment")

        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        targeting_thread = threading.Thread(target=self.targeting_thread, daemon=True)
        control_thread = threading.Thread(target=self.control_thread, daemon=True)

        detection_thread.start()
        targeting_thread.start()
        control_thread.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Shutting down...")
            self.running = False
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down Autonomous Predictive Weed Targeting System (Dual Motor)...")
        self.running = False

        # Stop autonomous execution
        self.autonomous_follower.stop_current_execution()

        # Turn off laser before exit
        if self.esp32_connected:
            self.send_laser_command("OFF", False)
            print("Laser turned OFF")

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        if hasattr(self, 'HeliosLib'):
            try:
                self.HeliosLib.CloseDevices()
            except Exception as e:
                print(f"Error closing Helios devices: {e}")

        if self.esp32 and self.esp32.is_open:
            self.esp32.close()
            print("ESP32 connection closed")

        cv2.destroyAllWindows()
        print("Shutdown complete")


if __name__ == "__main__":
    try:
        model_file = 'weed4.pt'
        calibration_file_motor0 = 'calibration_data_motor_0.json'
        calibration_file_motor1 = 'calibration_data_motor_1.json'

        targeting_system = EnhancedWeedTargeting(
            model_path=model_file,
            calibration_file_motor0=calibration_file_motor0,
            calibration_file_motor1=calibration_file_motor1
        )
        targeting_system.run()

    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found: {e}")
        print("Please ensure calibration files and model file exist.")
    except Exception as e:
        print(f"\nCritical Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("Application finished.")
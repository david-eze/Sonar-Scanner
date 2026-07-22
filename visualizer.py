#!/usr/bin/env python3
"""
AI-Enhanced Sonar Scanner Visualizer
Real-time polar-to-cartesian mapping with computer vision pipeline

Features:
- Real-time occupancy grid mapping with decay (fading radar trail)
- DBSCAN clustering for object detection
- Bounding boxes with velocity vectors
- AI confidence labels
- Polar coordinate visualization
"""

import serial
import json
import numpy as np
import cv2
import time
from collections import deque
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import math

class SonarVisualizer:
    def __init__(self, serial_port='COM3', baud_rate=921600):
        """
        Initialize the sonar visualizer
        
        Args:
            serial_port: Serial port for ESP32 connection
            baud_rate: Baud rate for serial communication
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # Grid parameters
        self.grid_size = 400  # 400x400 pixel grid
        self.scale = 100      # pixels per meter
        self.center = (self.grid_size // 2, self.grid_size // 2)
        
        # Occupancy grid with decay
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.decay_rate = 0.98  # Decay factor for radar trail
        
        # Measurement history for velocity calculation
        self.max_history = 100
        self.measurement_history = deque(maxlen=self.max_history)
        
        # Object class names
        self.class_names = {
            0: "Unknown",
            1: "Wall/Flat",
            2: "Corner/Edge",
            3: "Dynamic/Moving",
            4: "Human/Soft"
        }
        
        # Class colors (BGR)
        self.class_colors = {
            0: (128, 128, 128),  # Gray - Unknown
            1: (0, 255, 0),      # Green - Wall
            2: (255, 0, 255),    # Magenta - Corner
            3: (0, 255, 255),    # Cyan - Dynamic
            4: (255, 0, 0)       # Red - Human
        }
        
        # Initialize serial connection
        self.serial_conn = None
        self.connect_serial()
        
    def connect_serial(self):
        """Establish serial connection with ESP32"""
        try:
            self.serial_conn = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=1
            )
            print(f"Connected to {self.serial_port} at {self.baud_rate} baud")
        except serial.SerialException as e:
            print(f"Failed to connect to serial port: {e}")
            print("Continuing in simulation mode...")
            self.serial_conn = None
    
    def polar_to_cartesian(self, distance, angle_deg):
        """
        Convert polar coordinates to Cartesian coordinates
        
        Mathematical transformation:
        x = r * cos(θ)
        y = r * sin(θ)
        
        Args:
            distance: Radial distance (meters)
            angle_deg: Angle in degrees (0 = forward, positive = right)
            
        Returns:
            tuple: (x, y) in meters
        """
        angle_rad = math.radians(angle_deg)
        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)
        return x, y
    
    def cartesian_to_grid(self, x, y):
        """
        Convert Cartesian coordinates to grid coordinates
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            
        Returns:
            tuple: (grid_x, grid_y) pixel coordinates
        """
        grid_x = int(self.center[0] + x * self.scale)
        grid_y = int(self.center[1] - y * self.scale)  # Flip Y for image coordinates
        return grid_x, grid_y
    
    def update_occupancy_grid(self, x, y, confidence):
        """
        Update occupancy grid with new measurement
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            confidence: Measurement confidence (0-1)
        """
        grid_x, grid_y = self.cartesian_to_grid(x, y)
        
        # Check bounds
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            # Add point with confidence weight
            self.occupancy_grid[grid_y, grid_x] = max(
                self.occupancy_grid[grid_y, grid_x],
                confidence
            )
            
            # Add small Gaussian blob around point for better visualization
            radius = 3
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        dist = math.sqrt(dx*dx + dy*dy)
                        weight = confidence * math.exp(-dist * dist / (2 * radius * radius))
                        self.occupancy_grid[ny, nx] = max(
                            self.occupancy_grid[ny, nx],
                            weight
                        )
    
    def decay_occupancy_grid(self):
        """Apply exponential decay to occupancy grid for fading radar trail"""
        self.occupancy_grid *= self.decay_rate
        # Remove very small values
        self.occupancy_grid[self.occupancy_grid < 0.01] = 0
    
    def extract_points_from_grid(self):
        """
        Extract high-confidence points from occupancy grid for clustering
        
        Returns:
            numpy array: Nx2 array of (x, y) coordinates in meters
        """
        points = []
        threshold = 0.3
        
        for y in range(0, self.grid_size, 2):  # Step by 2 for performance
            for x in range(0, self.grid_size, 2):
                if self.occupancy_grid[y, x] > threshold:
                    # Convert back to Cartesian
                    mx = (x - self.center[0]) / self.scale
                    my = (self.center[1] - y) / self.scale
                    points.append([mx, my])
        
        return np.array(points) if points else np.empty((0, 2))
    
    def cluster_points(self, points):
        """
        Apply DBSCAN clustering to detect objects
        
        DBSCAN Parameters:
        - eps: Maximum distance between points in same cluster (meters)
        - min_samples: Minimum points to form a cluster
        
        Args:
            points: Nx2 array of (x, y) coordinates
            
        Returns:
            tuple: (labels, cluster_centers)
        """
        if len(points) < 3:
            return np.array([]), np.empty((0, 2))
        
        # DBSCAN clustering
        eps = 0.3  # 30cm cluster radius
        min_samples = 3
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # Calculate cluster centers
        unique_labels = set(labels)
        cluster_centers = []
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            mask = labels == label
            cluster_points = points[mask]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        
        return labels, np.array(cluster_centers)
    
    def calculate_velocity(self, current_pos, history):
        """
        Calculate velocity vector from measurement history
        
        Args:
            current_pos: Current (x, y) position
            history: Deque of previous measurements
            
        Returns:
            tuple: (vx, vy) velocity in m/s
        """
        if len(history) < 2:
            return 0.0, 0.0
        
        # Find most recent measurement from similar position
        for i in range(len(history) - 1, -1, -1):
            prev_pos = history[i]
            dist = math.sqrt(
                (current_pos[0] - prev_pos[0])**2 + 
                (current_pos[1] - prev_pos[1])**2
            )
            
            if dist < 0.5:  # Within 50cm
                dt = (time.time() - prev_pos[2]) if len(prev_pos) > 2 else 0.1
                if dt > 0:
                    vx = (current_pos[0] - prev_pos[0]) / dt
                    vy = (current_pos[1] - prev_pos[1]) / dt
                    return vx, vy
        
        return 0.0, 0.0
    
    def draw_occupancy_grid(self, image):
        """
        Draw occupancy grid on image
        
        Args:
            image: OpenCV image to draw on
        """
        # Create colored representation
        colored_grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Map confidence to color intensity (green)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                confidence = self.occupancy_grid[y, x]
                if confidence > 0.01:
                    intensity = int(confidence * 255)
                    colored_grid[y, x] = (0, intensity, 0)
        
        # Blend with original image
        image = cv2.addWeighted(image, 0.5, colored_grid, 0.5, 0)
        return image
    
    def draw_detections(self, image, cluster_centers, labels, points):
        """
        Draw bounding boxes and velocity vectors for detected objects
        
        Args:
            image: OpenCV image to draw on
            cluster_centers: Array of cluster centers
            labels: Cluster labels for each point
            points: Array of point coordinates
        """
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            mask = labels == label
            cluster_points = points[mask]
            
            if len(cluster_points) < 3:
                continue
            
            # Calculate bounding box
            min_x = np.min(cluster_points[:, 0])
            max_x = np.max(cluster_points[:, 0])
            min_y = np.min(cluster_points[:, 1])
            max_y = np.max(cluster_points[:, 1])
            
            # Convert to grid coordinates
            top_left = self.cartesian_to_grid(min_x, max_y)
            bottom_right = self.cartesian_to_grid(max_x, min_y)
            
            # Draw bounding box
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), 2)
            
            # Draw cluster center
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_grid = self.cartesian_to_grid(center_x, center_y)
            cv2.circle(image, center_grid, 5, (255, 255, 0), -1)
    
    def draw_polar_overlay(self, image, angle, distance):
        """
        Draw polar coordinate overlay (radar sweep line)
        
        Args:
            image: OpenCV image to draw on
            angle: Current angle in degrees
            distance: Current distance in meters
        """
        # Calculate sweep line endpoint
        x, y = self.polar_to_cartesian(distance, angle)
        end_point = self.cartesian_to_grid(x, y)
        
        # Draw sweep line
        cv2.line(image, self.center, end_point, (0, 255, 255), 1)
        
        # Draw current position marker
        cv2.circle(image, end_point, 8, (0, 255, 255), 2)
    
    def draw_info_overlay(self, image, data):
        """
        Draw information overlay with telemetry data
        
        Args:
            image: OpenCV image to draw on
            data: Dictionary containing telemetry data
        """
        y_offset = 20
        line_height = 25
        
        info_text = [
            f"Distance: {data.get('d', 0):.2f} m",
            f"Velocity: {data.get('v', 0):.2f} m/s",
            f"Angle: {data.get('a', 0):.1f}°",
            f"Confidence: {data.get('c', 0):.2f}",
            f"Class: {self.class_names.get(data.get('oc', 0), 'Unknown')}",
            f"Class Conf: {data.get('cc', 0):.2f}"
        ]
        
        for text in info_text:
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
    
    def process_telemetry(self, line):
        """
        Process incoming telemetry line from ESP32
        
        Args:
            line: JSON string containing telemetry data
            
        Returns:
            dict: Parsed telemetry data or None if invalid
        """
        try:
            data = json.loads(line)
            
            # Extract polar coordinates
            distance = data.get('d', 0)
            angle = data.get('a', 0)
            confidence = data.get('c', 0)
            
            # Convert to Cartesian
            x, y = self.polar_to_cartesian(distance, angle)
            
            # Update occupancy grid
            self.update_occupancy_grid(x, y, confidence)
            
            # Add to history with timestamp
            self.measurement_history.append((x, y, time.time()))
            
            return data
            
        except json.JSONDecodeError:
            return None
    
    def run(self):
        """Main visualization loop"""
        print("Starting Sonar Visualizer...")
        print("Press 'q' to quit")
        
        # Create display window
        cv2.namedWindow('Sonar Scanner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sonar Scanner', 800, 800)
        
        last_decay_time = time.time()
        
        while True:
            # Create base image
            image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            
            # Draw grid lines
            cv2.line(image, (self.center[0], 0), (self.center[0], self.grid_size), (50, 50, 50), 1)
            cv2.line(image, (0, self.center[1]), (self.grid_size, self.center[1]), (50, 50, 50), 1)
            
            # Draw range circles
            for r in [1, 2, 3, 4]:
                radius = int(r * self.scale)
                cv2.circle(image, self.center, radius, (50, 50, 50), 1)
            
            # Process serial data
            if self.serial_conn and self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    data = self.process_telemetry(line)
                    if data:
                        # Draw polar overlay
                        self.draw_polar_overlay(image, data.get('a', 0), data.get('d', 0))
                        self.draw_info_overlay(image, data)
            
            # Decay occupancy grid periodically
            if time.time() - last_decay_time > 0.1:  # Every 100ms
                self.decay_occupancy_grid()
                last_decay_time = time.time()
            
            # Draw occupancy grid
            image = self.draw_occupancy_grid(image)
            
            # Extract points and cluster
            points = self.extract_points_from_grid()
            if len(points) > 0:
                labels, cluster_centers = self.cluster_points(points)
                self.draw_detections(image, cluster_centers, labels, points)
            
            # Display image
            cv2.imshow('Sonar Scanner', image)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.serial_conn:
            self.serial_conn.close()
        cv2.destroyAllWindows()
        print("Simulation stopped")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Enhanced Sonar Scanner Visualizer')
    parser.add_argument('--port', default='COM3', help='Serial port (default: COM3)')
    parser.add_argument('--baud', type=int, default=921600, help='Baud rate (default: 921600)')
    
    args = parser.parse_args()
    
    visualizer = SonarVisualizer(serial_port=args.port, baud_rate=args.baud)
    visualizer.run()

if __name__ == '__main__':
    main()

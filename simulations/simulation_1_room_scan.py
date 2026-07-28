#!/usr/bin/env python3
"""
Simulation 1: Room Scan
Simulates the sonar scanner sweeping across a room with walls, corners, and objects.
Generates a GIF showing the real-time occupancy grid building up.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SIZE = 300
SCALE = 75
CENTER = (GRID_SIZE // 2, GRID_SIZE // 2)
MAX_DISTANCE = 4.0
SWEEP_PERIOD = 2.0
SIM_DURATION = 4.0

room_objects = [
    *[(angle, 3.0, 'wall') for angle in np.linspace(-30, 30, 10)],
    *[(angle, 2.5 / math.cos(math.radians(abs(angle) - 45)), 'wall') 
      for angle in np.linspace(-90, -30, 8) if abs(angle) > 30],
    *[(angle, 2.5 / math.cos(math.radians(45 - abs(angle))), 'wall') 
      for angle in np.linspace(30, 90, 8) if abs(angle) > 30],
    *[(angle, 1.8 / math.cos(math.radians(angle - 35)), 'corner') 
      for angle in np.linspace(30, 40, 4)],
    *[(angle, 1.12 / math.cos(math.radians(angle + 63)), 'object') 
      for angle in np.linspace(-68, -58, 4)],
]

def polar_to_cartesian(distance, angle_deg):
    angle_rad = math.radians(angle_deg)
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    return x, y

def cartesian_to_grid(x, y):
    grid_x = int(CENTER[0] + x * SCALE)
    grid_y = int(CENTER[1] - y * SCALE)
    return grid_x, grid_y

def simulate_measurement(angle_deg, room_objects, noise=0.05):
    min_dist = MAX_DISTANCE
    detected_type = 'none'
    confidence = 0.0
    beam_width = 15
    for obj_angle, obj_dist, obj_type in room_objects:
        angle_diff = abs(angle_deg - obj_angle)
        if angle_diff < beam_width:
            effective_dist = obj_dist * math.cos(math.radians(angle_diff))
            if effective_dist < min_dist:
                min_dist = effective_dist
                detected_type = obj_type
    if min_dist < MAX_DISTANCE:
        noisy_dist = min_dist + np.random.normal(0, noise)
        noisy_dist = max(0.1, min(noisy_dist, MAX_DISTANCE))
        if detected_type == 'wall':
            confidence = max(0.5, 1.0 - noisy_dist / 5.0)
        elif detected_type == 'corner':
            confidence = max(0.3, 0.8 - noisy_dist / 5.0)
        elif detected_type == 'object':
            confidence = max(0.4, 0.9 - noisy_dist / 5.0)
        confidence += np.random.normal(0, 0.05)
        confidence = max(0.1, min(1.0, confidence))
        return noisy_dist, confidence, detected_type
    return None, 0.0, 'none'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Sonar Scanner - Room Scan Simulation', fontsize=14, fontweight='bold')

occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
decay_rate = 0.95
measurement_history = []

def update(frame):
    global occupancy_grid, measurement_history
    ax1.clear()
    ax2.clear()
    
    t = frame / 15.0
    sweep_angle = 90 * math.sin(2 * math.pi * t / SWEEP_PERIOD)
    
    for _ in range(4):
        angle = sweep_angle + np.random.normal(0, 1)
        dist, conf, obj_type = simulate_measurement(angle, room_objects)
        if dist is not None:
            x, y = polar_to_cartesian(dist, angle)
            measurement_history.append((x, y, conf, obj_type, t))
            gx, gy = cartesian_to_grid(x, y)
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                occupancy_grid[gy, gx] = max(occupancy_grid[gy, gx], conf)
                radius = 2
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            dist_px = math.sqrt(dx*dx + dy*dy)
                            weight = conf * math.exp(-dist_px * dist_px / (2 * radius * radius))
                            occupancy_grid[ny, nx] = max(occupancy_grid[ny, nx], weight)
    
    occupancy_grid *= decay_rate
    occupancy_grid[occupancy_grid < 0.01] = 0
    
    ax1.imshow(occupancy_grid, cmap='viridis', origin='upper', extent=[-2, 2, -2, 2])
    ax1.set_title(f'Occupancy Grid (t={t:.1f}s)', fontsize=11)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True, alpha=0.3)
    
    end_x, end_y = polar_to_cartesian(1.5, sweep_angle)
    ax1.plot([0, end_x], [0, end_y], 'r-', alpha=0.5, linewidth=1)
    ax1.plot(0, 0, 'ro', markersize=8, label='Scanner')
    
    if measurement_history:
        recent = measurement_history[-30:]
        xs = [m[0] for m in recent]
        ys = [m[1] for m in recent]
        colors = {'wall': 'green', 'corner': 'purple', 'object': 'cyan', 'none': 'gray'}
        c = [colors.get(m[3], 'gray') for m in recent]
        ax1.scatter(xs, ys, c=c, s=8, alpha=0.6)
    
    ax1.legend(loc='upper right')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    
    angles = np.linspace(-90, 90, 90)
    distances = []
    for a in angles:
        d, _, _ = simulate_measurement(a, room_objects, noise=0.02)
        distances.append(d if d else MAX_DISTANCE)
    ax2.plot(angles, distances, 'b-', linewidth=1, alpha=0.7)
    ax2.axvline(x=sweep_angle, color='r', linestyle='--', alpha=0.5, label='Current sweep')
    ax2.set_title('Polar Scan Profile', fontsize=11)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Distance (meters)')
    ax2.set_ylim(0, MAX_DISTANCE + 0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'simulation_1_room_scan.gif')
ani = FuncAnimation(fig, update, frames=range(0, int(SIM_DURATION * 15)), interval=66, repeat=False)
ani.save(output_path, writer='pillow', fps=10, dpi=80)
print(f"Simulation 1 saved: {output_path}")
plt.close()

#!/usr/bin/env python3
"""
Simulation 4: Sweep Motion & DBSCAN Clustering
Shows the sinusoidal servo sweep motion and DBSCAN clustering
of detected objects into discrete groups with bounding boxes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SWEEP_PERIOD = 2.0
SIM_DURATION = 4.0

scene_objects = [
    {'x': 2.0, 'y': 0.0, 'type': 'wall', 'size': 0.8},
    {'x': 1.5, 'y': 1.2, 'type': 'corner', 'size': 0.4},
    {'x': 0.8, 'y': -1.0, 'type': 'moving', 'size': 0.3},
    {'x': -1.5, 'y': 0.5, 'type': 'wall', 'size': 0.6},
    {'x': -0.5, 'y': -1.8, 'type': 'human', 'size': 0.5},
]

TYPE_COLORS = {'wall': 'green', 'corner': 'purple', 'moving': 'cyan', 'human': 'red', 'unknown': 'gray'}

def simulate_sweep_measurement(t, scene_objects):
    sweep_angle = 90 * math.sin(2 * math.pi * t / SWEEP_PERIOD)
    measurements = []
    for _ in range(50):
        angle = sweep_angle + np.random.normal(0, 2)
        min_dist = 4.0
        detected_type = 'unknown'
        for obj in scene_objects:
            obj_angle = math.degrees(math.atan2(obj['y'], obj['x']))
            obj_dist = math.sqrt(obj['x']**2 + obj['y']**2)
            angle_diff = abs(angle - obj_angle)
            if angle_diff < 12:
                dist_from_center = obj_dist * math.sin(math.radians(angle_diff))
                if dist_from_center < obj['size']:
                    effective_dist = obj_dist * math.cos(math.radians(angle_diff))
                    if effective_dist < min_dist:
                        min_dist = effective_dist
                        detected_type = obj['type']
        if min_dist < 4.0:
            noise = np.random.normal(0, 0.03)
            min_dist = max(0.1, min_dist + noise)
            x = min_dist * math.cos(math.radians(angle))
            y = min_dist * math.sin(math.radians(angle))
            measurements.append((x, y, detected_type, min_dist))
    return measurements, sweep_angle

def run_dbscan(points, eps=0.3, min_samples=3):
    if len(points) < min_samples:
        return np.array([]), []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    cluster_info = []
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        cluster_pts = points[mask]
        center = np.mean(cluster_pts, axis=0)
        min_x, max_x = np.min(cluster_pts[:, 0]), np.max(cluster_pts[:, 0])
        min_y, max_y = np.min(cluster_pts[:, 1]), np.max(cluster_pts[:, 1])
        cluster_info.append({
            'center': center,
            'bbox': (min_x, min_y, max_x, max_y),
            'num_points': len(cluster_pts)
        })
    return labels, cluster_info

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Sonar Scanner - Sweep Motion & DBSCAN Clustering', fontsize=16, fontweight='bold')

all_measurements = []

def update(frame):
    global all_measurements
    ax1.clear(); ax2.clear()
    t = frame / 15.0
    measurements, sweep_angle = simulate_sweep_measurement(t, scene_objects)
    all_measurements.extend(measurements)
    if len(all_measurements) > 150:
        all_measurements = all_measurements[-150:]
    
    ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (meters)'); ax1.set_ylabel('Y (meters)')
    ax1.set_title(f'Occupancy Grid with DBSCAN Clusters (t={t:.1f}s)', fontsize=11)
    
    ax1.plot(0, 0, 'ro', markersize=10, label='Scanner', zorder=5)
    sweep_rad = math.radians(sweep_angle)
    ax1.plot([0, 1.5 * math.cos(sweep_rad)], [0, 1.5 * math.sin(sweep_rad)], 'r-', alpha=0.3, linewidth=1, zorder=2)
    
    for obj in scene_objects:
        color = TYPE_COLORS.get(obj['type'], 'gray')
        circle = plt.Circle((obj['x'], obj['y']), obj['size']/2, color=color, fill=False, linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.add_patch(circle)
        ax1.plot(obj['x'], obj['y'], marker='*', color=color, markersize=12, alpha=0.5)
    
    if all_measurements:
        xs = [m[0] for m in all_measurements]
        ys = [m[1] for m in all_measurements]
        colors = [TYPE_COLORS.get(m[2], 'gray') for m in all_measurements]
        ax1.scatter(xs, ys, c=colors, s=8, alpha=0.5, zorder=3)
    
    points = np.array([[m[0], m[1]] for m in all_measurements])
    if len(points) >= 3:
        labels, clusters = run_dbscan(points)
        for cluster in clusters:
            min_x, min_y, max_x, max_y = cluster['bbox']
            rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                           linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.8)
            ax1.add_patch(rect)
            ax1.plot(cluster['center'][0], cluster['center'][1], marker='+', color='yellow', markersize=12, linewidth=2)
            ax1.text(cluster['center'][0], cluster['center'][1] + 0.15, f'{cluster["num_points"]} pts', 
                    ha='center', fontsize=7, color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    ax1.legend(loc='upper right', fontsize=7)
    
    t_vals = np.linspace(0, SIM_DURATION, 100)
    angle_vals = [90 * math.sin(2 * math.pi * t / SWEEP_PERIOD) for t in t_vals]
    ax2.plot(t_vals, angle_vals, 'b-', linewidth=2, label='Sinusoidal sweep')
    ax2.axvline(x=t, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=sweep_angle, color='r', linestyle=':', alpha=0.5)
    ax2.plot(t, sweep_angle, 'ro', markersize=8)
    ax2.text(t, sweep_angle + 5, f'{sweep_angle:.1f}°', fontsize=9, ha='center', color='red', fontweight='bold')
    ax2.set_title('Sinusoidal Servo Sweep Profile', fontsize=11)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Angle (degrees)')
    ax2.set_ylim(-100, 100); ax2.set_xlim(0, SIM_DURATION)
    ax2.grid(True, alpha=0.3); ax2.legend(loc='upper right')
    ax2.text(0.02, 0.95, f'Period: {SWEEP_PERIOD}s\nPeak: ±90°\nSmooth continuous motion', 
            transform=ax2.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'simulation_4_sweep_clustering.gif')
ani = FuncAnimation(fig, update, frames=range(0, int(SIM_DURATION * 15)), interval=66, repeat=False)
ani.save(output_path, writer='pillow', fps=10, dpi=80)
print(f"Simulation 4 saved: {output_path}")
plt.close()

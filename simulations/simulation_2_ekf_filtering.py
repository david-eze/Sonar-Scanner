#!/usr/bin/env python3
"""
Simulation 2: EKF Filtering Demo
Shows how the Extended Kalman Filter smooths noisy sonar measurements.
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

DT = 0.01
SIM_DURATION = 4.0
NUM_STEPS = int(SIM_DURATION / DT)

t_vals = np.arange(0, SIM_DURATION, DT)

true_distance = np.zeros_like(t_vals)
true_velocity = np.zeros_like(t_vals)

for i, t in enumerate(t_vals):
    if t < 1.0:
        true_distance[i] = 3.0 - 1.5 * t
        true_velocity[i] = -1.5
    elif t < 1.5:
        true_distance[i] = 1.5
        true_velocity[i] = 0.0
    elif t < 3.0:
        true_distance[i] = 1.5 + 1.0 * (t - 1.5)
        true_velocity[i] = 1.0
    else:
        true_distance[i] = 3.0
        true_velocity[i] = 0.0

noise_std = 0.15
measurements = true_distance + np.random.normal(0, noise_std, NUM_STEPS)

class SimpleEKF:
    def __init__(self, q=0.1, r=0.3):
        self.state = np.array([0.0, 0.0])
        self.P = np.eye(2)
        self.Q = np.diag([q, q * 0.1])
        self.R = r
        self.H = np.array([[1.0, 0.0]])
        
    def predict(self, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z, confidence=1.0):
        adapted_R = self.R / (confidence * confidence + 0.01)
        S = self.H @ self.P @ self.H.T + adapted_R
        K = self.P @ self.H.T / S
        innovation = z - self.state[0]
        self.state = self.state + K.flatten() * innovation
        self.P = (np.eye(2) - np.outer(K, self.H[0])) @ self.P
        self.P[0, 1] = self.P[1, 0] = (self.P[0, 1] + self.P[1, 0]) / 2

ekf = SimpleEKF(q=0.1, r=0.3)
ekf_states = []
ekf_innovations = []

for i, z in enumerate(measurements):
    ekf.predict(DT)
    residual = abs(z - ekf.state[0])
    confidence = 1.0 / (1.0 + residual * 2)
    ekf.update(z, confidence)
    ekf_states.append(ekf.state.copy())
    ekf_innovations.append(z - ekf.state[0])

ekf_states = np.array(ekf_states)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Sonar Scanner - Extended Kalman Filter Simulation', fontsize=14, fontweight='bold')

step_skip = 5

def update(frame):
    f = frame * step_skip
    if f >= NUM_STEPS:
        f = NUM_STEPS - 1
    
    ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
    t = t_vals[f]
    
    ax1.plot(t_vals[:f], true_distance[:f], 'g-', linewidth=2, label='True')
    ax1.plot(t_vals[:f], measurements[:f], 'b.', alpha=0.3, markersize=1, label='Raw')
    ax1.plot(t_vals[:f], ekf_states[:f, 0], 'r-', linewidth=2, label='EKF')
    ax1.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title(f'Distance Tracking (t={t:.1f}s)', fontsize=10)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Distance (m)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, SIM_DURATION); ax1.set_ylim(-0.5, 4.5)
    
    ax2.plot(t_vals[:f], true_velocity[:f], 'g-', linewidth=2, label='True')
    ax2.plot(t_vals[:f], ekf_states[:f, 1], 'r-', linewidth=2, label='EKF')
    ax2.axhline(y=0, color='gray', alpha=0.3)
    ax2.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    ax2.set_title('Velocity Tracking', fontsize=10)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Velocity (m/s)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, SIM_DURATION); ax2.set_ylim(-2.5, 2.5)
    
    ax3.plot(t_vals[:f], ekf_innovations[:f], 'purple', linewidth=1)
    ax3.axhline(y=0, color='gray', alpha=0.3)
    ax3.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    ax3.set_title('Innovation (Measurement - Prediction)', fontsize=10)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Residual (m)')
    ax3.grid(True, alpha=0.3); ax3.set_xlim(0, SIM_DURATION)
    ax3.set_ylim(-0.5, 0.5)
    
    ekf_err = np.abs(ekf_states[:f, 0] - true_distance[:f])
    raw_err = np.abs(measurements[:f] - true_distance[:f])
    ax4.plot(t_vals[:f], raw_err, 'b-', alpha=0.4, label='Raw error')
    ax4.plot(t_vals[:f], ekf_err, 'r-', linewidth=2, label='EKF error')
    ax4.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
    if f > 0:
        ax4.text(0.02, 0.95, f'EKF RMSE: {np.sqrt(np.mean(ekf_err**2)):.3f}m', 
                transform=ax4.transAxes, color='red', fontsize=8, verticalalignment='top')
        ax4.text(0.02, 0.85, f'Raw RMSE: {np.sqrt(np.mean(raw_err**2)):.3f}m', 
                transform=ax4.transAxes, color='blue', fontsize=8, verticalalignment='top')
    ax4.set_title('Error Comparison', fontsize=10)
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Absolute Error (m)')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, SIM_DURATION); ax4.set_ylim(0, 0.6)
    
    plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'simulation_2_ekf_filtering.gif')
total_frames = NUM_STEPS // step_skip
ani = FuncAnimation(fig, update, frames=range(0, total_frames), interval=50, repeat=False)
ani.save(output_path, writer='pillow', fps=15, dpi=80)
print(f"Simulation 2 saved: {output_path}")
plt.close()

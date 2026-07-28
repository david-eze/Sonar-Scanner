#!/usr/bin/env python3
"""
Run all simulations and save to correct output directory.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(BASE_DIR)

print("=" * 60)
print("Running Simulation 1: Room Scan (GIF)...")
print("=" * 60)
exec(open(os.path.join(BASE_DIR, 'simulation_1_room_scan.py')).read())

print("\n" + "=" * 60)
print("Running Simulation 2: EKF Filtering (GIF)...")
print("=" * 60)
exec(open(os.path.join(BASE_DIR, 'simulation_2_ekf_filtering.py')).read())

print("\n" + "=" * 60)
print("Running Simulation 3: Object Classification (PNG)...")
print("=" * 60)
exec(open(os.path.join(BASE_DIR, 'simulation_3_object_classification.py')).read())

print("\n" + "=" * 60)
print("Running Simulation 4: Sweep & Clustering (GIF)...")
print("=" * 60)
exec(open(os.path.join(BASE_DIR, 'simulation_4_sweep_motion.py')).read())

print("\n" + "=" * 60)
print("All simulations complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)

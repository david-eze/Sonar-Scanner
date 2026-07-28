#!/usr/bin/env python3
"""
Simulation 3: Object Classification with k-NN
Demonstrates the k-Nearest Neighbors classifier distinguishing between
walls, corners, moving objects, and human-like targets.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['Unknown', 'Wall/Flat', 'Corner/Edge', 'Dynamic/Moving', 'Human/Soft']
CLASS_COLORS = ['gray', 'green', 'purple', 'cyan', 'red']
CLASS_MARKERS = ['x', 'o', 's', 'D', '^']

np.random.seed(42)

def generate_class_samples(n_samples, class_type, feature_ranges):
    samples = []
    for _ in range(n_samples):
        features = {
            'distance': np.random.uniform(feature_ranges['distance'][0], feature_ranges['distance'][1]),
            'velocity': np.random.uniform(feature_ranges['velocity'][0], feature_ranges['velocity'][1]),
            'variance': np.random.uniform(feature_ranges['variance'][0], feature_ranges['variance'][1]),
            'amplitude': np.random.uniform(feature_ranges['amplitude'][0], feature_ranges['amplitude'][1]),
            'gradient': np.random.uniform(feature_ranges['gradient'][0], feature_ranges['gradient'][1]),
            'consistency': np.random.uniform(feature_ranges['consistency'][0], feature_ranges['consistency'][1]),
        }
        samples.append((features, class_type))
    return samples

class_ranges = {
    1: {'distance': (1.0, 3.5), 'velocity': (-0.05, 0.05), 'variance': (0.01, 0.04),
        'amplitude': (0.8, 0.95), 'gradient': (-0.05, 0.05), 'consistency': (0.85, 0.98)},
    2: {'distance': (0.8, 2.5), 'velocity': (-0.2, 0.2), 'variance': (0.12, 0.25),
        'amplitude': (0.4, 0.65), 'gradient': (-0.8, 0.8), 'consistency': (0.3, 0.55)},
    3: {'distance': (0.5, 2.5), 'velocity': (-1.5, 1.5), 'variance': (0.06, 0.15),
        'amplitude': (0.55, 0.8), 'gradient': (-2.0, 2.0), 'consistency': (0.4, 0.7)},
    4: {'distance': (0.5, 3.0), 'velocity': (-0.15, 0.15), 'variance': (0.05, 0.12),
        'amplitude': (0.15, 0.4), 'gradient': (-0.2, 0.2), 'consistency': (0.3, 0.5)}
}

all_samples = []
for cls in [1, 2, 3, 4]:
    all_samples.extend(generate_class_samples(30, cls, class_ranges[cls]))

def knn_classify(test_point, samples, k=3):
    distances = []
    for features, label in samples:
        dist = 0
        for key in features:
            dist += (test_point[key] - features[key]) ** 2
        dist = math.sqrt(dist)
        distances.append((dist, label))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    weights = {}
    for dist, label in neighbors:
        weight = 1.0 / (dist + 0.001)
        weights[label] = weights.get(label, 0) + weight
    predicted = max(weights, key=weights.get)
    confidence = weights[predicted] / sum(weights.values())
    return predicted, confidence

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Sonar Scanner - k-NN Object Classification Simulation', fontsize=16, fontweight='bold')

feature_pairs = [
    ('variance', 'amplitude', 'Variance (m)', 'Amplitude (signal strength)'),
    ('velocity', 'consistency', 'Velocity (m/s)', 'Consistency'),
    ('distance', 'variance', 'Distance (m)', 'Variance (m)'),
    ('gradient', 'amplitude', 'Gradient (m/s)', 'Amplitude (signal strength)')
]

bounds = {
    'distance': (0, 4), 'velocity': (-2, 2), 'variance': (0, 0.5),
    'amplitude': (0, 1), 'gradient': (-3, 3), 'consistency': (0, 1)
}

for idx, (ax, (f1, f2, label1, label2)) in enumerate(zip(axes.flat, feature_pairs)):
    for cls in [1, 2, 3, 4]:
        cls_samples = [(s, l) for s, l in all_samples if l == cls]
        xs = [s[f1] for s, _ in cls_samples]
        ys = [s[f2] for s, _ in cls_samples]
        ax.scatter(xs, ys, c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls], 
                  label=CLASSES[cls], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    x_min, x_max = bounds[f1]
    y_min, y_max = bounds[f2]
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25))
    Z = np.zeros(xx.shape)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            test_point = {f1: xx[i, j], f2: yy[i, j]}
            for key in ['distance', 'velocity', 'variance', 'amplitude', 'gradient', 'consistency']:
                if key not in [f1, f2]:
                    test_point[key] = (bounds[key][0] + bounds[key][1]) / 2
            cls, _ = knn_classify(test_point, all_samples, k=5)
            Z[i, j] = cls
    
    ax.contourf(xx, yy, Z, alpha=0.15, levels=[0.5, 1.5, 2.5, 3.5, 4.5], 
                colors=['gray', 'green', 'purple', 'cyan', 'red'])
    ax.set_xlabel(label1, fontsize=10)
    ax.set_ylabel(label2, fontsize=10)
    ax.set_title(f'Feature Space: {label1} vs {label2}', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'simulation_3_classification_feature_space.png'), dpi=150, bbox_inches='tight')
print(f"Feature space saved: {os.path.join(OUTPUT_DIR, 'simulation_3_classification_feature_space.png')}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle('k-NN Classification Confidence by Object Type', fontsize=14, fontweight='bold')

test_scenarios = [
    ('Wall/Flat', {'distance': 2.0, 'velocity': 0.0, 'variance': 0.02, 'amplitude': 0.9, 'gradient': 0.0, 'consistency': 0.95}),
    ('Corner/Edge', {'distance': 1.8, 'velocity': 0.1, 'variance': 0.15, 'amplitude': 0.6, 'gradient': 0.5, 'consistency': 0.5}),
    ('Moving object', {'distance': 1.5, 'velocity': 0.8, 'variance': 0.1, 'amplitude': 0.7, 'gradient': 1.2, 'consistency': 0.6}),
    ('Human/Soft', {'distance': 1.8, 'velocity': 0.05, 'variance': 0.08, 'amplitude': 0.3, 'gradient': 0.1, 'consistency': 0.4}),
]

scenario_names = []
scenario_confidences = []
scenario_classes = []
scenario_colors = []

for name, test_point in test_scenarios:
    cls, conf = knn_classify(test_point, all_samples, k=5)
    scenario_names.append(name)
    scenario_confidences.append(conf)
    scenario_classes.append(CLASSES[cls])
    scenario_colors.append(CLASS_COLORS[cls])

bars = ax2.bar(scenario_names, scenario_confidences, color=scenario_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('Classification Confidence', fontsize=11)
ax2.set_xlabel('Test Scenario', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

for bar, cls_name, conf in zip(bars, scenario_classes, scenario_confidences):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{cls_name}\n({conf:.1%})', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'simulation_3_classification_confidence.png'), dpi=150, bbox_inches='tight')
print(f"Confidence chart saved: {os.path.join(OUTPUT_DIR, 'simulation_3_classification_confidence.png')}")
plt.close('all')

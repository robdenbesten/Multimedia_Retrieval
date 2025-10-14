import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import os

def parse_hist_file(file_path):
    hist_data = {}
    with open(file_path, 'r') as f:
        content = f.read()
    for key in ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']:
        match = re.search(rf'{key}:\n([0-9, \n]+)', content)
        if match:
            values = match.group(1).replace('\n', '').split(',')
            values = [int(v) for v in values if v.strip() != '']
            hist_data[key] = values
    return hist_data

def plot_histogram(ax, hist_values, color):
    if hist_values is None:
        return
    x = np.arange(len(hist_values))
    bars = ax.bar(
        x, hist_values, color=color, edgecolor='white', alpha=0.85, linewidth=1.5
    )
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, max(hist_values), num=5, dtype=int))
    ax.set_ylabel('Count')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    # Optional: round bar corners if using Matplotlib >=3.4
    for bar in bars:
        bar.set_linewidth(1.5)
        bar.set_edgecolor('white')
        try:
            bar.set_capstyle('round')
        except Exception:
            pass  # Ignore if not supported

# Use a more vibrant color palette
row_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

file_paths = [
    'ShapeDatabase_INFOMR-master/features_test/AircraftBuoyant/m1338_copy.txt', #round
    'ShapeDatabase_INFOMR-master/features_test/Cellphone/D00192_copy.txt', #flat
    'ShapeDatabase_INFOMR-master/features_test/Bottle/D00166_copy.txt',  #elongated
    'ShapeDatabase_INFOMR-master/features_test/Quadruped/D00226_copy.txt', #irregular
]

all_files_data = [parse_hist_file(fp) for fp in file_paths]

row_labels = ['A3', 'D1', 'D2', 'D3', 'D4']
row_hist_keys = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']
col_labels = ['Round \n(AircraftBuoyant/m1338)', 'Flat \n(Cellphone/D00192)', 'Elongated \n(Bottle/D00166_copy.txt)', 'irregular \n(Quadruped/D00226_copy.txt)']
row_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (hist_key, color) in enumerate(zip(row_hist_keys, row_colors)):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for j, ax in enumerate(axes):
        file_data = all_files_data[j]
        hist_values = file_data.get(hist_key)
        plot_histogram(ax, hist_values, color)
        ax.set_title(col_labels[j], fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    fig.suptitle(f'{row_labels[i]} Descriptor', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)


plt.show() # This will now display both histograms and metric plots


def parse_metrics_file(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line and not line.lower().startswith('extents'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    k, v = parts
                    try:
                        metrics[k.strip()] = float(v.strip())
                    except ValueError:
                        continue
    print(metrics)
    return metrics

def normalize_metrics_max(metrics_list, keys):
    arr = np.array([[m.get(k, np.nan) for k in keys] for m in metrics_list], dtype=float)
    col_max = np.nanmax(arr, axis=0)
    col_max[col_max == 0] = 1.0  # Avoid division by zero
    norm = arr / col_max
    return np.nan_to_num(norm)

def metrics_barplot(file_paths, labels, colors):
    keys = [
        'Surface area',
        'Sphericity',
        'Rectangularity',
        'Diameter',
        'Convexity',
        'Eccentricity'
    ]
    metrics_list = [parse_metrics_file(fp) for fp in file_paths]
    norm_metrics = normalize_metrics_max(metrics_list, keys)
    x = np.arange(len(keys))
    width = 0.8 / len(file_paths)

    #fig, ax = plt.subplots(figsize=(12, 5))
    #for i, (vals, label, color) in enumerate(zip(norm_metrics, labels, colors)):
    #    ax.bar(x + (i - (len(file_paths) - 1) / 2) * width, vals, width=width, label=label, color=color)
    #ax.set_xticks(x)
    #ax.set_xticklabels(keys, rotation=30, ha='right')
    #ax.set_ylabel('Normalized Value')
    #ax.set_title('Normalized Shape Metrics Comparison')
    #ax.legend()
    #plt.tight_layout()
    #plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (vals, label, color) in enumerate(zip(metrics_list, labels, colors)):
        y = [vals.get(k, 0) for k in keys]  # Use raw values, fill missing with 0
        ax.bar(x + (i - (len(file_paths) - 1) / 2) * width, y, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Shape Metrics Comparison')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
labels = ['Round \n(AircraftBuoyant/m1338)', 'Flat \n(Cellphone/D00192)', 'Elongated \n(Bottle/D00166)', 'irregular \n(Quadruped/D00226)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

metrics_barplot(file_paths, labels, colors)
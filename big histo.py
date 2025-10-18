import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def collect_category_data(all_features: dict):
    """Collects histogram data from the features dictionary, grouped by category."""
    descriptors = ['D1', 'D2', 'A3', 'D3', 'D4']
    data = {d: defaultdict(list) for d in descriptors}

    for rel_path, features in all_features.items():
        # Extract category from relative path (e.g., 'ants' from 'ants/D00004.obj')
        category = os.path.dirname(rel_path)
        if not category:
            continue

        if 'histograms' in features:
            for d in descriptors:
                if d in features['histograms']:
                    # Convert list back to numpy array for processing/plotting
                    hist_array = np.asarray(features['histograms'][d], dtype=float)
                    data[d][category].append(hist_array)
    return data

def plot_descriptor_grid(descriptor, cat_to_arrays, out_dir, linewidth=0.9):
    """Plots a grid of histograms for a given descriptor, one subplot per category."""
    cats = sorted(cat_to_arrays.keys())
    if not cats:
        return

    n = len(cats)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, cat in enumerate(cats):
        ax = axes[ax_idx]
        arrays = cat_to_arrays[cat]
        if not arrays:
            ax.set_title(cat)
            ax.set_axis_off()
            continue

        # Unique color per line using a continuous colormap
        cmap = plt.get_cmap('nipy_spectral')
        colors = cmap(np.linspace(0, 1, len(arrays)))

        for arr, c in zip(arrays, colors):
            x = np.arange(len(arr))
            ax.plot(x, arr, color=c, linewidth=linewidth)

        ax.set_title(cat, fontsize=9)
        ax.set_xlabel('Bin', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=8)

    for j in range(len(cats), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f'{descriptor} histograms by category', fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{descriptor}_by_category.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'Saved {descriptor} figure to: {out_path}')

def collect_metric_values(all_features: dict, metric_name: str):
    """Collects all values for a given numerical metric from the features dictionary."""
    values = []
    for features in all_features.values():
        if 'metrics' in features and metric_name in features['metrics']:
            value = features['metrics'][metric_name]
            # Ensure the metric is a number (not a list like 'extents')
            if isinstance(value, (int, float)):
                values.append(value)
    return values

def plot_metric_histogram(values, metric_name, out_dir):
    """Plots a single histogram for a given numerical metric."""
    if not values:
        print(f'No values found for metric: {metric_name}')
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=40, color='teal', edgecolor='black', alpha=0.8)
    plt.title(f'{metric_name.capitalize()} Distribution (all objects)')
    plt.xlabel(metric_name.capitalize())
    plt.ylabel('Count')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{metric_name}_histogram.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved {metric_name} histogram to: {out_path}')

def main():
    """Loads features from JSON and generates plots."""
    json_path = os.path.join('ShapeDatabase_INFOMR-master', 'Features', 'features.json')
    out_dir = 'plots'

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at '{json_path}'")
        return

    print(f"Loading features from '{json_path}'...")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_features = json.load(f)
    print(f"Loaded data for {len(all_features)} meshes.")

    # Generate descriptor plots
    hist_data = collect_category_data(all_features)
    for descriptor, cat_to_arrays in hist_data.items():
        plot_descriptor_grid(descriptor, cat_to_arrays, out_dir)

    # Generate histograms for specified numerical metrics
    metrics_to_plot = ['convexity', 'compactness', 'sphericity']
    for metric in metrics_to_plot:
        values = collect_metric_values(all_features, metric)
        plot_metric_histogram(values, metric, out_dir)

if __name__ == '__main__':
    main()

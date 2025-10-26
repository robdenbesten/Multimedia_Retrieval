# python
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Path to the features CSV file
CSV_PATH = 'ShapeDatabase_INFOMR-master/all_features.csv'

# List of relative paths for the objects to plot (original separators kept; they are normalized at lookup)
selected_rel_paths = [
    'm1338.obj',    # round
    'D00192.obj',         # flat
    'D00166.obj',            # elongated
    'D00226.obj',         # irregular
]

row_labels = ['A3', 'D1', 'D2', 'D3', 'D4']
row_hist_keys = ['A3', 'D1', 'D2', 'D3', 'D4']
col_labels = [
    'Round \n(AircraftBuoyant/m1338)',
    'Flat \n(Cellphone/D00192)',
    'Elongated \n(Bottle/D00166)',
    'Irregular \n(Quadruped/D00226)'
]
row_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def normalize_relpath(p: str) -> str:
    if p is None:
        return p
    return p.replace('\\', '/').lstrip('./')

def load_features(csv_path):
    """
    Load features from a single CSV (`Relative Path` + metrics + ext bins).
    Normalizes relative path keys to use forward slashes.
    Returns dict: rel_path -> {'metrics': {.., 'extents': np.array}, 'histograms': {'D1': arr, ...}}
    """
    hist_order = ['D1', 'D2', 'A3', 'D3', 'D4']
    metric_keys = [
        "Mesh volume", "Surface area", "Diameter", "Compactness",
        "Rectangularity", "Convexity", "Eccentricity", "Sphericity"
    ]
    ext_keys = ['extents_0', 'extents_1', 'extents_2']

    features = {}
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        header = [h.strip() for h in header]
        # map header name -> index
        h2i = {h: i for i, h in enumerate(header)}

        # precompute histogram column groups (keep header order)
        hist_cols = {}
        for h in hist_order:
            cols = [col for col in header if col.startswith(h + '_bin_')]
            cols.sort(key=lambda c: header.index(c))
            hist_cols[h] = cols

        rel_key = 'Object'
        if rel_key not in h2i:
            raise ValueError(f"CSV missing required column `{rel_key}`")

        for row in reader:
            # pad row if shorter than header
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))

            raw_rel = row[h2i[rel_key]].strip()
            rel_path = normalize_relpath(raw_rel)
            metrics = {}

            for k in metric_keys:
                if k in h2i:
                    try:
                        metrics[k] = float(row[h2i[k]])
                    except Exception:
                        metrics[k] = float('nan')
                else:
                    metrics[k] = float('nan')

            # extents as numpy array
            ex = []
            for ek in ext_keys:
                if ek in h2i:
                    try:
                        ex.append(float(row[h2i[ek]]))
                    except Exception:
                        ex.append(float('nan'))
                else:
                    ex.append(float('nan'))
            metrics['extents'] = np.array(ex, dtype=float)

            # histograms
            hists = {}
            for h, cols in hist_cols.items():
                vals = []
                for c in cols:
                    try:
                        vals.append(float(row[header.index(c)]))
                    except Exception:
                        vals.append(0.0)
                hists[h] = np.array(vals, dtype=float)

            features[rel_path] = {'metrics': metrics, 'histograms': hists}

    return features

def get_histograms_for_selected(features, rel_paths):
    # Returns a list of dicts: each dict has histograms for one object
    all_files_data = []
    for rel_path in rel_paths:
        key = normalize_relpath(rel_path)
        data = features.get(key)
        if data and 'histograms' in data:
            all_files_data.append(data['histograms'])
        else:
            all_files_data.append({})
    return all_files_data

def plot_histogram(ax, hist_values, color):
    if hist_values is None or len(hist_values) == 0:
        return
    x = np.arange(len(hist_values))
    bars = ax.bar(
        x, hist_values, color=color, edgecolor='white', alpha=0.85, linewidth=1.5
    )
    ax.set_ylim(bottom=0)
    max_val = float(np.max(hist_values)) if np.any(hist_values) else 0.0
    # avoid division by zero / invalid ticks
    if max_val > 0:
        ax.set_yticks(np.linspace(0, max_val, num=5))
    else:
        ax.set_yticks([0.0])
    ax.set_xticks([])
    ax.set_ylabel('Count')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    for bar in bars:
        bar.set_linewidth(1.5)
        bar.set_edgecolor('white')
        try:
            bar.set_capstyle('round')
        except Exception:
            pass

def get_metrics_for_selected(features, rel_paths):
    metrics_list = []
    for rel_path in rel_paths:
        key = normalize_relpath(rel_path)
        data = features.get(key)
        if data and 'metrics' in data:
            metrics_list.append(data['metrics'])
        else:
            metrics_list.append({})
    return metrics_list

def normalize_metrics_max(metrics_list, keys):
    arr = np.array([[m.get(k, np.nan) for k in keys] for m in metrics_list], dtype=float)
    col_max = np.nanmax(arr, axis=0)
    col_max[col_max == 0] = 1.0
    norm = arr / col_max
    return np.nan_to_num(norm)

def metrics_barplot(metrics_list, labels, colors):
    keys = [
        'Surface area',
        'Sphericity',
        'Rectangularity',
        'Diameter',
        'Convexity',
        'Eccentricity'
    ]

    x = np.arange(len(keys))
    width = 0.8 / max(1, len(metrics_list))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (vals, label, color) in enumerate(zip(metrics_list, labels, colors)):
        y = [vals.get(k, 0) for k in keys]
        ax.bar(x + (i - (len(metrics_list) - 1) / 2) * width, y, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('_', ' ').title() for k in keys], rotation=30, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Shape Metrics Comparison')
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    features = load_features(CSV_PATH)
    all_files_data = get_histograms_for_selected(features, selected_rel_paths)

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

    metrics_list = get_metrics_for_selected(features, selected_rel_paths)
    metrics_barplot(metrics_list, col_labels, row_colors[:len(col_labels)])

    plt.show()

if __name__ == '__main__':
    main()
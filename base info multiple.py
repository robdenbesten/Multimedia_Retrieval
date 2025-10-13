import os
import re
from pathlib import Path
import numpy as np
import matplotlib

# 1) Select a GUI backend (Windows-friendly). Fallback to Agg and save to disk.
def _select_backend():
    # Prefer Qt or Tk backends on Windows
    for b in ['QtAgg', 'Qt5Agg', 'TkAgg', 'WXAgg']:
        try:
            matplotlib.use(b, force=True)
            return b
        except Exception:
            continue
    matplotlib.use('Agg', force=True)
    return 'Agg'

BACKEND = _select_backend()
import matplotlib.pyplot as plt  # import after backend selection

# 2) IO paths
OUT_DIR = Path('plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3) Data sources (same four files you are viewing)
file_paths = [
    'ShapeDatabase_INFOMR-master/Features/AircraftBuoyant/m1338_copy.txt',
    'ShapeDatabase_INFOMR-master/Features/Knife/D01057_copy.txt',
    'ShapeDatabase_INFOMR-master/Features/Monitor/D00141_copy.txt',
    'ShapeDatabase_INFOMR-master/Features/Tree/D00221_copy.txt'
]
col_labels = ['Aircraft Buoyant', 'Knife', 'Monitor', 'Tree']

# 4) Parsing utilities
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
        else:
            hist_data[key] = None
    return hist_data

def parse_metrics_file(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    in_metrics = False
    for line in lines:
        if line.strip().startswith('Metrics:'):
            in_metrics = True
            continue
        if in_metrics:
            if line.strip() == '':
                break
            if line.lower().startswith('extents'):
                m = re.search(r'\[([^\]]+)\]', line)
                if m:
                    vals = [float(x) for x in m.group(1).replace(',', ' ').split()]
                    metrics['extents'] = vals
                continue
            if ':' in line:
                k, v = line.split(':', 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return metrics

def derive_shape_metrics(m):
    ex = m.get('extents', None)
    elongation = np.nan
    flatness = np.nan
    if ex and len(ex) == 3:
        ex = np.asarray(ex, dtype=float)
        mi, ma = np.min(ex), np.max(ex)
        if mi > 0:
            elongation = float(ma / mi)   # larger -> more elongated
            flatness = float(mi / ma)     # smaller -> flatter
    return {'Elongation': elongation, 'Flatness': flatness}

def _normalize_columns(M):
    M = np.asarray(M, dtype=float)
    col_min = np.nanmin(M, axis=0)
    col_max = np.nanmax(M, axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    return (M - col_min) / denom

def _radar_factory(num_vars):
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    return angles

# 5) Plot helpers
def plot_histogram(ax, hist_values, color):
    if not hist_values:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('No data')
        return
    x = np.arange(len(hist_values))
    bars = ax.bar(x, hist_values, color=color, edgecolor='white', alpha=0.9, linewidth=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, max(hist_values), num=5, dtype=int))
    ax.set_ylabel('Count')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    for bar in bars:
        bar.set_linewidth(1.0)
        bar.set_edgecolor('white')

def save_fig(fig, name):
    path = OUT_DIR / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return path

# 6) High-level visualizations
def plot_all_histograms(file_paths, col_labels):
    all_files_data = [parse_hist_file(fp) for fp in file_paths]
    row_labels = ['A3', 'D1', 'D2', 'D3', 'D4']
    row_hist_keys = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']
    row_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    outputs = []
    for i, (hist_key, color) in enumerate(zip(row_hist_keys, row_colors)):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for j, ax in enumerate(axes):
            file_data = all_files_data[j]
            hist_values = file_data.get(hist_key)
            plot_histogram(ax, hist_values, color)
            ax.set_title(col_labels[j], fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        fig.suptitle(f'{row_labels[i]} shape-property distribution', fontsize=14)
        outputs.append(save_fig(fig, f'hist_{row_labels[i]}'))
    return outputs

def visualize_base_metrics(file_paths, labels, colors=None):
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    raw_list = [parse_metrics_file(fp) for fp in file_paths]
    derived_list = [derive_shape_metrics(m) for m in raw_list]

    core_keys = ['Sphericity', 'Compactness', 'Rectangularity', 'Convexity', 'Eccentricity']
    derived_keys = ['Elongation', 'Flatness']
    all_shape_keys = core_keys + derived_keys

    def get_val(m, key, fallback=np.nan):
        return m.get(key, fallback)

    shape_mat = []
    for m, d in zip(raw_list, derived_list):
        row = [get_val(m, k) for k in core_keys] + [get_val(d, k) for k in derived_keys]
        shape_mat.append(row)
    shape_mat = np.array(shape_mat, dtype=float)

    diam = [get_val(m, 'Diameter') for m in raw_list]
    norm_shape_mat = _normalize_columns(shape_mat)
    n_objs, n_feats = norm_shape_mat.shape
    x = np.arange(n_feats)
    width = 0.8 / n_objs

    outputs = []

    # Grouped bars
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    for i in range(n_objs):
        ax1.bar(x + (i - (n_objs - 1) / 2) * width, norm_shape_mat[i], width=width,
                color=colors[i % len(colors)], alpha=0.85, label=labels[i], edgecolor='white', linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_shape_keys, rotation=30, ha='right')
    ax1.set_ylabel('Normalized value')
    ax1.set_title('Shape descriptors (normalized)')
    ax1.legend(frameon=False, ncol=min(4, n_objs))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    outputs.append(save_fig(fig1, 'metrics_bars'))

    # Scatter comparisons
    pairs = [
        ('Sphericity', 'Compactness'),
        ('Rectangularity', 'Eccentricity'),
        ('Diameter', 'Compactness'),
    ]
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (xk, yk) in zip(axes, pairs):
        for i, (m, lbl) in enumerate(zip(raw_list, labels)):
            xv = get_val(m, xk)
            yv = get_val(m, yk)
            ax.scatter(xv, yv, s=80, color=colors[i % len(colors)], label=lbl, edgecolors='white', linewidths=1.0)
            ax.annotate(lbl, (xv, yv), textcoords='offset points', xytext=(5, 5), fontsize=9)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(f'{xk} vs. {yk}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig2.legend(handles, leg_labels, loc='upper center', ncol=min(4, n_objs), frameon=False)
    outputs.append(save_fig(fig2, 'metrics_scatter'))

    # Radar chart
    categories = all_shape_keys
    angles = _radar_factory(len(categories))
    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i in range(n_objs):
        vals = norm_shape_mat[i].tolist()
        vals += vals[:1]
        ax3.plot(angles, vals, color=colors[i % len(colors)], linewidth=2, label=labels[i])
        ax3.fill(angles, vals, color=colors[i % len(colors)], alpha=0.15)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_yticklabels([])
    ax3.set_title('Radar: normalized shape descriptors')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=False)
    outputs.append(save_fig(fig3, 'metrics_radar'))

    # Diameter bars
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.bar(labels, diam, color=colors[:n_objs], edgecolor='white', linewidth=1.0, alpha=0.9)
    ax4.set_ylabel('Diameter')
    ax4.set_title('Overall size indicator (Diameter)')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.3)
    outputs.append(save_fig(fig4, 'metrics_diameter'))

    return outputs

# 7) Orchestrate
def main():
    # Histograms per descriptor row
    plot_all_histograms(file_paths, col_labels)
    # Base metrics visuals
    visualize_base_metrics(file_paths, col_labels)

    # Show if GUI backend; otherwise notify save location
    if BACKEND.lower().endswith('agg'):
        print(f'Non-GUI backend in use. Figures saved to: {OUT_DIR.resolve()}')
    else:
        plt.show()

if __name__ == '__main__':
    main()

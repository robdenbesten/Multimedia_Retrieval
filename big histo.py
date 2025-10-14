import os
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_histograms(txt_path):
    wanted = {'D1', 'D2', 'A3', 'D3', 'D4'}
    out = {}
    current = None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if line.endswith('_hist:'):
                    key = line.replace('_hist:', '')
                    current = key if key in wanted else None
                    continue
                if current and line:
                    parts = [p for p in line.replace(';', ',').split(',') if p.strip() != '']
                    if len(parts) == 1 and ' ' in line:
                        parts = [p for p in line.split() if p.strip() != '']
                    vals = []
                    for p in parts:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            pass
                    if vals:
                        out[current] = np.asarray(vals, dtype=float)
                    current = None
    except Exception:
        pass
    return out

def collect_category_data(base_dir):
    descriptors = ['D1', 'D2', 'A3', 'D3', 'D4']
    data = {d: defaultdict(list) for d in descriptors}

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f'Base dir not found: {base_dir}')

    for cat in sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]):
        cat_dir = os.path.join(base_dir, cat)
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.lower().endswith('.txt'):
                    continue
                fpath = os.path.join(root, file)
                hists = parse_histograms(fpath)
                for d in descriptors:
                    if d in hists:
                        data[d][cat].append(hists[d])
    return data

def plot_descriptor_grid(descriptor, cat_to_arrays, out_dir, linewidth=0.9):
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

def main():
    base_dir = os.path.join('ShapeDatabase_INFOMR-master', 'features_test')
    out_dir = 'plots'  # write figures to `plots`

    data = collect_category_data(base_dir)
    for descriptor, cat_to_arrays in data.items():
        plot_descriptor_grid(descriptor, cat_to_arrays, out_dir)

if __name__ == '__main__':
    main()
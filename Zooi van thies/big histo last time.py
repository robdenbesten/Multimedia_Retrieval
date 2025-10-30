import os
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = '../Feature-matrix/all_features.csv'
OUT_DIR = '../plots'

HIST_PREFIXES = ['D1', 'D2', 'D3', 'D4', 'A3']


def normalize_relpath(p: str) -> str:
    if not p:
        return ''
    return p.replace('\\', '/').lstrip('./')


def to_float_or_nan(v):
    try:
        if v is None or v == '':
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def load_features_from_csv(csv_path: str) -> dict:
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        # Assume first column is 'object', second is 'category'
        if len(fieldnames) < 2:
            raise ValueError('CSV must have at least two columns: object and category.')

        hist_cols = {h: [] for h in HIST_PREFIXES}
        for fn in fieldnames:
            for h in HIST_PREFIXES:
                if fn.startswith(f'{h}_'):
                    hist_cols[h].append(fn)

        for h in HIST_PREFIXES:
            def _bin_idx(col):
                try:
                    return int(col.split('_')[-1])
                except (ValueError, IndexError):
                    return -1
            hist_cols[h].sort(key=_bin_idx)

        excluded = {fieldnames[0], fieldnames[1]}  # Exclude object and category columns
        for cols in hist_cols.values():
            excluded.update(cols)

        extent_keys = [k for k in ['extents_0', 'extents_1', 'extents_2'] if k in fieldnames]
        excluded.update(extent_keys)

        all_features = {}
        for row in reader:
            obj = row.get(fieldnames[0], '').strip()
            category = row.get(fieldnames[1], '').strip()
            if not obj or not category:
                continue
            rel_path = os.path.join(category, obj).replace('\\', '/')

            histograms = {}
            for h, cols in hist_cols.items():
                if not cols:
                    continue
                vals = np.array([to_float_or_nan(row.get(c)) for c in cols], dtype=float)
                vals = np.nan_to_num(vals, nan=0.0)
                s = float(vals.sum())
                if s > 0:
                    vals /= s
                histograms[h] = vals

            metrics = {}
            for fn in fieldnames:
                if fn in excluded:
                    continue
                val = to_float_or_nan(row.get(fn))
                if not np.isnan(val):
                    metrics[fn] = float(val)

            if extent_keys:
                ext = np.array([to_float_or_nan(row.get(k)) for k in extent_keys], dtype=float)
                metrics['extents'] = np.nan_to_num(ext)

            all_features[rel_path] = {
                'category': category,
                'metrics': metrics,
                'histograms': histograms,
            }

    return all_features


def collect_category_data(all_features: dict):
    desc_to_cat_arrays = {h: defaultdict(list) for h in HIST_PREFIXES}
    for feat in all_features.values():
        cat = feat.get('category', 'Unknown')
        hists = feat.get('histograms', {})
        for h, arr in hists.items():
            if h in desc_to_cat_arrays and arr.size > 0:
                desc_to_cat_arrays[h][cat].append(arr)
    return {d: c for d, c in desc_to_cat_arrays.items() if any(c.values())}


def plot_descriptor_grid(descriptor, cat_to_arrays, out_dir, linewidth=1.5, alpha=0.9):
    cats = sorted(cat_to_arrays.keys())
    if not cats:
        return

    n = len(cats)
    cols = 6
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)

    for i, cat in enumerate(cats):
        r, c = divmod(i, cols)
        ax = axes[r][c]

        hist_list = cat_to_arrays.get(cat, [])
        if not hist_list:
            ax.axis('off')
            continue

        num_items = len(hist_list)
        cmap = plt.get_cmap('Set1')
        base_colors = [cmap(j % cmap.N) for j in range(num_items)]

        max_y = 0
        for j, hist_values in enumerate(hist_list):
            x = np.arange(hist_values.size)
            ax.plot(x, hist_values, lw=linewidth, color=base_colors[j], alpha=alpha)
            if hist_values.size > 0:
                max_y = max(max_y, np.max(hist_values))

        ax.set_title(f"{cat} ({num_items} items)", fontsize=9)
        ax.set_ylim(0, max(1e-6, max_y * 1.1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')

    fig.suptitle(f'{descriptor} Histograms by Category', fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'LastVersion{descriptor}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    all_features = load_features_from_csv(CSV_PATH)

    desc_to_cats = collect_category_data(all_features)
    os.makedirs(OUT_DIR, exist_ok=True)

    for descriptor, cat_to_arrays in desc_to_cats.items():
        if not cat_to_arrays:
            continue
        print(f"Plotting descriptor: {descriptor}")
        plot_descriptor_grid(descriptor, cat_to_arrays, OUT_DIR)


if __name__ == '__main__':
    main()
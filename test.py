# python
import csv
import json
import os
from collections import defaultdict

def summarize_weightmap(wm_path: str):
    if not os.path.exists(wm_path):
        print(f"No weightmap found at {wm_path}")
        return
    with open(wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    print(f"Distance weightmap summary: {wm_path}")
    for metric, groups in wm.items():
        vals = list(groups.values())
        if not vals:
            continue
        vmin, vmax = min(vals), max(vals)
        ratio = (vmax / vmin) if vmin > 0 else float("inf")
        print(f"- {metric}: min={vmin:.4g}, max={vmax:.4g}, max/min={ratio:.3f}, groups={len(groups)}")

def category_from_label(label: str) -> str:
    return label.split('/', 1)[0] if '/' in label else label

def top1_and_map_for_group(rows):
    # rows are dicts with keys: query_label, query_category, retrieved_cat_rank_*
    total = len(rows)
    if total == 0:
        return 0.0, 0.0

    top1_hits = 0
    ap_sum = 0.0

    for row in rows:
        qcat = row['query_category']
        # collect retrieved cats in rank order
        rk = []
        i = 1
        while True:
            key = f"retrieved_cat_rank_{i}"
            if key not in row or row[key] == '':
                break
            rk.append(row[key])
            i += 1
        if not rk:
            continue

        # Top-1
        if rk[0] == qcat:
            top1_hits += 1

        # AP over the available retrieved list
        # Compute number of relevant items N for this query in the file
        # (we approximate by counting same-category queries minus 1)
        # Note: exact N per query would need class counts precomputed; use a safe fallback
        N = sum(1 for r in rows if r['query_category'] == qcat) - 1
        N = max(N, 1)

        precs = []
        correct = 0
        for i, cat in enumerate(rk, start=1):
            if cat == qcat:
                correct += 1
                precs.append(correct / i)
        ap = (sum(precs) / N) if N > 0 else 0.0
        ap_sum += ap

    top1 = top1_hits / total if total > 0 else 0.0
    mAP = ap_sum / total if total > 0 else 0.0
    return top1, mAP

def summarize_results(csv_path: str, weights_col: str = 'weights_map'):
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'metric' not in reader.fieldnames or weights_col not in reader.fieldnames:
            print(f"Required columns missing in {csv_path}. Need 'metric' and '{weights_col}'.")
            return

        # group rows by (weights_map, metric)
        groups = defaultdict(list)
        for row in reader:
            key = (row[weights_col], row['metric'])
            groups[key].append(row)

    print(f"\nRanking-sensitive summary from {csv_path}")
    for (wmap, metric), rows in sorted(groups.items()):
        top1, mAP = top1_and_map_for_group(rows)
        print(f"- weights={wmap:>9} | metric={metric:<24} | Top-1={top1:.4f} | mAP={mAP:.4f}")

if __name__ == "__main__":
    # Adjust paths as needed
    RESULTS = "raw_results_all.csv"
    WEIGHTMAP_JSON = os.path.join(os.path.dirname("Feature-matrix/all_features.csv") or ".", "distance_weightmap.json")

    summarize_weightmap(WEIGHTMAP_JSON)
    summarize_results(RESULTS, weights_col='weights_map')
import os
import numpy as np

def parse_metrics_file(file_path):

    metrics = {}
    with open(file_path, 'r', newline='') as f:
        for line in f:
            if ':' not in line:
                continue
            head = line.split(':', 1)[0].strip()
            if head in {'A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist'}:
                continue
            key, val = line.split(':', 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except (ValueError, TypeError):
                continue
    return metrics


def compute_metrics_statestiek(metrics_list):

    all_keys = set().union(*(m.keys() for _, m in metrics_list))
    statestiek = {}
    for key in all_keys:
        values = [m[key] for _, m in metrics_list if key in m and np.isfinite(m[key])]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values)) if np.std(values) != 0 else 1.0
        statestiek[key] = (mean, std)
    return statestiek


def normalize_metrics(metrics, statestiek):

    normalized_version = {}
    for key, value in metrics.items():
        if key in statestiek:
            mean, std = statestiek[key]
            normalized_version[key] = (value - mean) / std if std != 0 else 0.0
    return normalized_version


def normalize_database(src_root='Features',
                       out_root='normalized_single_features'):

    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)

    ### 1. Find all metric files
    files = []
    for root, _, filenames in os.walk(src_root):
        if os.path.commonpath([os.path.abspath(root), out_root]) == out_root:
            continue
        for fn in filenames:
            if fn.lower().endswith('.txt'):
                in_path = os.path.join(root, fn)
                rel_dir = os.path.relpath(root, src_root)
                out_path = os.path.join(out_root, rel_dir, fn)
                files.append((in_path, out_path))

    if not files:
        print("No metric files found to process.")
        return

    print(f"Found {len(files)} metric files. Output folder: {out_root}")

    ### 2. Parse all files
    metrics_list = []
    for in_path, _ in files:
        try:
            metrics = parse_metrics_file(in_path)
            metrics_list.append((in_path, metrics))
        except Exception as e:
            print(f"Error parsing {in_path}: {e}")

    ### 3. Compute the mean and std for all metrics
    statestiek = compute_metrics_statestiek(metrics_list)


    ### 4. Normalize and save each file
    processed = 0
    skipped = 0
    for i, (in_path, out_path) in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Normalizing {in_path}")
        try:
            metrics = parse_metrics_file(in_path)
            normalized = normalize_metrics(metrics, statestiek)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Write normalized metrics file
            with open(out_path, 'w') as f:
                for key, val in normalized.items():
                    f.write(f"{key}: {val:.6f}\n")

            processed += 1
        except Exception as e:
            print(f"Failed to normalize {in_path}: {e}")
            skipped += 1

    print(f"\n✅ Done. Normalized: {processed}, Skipped: {skipped}")
    print(f"Normalized files saved to: {out_root}")


if __name__ == '__main__':
    normalize_database()

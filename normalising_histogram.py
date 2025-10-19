import os
import numpy as np
import re


def parse_hist_file(file_path):
    with open(file_path, 'r', newline='') as f:
        content = f.read()

    histograms = {}
    pattern = r'([A-Z]\d_hist)\s*:\s*([\s\S]*?)(?=\n[A-Z]\d_hist\s*:|\Z)'
    matches = re.findall(pattern, content, flags=re.MULTILINE)

    for key, block in matches:
        tokens = re.split(r'[,\s;]+', block.strip())
        values = []
        for t in tokens:
            if not t:
                continue
            try:
                values.append(float(t))
            except ValueError:
                continue
        histograms[key] = values

    return histograms


def normalize_histograms(hist_dict):
    normalized_version = {}
    for key, values in hist_dict.items():
        total = sum(values)
        if total > 0:
            normalized_version[key] = [v / total for v in values]
        else:
            normalized_version[key] = values
    return normalized_version


def normalize_database(src_root='Features', out_root='normalized_histograms'):
    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)

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

    processed = 0
    skipped = 0
    for i, (in_path, out_path) in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Normalizing {in_path}")
        try:
            histograms = parse_hist_file(in_path)
            normalized_version = normalize_histograms(histograms)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, 'w') as f:
                for key, val in normalized_version.items():
                    f.write(f"{key}:\n")
                    f.write(",".join(f"{v:.6f}" for v in val) + "\n\n")

            processed += 1
        except Exception as e:
            print(f"Failed to normalize {in_path}: {e}")
            skipped += 1

    print(f"\n✅ Done. Normalized: {processed}, Skipped: {skipped}")
    print(f"Normalized files saved to: {out_root}")


if __name__ == '__main__':
    normalize_database()

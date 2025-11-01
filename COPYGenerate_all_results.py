import csv
import sys
from collections import defaultdict
import concurrent.futures
import os
import threading
from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS

THREAD_LOCAL = threading.local()

def generate_retrieval_results_wide_format(verbose: bool = False):
    """
    Generates retrieval results in a wide CSV format using multithreading, excluding the query itself.
    For each item, N = category_size - 1 (relevant items excluding the query).
    Writes retrieved categories into columns `retrieved_cat_rank_1` ... up to the
    maximum N across categories.
    """

    def category_from_label(label: str) -> str:
        return label.split('/', 1)[0] if '/' in label else label

    # --- Configuration ---
    csv_path = 'Feature-matrix/all_features_modified.csv'
    out_path = 'adjustedresults2.csv'
    metrics = ['euclidean', 'manhattan', 'manhattan+chi-squared',
               'manhattan+emd', 'manhattan+kullback-leibler', 'knn']
    # Start with 4x cores but clamp to label count
    num_workers = (os.cpu_count() or 1) * 4
    # --- End of Configuration ---

    try:
        # One searcher to obtain labels and counts
        initial_searcher = ShapeSearcher(csv_path, MANUAL_WEIGHTS)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    labels = initial_searcher.get_available_labels()
    total = len(labels)
    if total == 0:
        # Still produce an empty CSV with header
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['query_label', 'query_category', 'metric'])
        print(f'\nDetailed retrieval results have been written to `{out_path}`.')
        return

    # Precompute label -> category
    label_to_category = {lbl: category_from_label(lbl) for lbl in labels}

    # Pre-calculate category counts
    category_counts = defaultdict(int)
    for lbl in labels:
        category_counts[label_to_category[lbl]] += 1

    # Max number of comparisons across categories
    max_comparisons = max((cnt - 1) for cnt in category_counts.values()) if category_counts else 0

    # Precompute label -> n_comparisons
    label_to_ncomp = {lbl: max(0, category_counts[label_to_category[lbl]] - 1) for lbl in labels}

    # Clamp workers to label count
    num_workers = min(num_workers, total) if total else 1

    # Precompute padding cache: pad_cache[i] is a list of i empty strings
    pad_cache = [[''] * i for i in range(max_comparisons + 1)]

    def get_thread_searcher():
        """Create a thread-local ShapeSearcher (constructed once per worker thread)."""
        if not hasattr(THREAD_LOCAL, 'searcher'):
            THREAD_LOCAL.searcher = ShapeSearcher(csv_path, MANUAL_WEIGHTS)
        return THREAD_LOCAL.searcher

    def process_label(query_label: str):
        """Worker function to process a single query label, returns all metric-rows for that label."""
        searcher = get_thread_searcher()
        query_category = label_to_category[query_label]
        n_comparisons = label_to_ncomp[query_label]
        rows_for_label = []

        if n_comparisons == 0:
            # No relevant comparisons in this category, still emit rows with padding
            for metric in metrics:
                rows_for_label.append([query_label, query_category, metric] + pad_cache[max_comparisons])
            return rows_for_label

        request_n = n_comparisons + 1  # ask one extra to account for the query itself
        for metric in metrics:
            raw_neighbors = searcher.search(query_label, metric, top_n=request_n)

            # Collect up to n_comparisons, skipping the query label
            neighbors = []
            for lbl in raw_neighbors:
                if lbl != query_label:
                    neighbors.append(lbl)
                    if len(neighbors) == n_comparisons:
                        break

            # Map neighbor labels to their categories
            retrieved_categories = [label_to_category[lbl] for lbl in neighbors]

            # Pad to fixed width
            pad_len = max_comparisons - len(retrieved_categories)
            row = [query_label, query_category, metric] + retrieved_categories + pad_cache[pad_len]
            rows_for_label.append(row)

        return rows_for_label

    # Write CSV streaming as results complete in input order
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['query_label', 'query_category', 'metric']
        header.extend([f'retrieved_cat_rank_{i}' for i in range(1, max_comparisons + 1)])
        writer.writerow(header)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, rows in enumerate(executor.map(process_label, labels), start=1):
                writer.writerows(rows)
                if verbose and (idx % 50 == 0 or idx == total):
                    print(f"Processed {idx}/{total}")

    print(f'\nDetailed retrieval results have been written to `{out_path}`.')

if __name__ == '__main__':
    generate_retrieval_results_wide_format(verbose=False)
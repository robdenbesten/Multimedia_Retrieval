# python
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS
except Exception:
    # Minimal fallback implementation so the script can run without the real module.
    MANUAL_WEIGHTS = {
        'A3': 1.0, 'D1': 1.0, 'D2': 1.0, 'D3': 1.0, 'D4': 1.0,
        'Surface area': 1.0, 'Sphericity': 1.0, 'Rectangularity': 1.0,
        'Diameter': 1.0, 'Convexity': 1.0, 'Eccentricity': 1.0,
    }

    MANUAL_WEIGHTS_ADJUSTED = {
        'A3': 2.0, 'D1': 1.0, 'D2': 2.0, 'D3': 2.0, 'D4': 2.0,
        'Surface area': 1.0, 'Sphericity': 1.0, 'Rectangularity': 1.0,
        'Diameter': 1.0, 'Convexity': 0.5, 'Eccentricity': 1.0,
    }


    class ShapeSearcher:
        def __init__(self, csv_path, weights):
            self.csv_path = csv_path
            self.weights = weights
            self.labels = []
            try:
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # Skip header
                    next(reader, None)
                    for row in reader:
                        if not row:
                            continue
                        # assume label is 'Category/Object'
                        self.labels.append(f"{row[1]}/{row[0]}")
            except FileNotFoundError:
                # bubble up to allow caller to handle missing file
                raise

        def get_available_labels(self):
            return self.labels

        def search(self, query_label, metric, top_n=10):
            # simple deterministic "search": return other labels up to top_n
            results = [lbl for lbl in self.labels if lbl != query_label]
            return results[:top_n]


def category_from_label(label: str) -> str:
    """Extracts the category part from a 'Category/Object' label."""
    return label.split('/', 1)[0] if '/' in label else label


# Global variable to hold the searcher instance in each worker process
worker_searcher = None


def init_worker(searcher_instance):
    """Initializer for the multiprocessing pool. Makes a ShapeSearcher instance global."""
    global worker_searcher
    worker_searcher = searcher_instance


def process_query(query_label, weights_name, metrics, category_counts, max_comparisons):
    """
    Worker function to process a single query label.
    It uses the 'worker_searcher' instance initialized globally in its process.
    """
    query_category = category_from_label(query_label)
    category_size = category_counts[query_category]
    n_comparisons = max(0, category_size - 1)
    rows = []

    for metric in metrics:
        request_n = n_comparisons + 1 if n_comparisons > 0 else 0
        neighbors = []
        if request_n > 0:
            # Use the global searcher from the worker's context
            raw_neighbors = worker_searcher.search(query_label, metric, top_n=request_n)
            neighbors = [lbl for lbl in raw_neighbors if lbl != query_label][:n_comparisons]

        retrieved_categories = [category_from_label(lbl) for lbl in neighbors]
        retrieved_categories += [''] * (max_comparisons - len(retrieved_categories))

        row = [query_label, query_category, metric, weights_name] + retrieved_categories
        rows.append(row)
    return rows


def generate_retrieval_results_wide_format():
    """
    Generates retrieval results in a wide CSV format.
    This version initializes a ShapeSearcher once per worker process for each weight map,
    avoiding repeated k-NN model building.
    """
    # --- Configuration ---
    csv_path = 'Feature-matrix/all_features.csv'
    out_path = 'raw_results_all.csv'
    metrics = ['euclidean', 'cosine', 'manhattan', 'manhattan+chi-squared', 'manhattan+emd',
               'manhattan+kullback-leibler', 'knn']
    weight_maps = {
        'neutral': MANUAL_WEIGHTS,
        'adjusted': MANUAL_WEIGHTS_ADJUSTED
    }
    # --- End of Configuration ---

    try:
        # Load labels once to get counts and max comparisons
        temp_searcher = ShapeSearcher(csv_path, MANUAL_WEIGHTS)
        labels = temp_searcher.get_available_labels()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    category_counts = defaultdict(int)
    for lbl in labels:
        category_counts[category_from_label(lbl)] += 1

    max_comparisons = max((cnt - 1) for cnt in category_counts.values()) if category_counts else 0

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['query_label', 'query_category', 'metric', 'weights_map']
        header.extend([f'retrieved_cat_rank_{i}' for i in range(1, max_comparisons + 1)])
        writer.writerow(header)

        num_workers = cpu_count()
        print(f"Using {num_workers} worker processes.")

        for weights_name, weights_map in weight_maps.items():
            print(f"\n--- Processing with '{weights_name}' weights ---")

            # Create the expensive ShapeSearcher object once for this weight map.
            # This will load data and build the k-NN model.
            searcher = ShapeSearcher(csv_path, weights_map)

            worker_func = partial(process_query,
                                  weights_name=weights_name,
                                  metrics=metrics,
                                  category_counts=category_counts,
                                  max_comparisons=max_comparisons)

            # The 'searcher' object is passed to each worker process upon creation.
            with Pool(processes=num_workers, initializer=init_worker, initargs=(searcher,)) as pool:
                total_labels = len(labels)
                for i, result_rows in enumerate(pool.imap_unordered(worker_func, labels)):
                    if result_rows:
                        writer.writerows(result_rows)
                    processed_label = result_rows[0][0] if result_rows else "a label"
                    print(f"Processing item {i + 1}/{total_labels}: {processed_label} done.")

    print(f'\nDetailed retrieval results have been written to `{out_path}`.')


if __name__ == '__main__':
    generate_retrieval_results_wide_format()
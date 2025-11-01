# python
import csv
import sys
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    # Import the primary search algorithm and its constants
    from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS, FEATURE_GROUP_ORDER
except ImportError:
    print("Error: Could not import from 'Comparison_algorithm.py'.", file=sys.stderr)
    print("Please ensure the file exists and is in the correct path.", file=sys.stderr)
    # Define fallbacks for basic script functionality if the import fails
    FEATURE_GROUP_ORDER = [
        'A3', 'D1', 'D2', 'D3', 'D4', 'Surface area', 'Sphericity',
        'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity'
    ]
    MANUAL_WEIGHTS = {key: 1.0 for key in FEATURE_GROUP_ORDER}


    class ShapeSearcher:
        """Fallback ShapeSearcher to allow the script to run with limited functionality."""

        def __init__(self, feature_csv_path, **kwargs):
            print("Warning: Using fallback ShapeSearcher. No real search will be performed.", file=sys.stderr)
            self.labels = []
            try:
                with open(feature_csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if row: self.labels.append(f"{row[1]}/{row[0]}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Fallback ShapeSearcher could not find '{feature_csv_path}'")

        def get_available_labels(self):
            return self.labels

        def search(self, query_label, metric, top_n=10):
            print(f"Fallback search called for {query_label} with metric {metric}", file=sys.stderr)
            return [lbl for lbl in self.labels if lbl != query_label][:top_n]


def category_from_label(label: str) -> str:
    """Extracts the category part from a 'Category/Object' label."""
    return label.split('/', 1)[0] if '/' in label else label


# Per-process globals (constructed lazily to be fork-safe)
_worker_cfg = None
_worker_searcher = None


def init_worker(cfg):
    """Initializes each worker process with the run configuration."""
    global _worker_cfg, _worker_searcher
    _worker_cfg = cfg
    _worker_searcher = None  # Lazily constructed on first use in the process


def _get_worker_searcher():
    """Gets or creates the ShapeSearcher instance for the current worker process."""
    global _worker_searcher, _worker_cfg
    if _worker_searcher is None:
        print(
            f"Worker (PID {os.getpid()}) initializing ShapeSearcher with method '{_worker_cfg['weighting_method']}'...")
        _worker_searcher = ShapeSearcher(
            feature_csv_path=_worker_cfg['csv_path'],
            weights=_worker_cfg['weights'],
            weighting_method=_worker_cfg['weighting_method'],
            distance_weightmap_path=_worker_cfg.get('wm_path'),
            build_distance_map=False  # Workers should never build the map
        )
    return _worker_searcher


def process_query(query_label, run_name, metrics, category_counts, max_comparisons):
    """Processes a single query label against all specified metrics for a given run."""
    searcher = _get_worker_searcher()
    query_category = category_from_label(query_label)
    category_size = category_counts.get(query_category, 0)
    n_comparisons = max(0, category_size - 1)
    rows = []

    for metric in metrics:
        # Request N+1 neighbors to account for the query item itself, then filter
        request_n = n_comparisons + 1 if n_comparisons > 0 else 0
        neighbors = []
        if request_n > 0:
            raw_neighbors = searcher.search(query_label, metric, top_n=request_n)
            # Ensure the query item is not in its own results and cap the count
            neighbors = [lbl for lbl in raw_neighbors if lbl != query_label][:n_comparisons]

        retrieved_categories = [category_from_label(lbl) for lbl in neighbors]
        # Pad the results to ensure all rows have the same number of columns
        retrieved_categories += [''] * (max_comparisons - len(retrieved_categories))
        rows.append([query_label, query_category, metric, run_name] + retrieved_categories)
    return rows


def generate_retrieval_results_wide_format():
    # --- Configuration ---
    csv_path = 'Feature-matrix/all_features_modified.csv'
    out_path = 'raw_results_all2.csv'
    wm_path = os.path.join(os.path.dirname(csv_path) or ".", "distance_weightmap.json")

    metrics = [
        'euclidean', 'euclidean_flat', 'manhattan', 'manhattan+chi-squared',
        'manhattan+emd', 'manhattan+kullback-leibler', 'knn'
    ]

    # Define the different experiment runs. Each tuple contains:
    # (run_name, weighting_method, weights_dictionary)
    runs = [
        # 1. Unweighted: All features contribute equally. Uses weighting_method='none'.
        ('unweighted', 'none', {}),

        # 2. Feature Weighted: Uses manually adjusted weights from MANUAL_WEIGHTS.
        ('adjusted', 'feature', MANUAL_WEIGHTS),

        # 3. Distance Weighted: Normalizes by the standard deviation of distances.
        #    The weights dictionary is ignored but required by the function signature.
        ('distance', 'distance', {}),
    ]
    # --- End of Configuration ---

    try:
        # Get all available labels by initializing a lightweight searcher
        labels = ShapeSearcher(csv_path, weights={}, weighting_method='none').get_available_labels()
    except (FileNotFoundError, NameError) as e:
        print(f"Fatal Error: Could not load labels. {e}", file=sys.stderr)
        sys.exit(1)

    # Build the distance weightmap once in the main process if needed for any 'distance' run
    if any(run[1] == 'distance' for run in runs):
        if not os.path.exists(wm_path):
            print(f"Building distance weightmap at `{wm_path}` (one-time operation)...")
            ShapeSearcher(
                feature_csv_path=csv_path,
                weights={},  # Weights are not used for building the map
                weighting_method='distance',
                distance_weightmap_path=wm_path,
                build_distance_map=True
            )
        else:
            print(f"Using existing distance weightmap at `{wm_path}`.")

    category_counts = defaultdict(int)
    for lbl in labels:
        category_counts[category_from_label(lbl)] += 1
    max_comparisons = max((cnt - 1 for cnt in category_counts.values()), default=0)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['query_label', 'query_category', 'metric', 'weights_map']
        header.extend([f'retrieved_cat_rank_{i}' for i in range(1, max_comparisons + 1)])
        writer.writerow(header)

        num_workers = min(cpu_count(), 16)  # Cap workers to avoid excessive overhead
        print(f"Using {num_workers} worker processes.")

        for run_name, weighting_method, weights_map in runs:
            print(f"\n--- Processing run `{run_name}` (mode: {weighting_method}) ---")
            # Create the configuration dictionary for the worker processes
            cfg = {
                'csv_path': csv_path,
                'weights': weights_map,
                'weighting_method': weighting_method,
                'wm_path': wm_path,
            }

            # Use functools.partial to pre-fill arguments for the worker function
            worker_func = partial(
                process_query,
                run_name=run_name,
                metrics=metrics,
                category_counts=category_counts,
                max_comparisons=max_comparisons
            )

            with Pool(processes=num_workers, initializer=init_worker, initargs=(cfg,)) as pool:
                total_labels = len(labels)
                # Use imap_unordered for efficient parallel processing
                for i, result_rows in enumerate(pool.imap_unordered(worker_func, labels)):
                    if result_rows:
                        writer.writerows(result_rows)
                    # Print progress periodically
                    if (i + 1) % 100 == 0 or (i + 1) == total_labels:
                        print(f"  Processed {i + 1}/{total_labels} queries...")

    print(f"\nDetailed retrieval results have been written to `{out_path}`.")


if __name__ == '__main__':
    generate_retrieval_results_wide_format()
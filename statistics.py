import os
import sys
import json
import time
import importlib.util
from multiprocessing import Pool, cpu_count
from functools import partial

# --- Module and Path Handling ---

def _find_comparison_algo_file(root_dir):
    """Finds the comparison algorithm file (case-insensitive, tolerant for spaces)."""
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            name = f.lower().replace("%20", "").replace(" ", "")
            if "first" in name and "comarison" in name and "algo" in name and name.endswith(".py"):
                return os.path.join(subdir, f)
    raise FileNotFoundError("Could not locate the comparison algorithm file (e.g., 'First comarison algo.py').")

def _import_module_from_path(mod_name, file_path):
    """Imports a module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for '{file_path}'")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

# --- Helper Functions ---

def _category_from_label(label: str) -> str:
    """Extracts the category from a file label."""
    return label.replace("\\", "/").split("/")[0]

def _default_group_weights() -> dict:
    """Provides default weights for feature groups."""
    return {
        'A3': 3.5,
        'D1': 2.0,
        'D2': 3.5,
        'D3': 3.0,
        'D4': 1.0,
        'Surface area': 1.0,
        'Sphericity': 2.0,
        'Rectangularity': 1.5,
        'Diameter': 1.0,
        'Convexity': 2.0,
        'Eccentricity': 1.5,
    }

# --- Worker Process Logic ---

# Global engine instance for each worker process
engine_instance = None

def init_worker(algo_path, feature_json, obj_root_dir, group_weights):
    """Initializes the ShapeSearchEngine for each worker process."""
    global engine_instance
    print(f"Initializing engine in worker PID: {os.getpid()}...")
    algo_mod = _import_module_from_path("comparison_algo", algo_path)
    engine_instance = algo_mod.ShapeSearchEngine(feature_json, obj_root_dir, group_weights)

def evaluate_chunk(query_labels_chunk: list, metric: str, top_n: int) -> tuple[int, int]:
    """Worker function to evaluate a chunk of query labels."""
    global engine_instance
    if engine_instance is None:
        raise RuntimeError("Search engine not initialized in worker process.")

    correct = 0
    total = 0
    query_method = getattr(engine_instance, 'search_vectorized', engine_instance.search)

    for query_label in query_labels_chunk:
        try:
            results = query_method(query_label, top_n=top_n, metric=metric)
            query_cat = _category_from_label(query_label)
            for r in results:
                total += 1
                if _category_from_label(r) == query_cat:
                    correct += 1
        except Exception as e:
            print(f"Warning: Could not run search for '{query_label}' with metric '{metric}': {e}", file=sys.stderr)
    return correct, total

# --- Main Execution Logic ---

def main():
    """Main function to configure and run the evaluation."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_dataset_root = os.path.join(project_root, "ShapeDatabase_INFOMR-master")

    top_n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    dataset_root = sys.argv[2] if len(sys.argv) > 2 else default_dataset_root

    # --- 1. Gather Configuration ---
    algo_path = _find_comparison_algo_file(project_root)
    feature_json = os.path.join(dataset_root, "features.json")
    obj_root_dir = os.path.join(dataset_root, "normalized_5000")

    if not os.path.isfile(feature_json):
        raise FileNotFoundError(f"Feature JSON not found at '{feature_json}'")

    with open(feature_json, 'r') as f:
        all_labels = list(json.load(f).keys())

    group_weights = _default_group_weights()
    metrics_to_test = sorted(['euclidean', 'manhattan', 'cosine', 'emd', 'chi-squared', 'kullback-leibler', 'cross-bin'])
    num_workers = cpu_count()

    print(f"Evaluating {len(metrics_to_test)} metrics with top_n={top_n} using {num_workers} workers...")
    print("-" * 50)

    # --- 2. Evaluate All Metrics ---
    results = {}
    init_args = (algo_path, feature_json, obj_root_dir, group_weights)

    for metric in metrics_to_test:
        metric_start_time = time.time()
        print(f"Evaluating metric: {metric}...")

        chunk_size = max(1, len(all_labels) // (num_workers * 2))
        chunks = [all_labels[i:i + chunk_size] for i in range(0, len(all_labels), chunk_size)]
        worker_func = partial(evaluate_chunk, metric=metric, top_n=top_n)

        total_correct = 0
        total_searched = 0

        with Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            # map the worker function to the chunks and collect scores
            for correct, total in pool.map(worker_func, chunks):
                total_correct += correct
                total_searched += total

        percent = (total_correct / total_searched * 100.0) if total_searched > 0 else 0.0
        results[metric] = (total_correct, total_searched, percent)
        elapsed = time.time() - metric_start_time
        print(f"  -> Completed: {metric} in {elapsed:.2f}s. Accuracy: {percent:.2f}%")

    # --- 3. Print Summary ---
    print("\n--- Overall Correct Retrieval (per-result) ---")
    for metric_name in metrics_to_test:
        if metric_name in results:
            correct, total, percent = results[metric_name]
            print(f"Top-{top_n} [{metric_name:<18}]: {percent:>6.2f}%  ({correct}/{total})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)

import os
import sys
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed


def _find_comparison_algo_file(root_dir):
    """
    Finds the comparison algorithm file (case-insensitive, tolerant for spaces or %20).
    """
    candidates = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            name = f.lower()
            if ("first" in name and "comarison" in name and "algo" in name and name.endswith(".py")):
                candidates.append(os.path.join(subdir, f))
    if not candidates:
        raise FileNotFoundError("Could not locate the comparison algorithm file (e.g., 'First comarison algo.py').")
    candidates.sort(key=len)
    return candidates[0]


def _import_module_from_path(mod_name, file_path):
    """
    Imports a module from an arbitrary file path (works with spaces).
    """
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for '{file_path}'")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _category_from_label(label: str) -> str:
    """
    Category is the first path component of the label.
    Handles both '/' and '\\' separators.
    """
    parts = label.replace("\\", "/").split("/")
    return parts[0] if parts and parts[0] else "(root)"


def _default_group_weights() -> dict:
    """
    Mirrors the weights used by the UI script.
    """
    return {
        # Histograms
        'A3': 1.5,
        'D1': 2.5,
        'D2': 1.5,
        'D3': 1.0,
        'D4': 1.0,
        # Scalars
        'Surface area': 1.2,
        'Sphericity': 1.0,
        'Rectangularity': 1.0,
        'Diameter': 1.0,
        'Convexity': 0.4,
        'Eccentricity': 1.0,
    }


def evaluate_metric(algo_path: str, feature_json: str, obj_root_dir: str, group_weights: dict, top_n: int, metric: str):
    """
    Evaluates a single metric.
    Initializes its own engine instance to be process-safe.
    Returns (metric, correct, total, percent).
    """
    # Each process imports the module and creates its own engine
    algo_mod = _import_module_from_path("comparison_algo", algo_path)
    engine = algo_mod.ShapeSearchEngine(feature_json, obj_root_dir, group_weights)

    correct = 0
    total = 0

    for query_label in engine.labels:
        try:
            results = engine.search(query_label, top_n=top_n, metric=metric)
            query_cat = _category_from_label(query_label)
            for r in results:
                total += 1
                if _category_from_label(r) == query_cat:
                    correct += 1
        except Exception as e:
            # Use a process-safe way to report warnings
            print(f"Warning: Could not run search for '{query_label}' with metric '{metric}': {e}", file=sys.stderr)

    percent = (correct / total * 100.0) if total > 0 else 0.0
    return metric, correct, total, percent


if __name__ == "__main__":
    # Resolve dataset path relative to this file by default
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_dataset_root = os.path.join(project_root, "ShapeDatabase_INFOMR-master")

    # CLI: python statistics.py [top_n] [dataset_root]
    # Defaults: top_n=5, dataset_root=.../ShapeDatabase_INFOMR-master
    try:
        top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    except ValueError:
        print("Invalid top_n argument; using default 5.")
        top_n = 5
    dataset_root = sys.argv[2] if len(sys.argv) > 2 else default_dataset_root

    try:
        # --- 1. Gather Configuration ---
        algo_path = _find_comparison_algo_file(project_root)
        feature_json = os.path.join(dataset_root, "features.json")
        obj_root_dir = os.path.join(dataset_root, "normalized_5000")

        if not os.path.isfile(feature_json):
            raise FileNotFoundError(f"Feature JSON not found at '{feature_json}'")
        if not os.path.isdir(obj_root_dir):
            raise FileNotFoundError(f"OBJ root directory not found at '{obj_root_dir}'")

        group_weights = _default_group_weights()

        # The list of metrics is known and can be defined here
        metrics_to_test = sorted([
            'euclidean', 'manhattan', 'cosine', 'emd',
            'chi-squared', 'kullback-leibler', 'cross-bin'
        ])

        print(f"Evaluating {len(metrics_to_test)} metrics with top_n={top_n}...")

        # --- 2. Evaluate All Metrics in Parallel ---
        results = {}
        with ProcessPoolExecutor() as executor:
            # Submit tasks with config arguments instead of the engine object
            future_to_metric = {
                executor.submit(evaluate_metric, algo_path, feature_json, obj_root_dir, group_weights, top_n,
                                metric): metric
                for metric in metrics_to_test
            }

            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    metric_name, correct, total, percent = future.result()
                    results[metric_name] = (correct, total, percent)
                    print(f"  - Completed: {metric_name}")
                except Exception as e:
                    print(f"Error evaluating metric '{metric_name}': {e}", file=sys.stderr)

        # --- 3. Print Summary ---
        print("\n--- Overall Correct Retrieval (per-result) ---")
        # Sort results for consistent output
        for metric_name in metrics_to_test:
            if metric_name in results:
                correct, total, percent = results[metric_name]
                print(f"Top-{top_n} [{metric_name:<18}]: {percent:>6.2f}%  ({correct}/{total})")

    except Exception as e:
        print(f"An error occurred during setup: {e}", file=sys.stderr)
        sys.exit(1)
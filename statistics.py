
import os
import sys
import importlib.util

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

def _category_of(feature_dir, file_path):
    """
    Category is the first directory under the feature_dir.
    """
    rel = os.path.relpath(file_path, feature_dir)
    parts = rel.split(os.sep)
    return parts[0] if parts else "(root)"

def evaluate_overall_correct_rate(feature_dir, top_n=5):
    """
    Overall per-result accuracy across all queries:
    - Count every retrieved item that matches the query's category as correct.
    - Count mismatches as wrong.
    Returns (correct, total, percent).
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    algo_path = _find_comparison_algo_file(project_root)
    algo_mod = _import_module_from_path("comparison_algo", algo_path)

    engine = algo_mod.EnhancedShapeSearchEngine(feature_dir)

    correct = 0
    total = 0

    for query_file in engine.files:
        results = engine.search(query_file, top_n=top_n)
        query_cat = _category_of(feature_dir, query_file)
        for r in results:
            total += 1
            if _category_of(feature_dir, r) == query_cat:
                correct += 1

    percent = (correct / total * 100.0) if total > 0 else 0.0
    return correct, total, percent

if __name__ == "__main__":
    # Resolve dataset path relative to this file
    project_root = os.path.dirname(os.path.abspath(__file__))
    feature_dir = os.path.join(project_root, "ShapeDatabase_INFOMR-master", "features_test")
    if not os.path.isdir(feature_dir):
        print(f"Error: feature directory not found at '{feature_dir}'")
        sys.exit(1)

    # Optional CLI arg: top_n (default: 5)
    try:
        top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    except ValueError:
        print("Invalid top_n argument; using default 5.")
        top_n = 5

    correct, total, percent = evaluate_overall_correct_rate(feature_dir, top_n=top_n)
    print(f"Overall correct retrieval (per-result) in top-{top_n}: {percent:.2f}%  ({correct}/{total})")
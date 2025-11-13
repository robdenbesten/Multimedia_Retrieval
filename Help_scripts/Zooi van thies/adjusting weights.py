"""
WEIGHT OPTIMIZATION TOOL
This file helps find the best weights for different features.
It tests different weight combinations and measures performance.
It calculates average precision for query results.
It can process large datasets in parallel for faster testing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed  # <-- add ThreadPoolExecutor
import os

from Querying import ShapeSearcher, MANUAL_WEIGHTS

def calculate_average_precision(query_category: str, result_labels: List[str], total_relevant: int) -> float:
    """
    Calculates the Average Precision (AP) for a single query.
    """
    hits = 0
    precision_sum = 0.0

    if total_relevant <= 0:
        return 0.0

    for i, label in enumerate(result_labels):
        result_category = label.split('/')[0]
        if result_category == query_category:
            hits += 1
            precision_at_k = hits / (i + 1)
            precision_sum += precision_at_k

    return precision_sum / total_relevant


def _process_chunk(searcher: ShapeSearcher, metric: str,
                   chunk: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, float]]:
    """Worker function to compute distances for a chunk of pairs."""
    results = []
    for i, j, orig_i, orig_j in chunk:
        dist = searcher.calculate_distance(orig_i, orig_j, metric)
        results.append((i, j, dist))
    return results


def compute_distance_matrix_parallel(searcher: ShapeSearcher, metric: str, indices: np.ndarray,
                                     max_workers: int) -> np.ndarray:
    """
    Computes the distance matrix for a given subset of indices in parallel by chunking the work.
    """
    n_items = len(indices)
    distance_matrix = np.zeros((n_items, n_items), dtype=np.float32)

    # Create all pairs of indices that need to be computed
    pairs_to_compute = []
    for i in range(n_items):
        for j in range(i, n_items):
            pairs_to_compute.append((i, j, indices[i], indices[j]))

    # Divide pairs into chunks for processing
    num_chunks = min(len(pairs_to_compute), max_workers * 4)  # Heuristic for chunking
    if not pairs_to_compute:
        return distance_matrix
    chunk_size = (len(pairs_to_compute) + num_chunks - 1) // num_chunks
    chunks = [pairs_to_compute[i:i + chunk_size] for i in range(0, len(pairs_to_compute), chunk_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each chunk as a single task
        future_to_chunk = {executor.submit(_process_chunk, searcher, metric, chunk): chunk for chunk in chunks}

        num_pairs = len(pairs_to_compute)
        completed_pairs = 0
        for future in as_completed(future_to_chunk):
            try:
                results = future.result()
                for i, j, dist in results:
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # Symmetric matrix
                completed_pairs += len(results)
                print(f"Computing distance matrix: {completed_pairs}/{num_pairs} pairs", end='\r')
            except Exception as e:
                print(f"Error processing a chunk: {e}")

    print("\nDistance matrix computation complete.")
    return distance_matrix


def evaluate_performance(
        feature_csv_path: str,
        weights: Dict[str, float],
        weighting_method: str,
        metric: str,
        top_n: int = 10,
        sample_size_per_category: int = 10
) -> float:
    """
    Evaluates retrieval performance on a sampled subset of the data for speed.
    """
    print(f"\n--- Evaluating: [Weighting: {weighting_method}, Metric: {metric}] ---")

    # 1. Initialize searcher with the full dataset.
    searcher = ShapeSearcher(feature_csv_path, weights, weighting_method)

    # 2. Create a representative sample from the full dataset.
    print(f"Sampling up to {sample_size_per_category} items per category...")
    all_labels_df = pd.DataFrame({
        'original_index': range(len(searcher.labels)),
        'label': searcher.labels
    })
    all_labels_df['category'] = all_labels_df['label'].apply(lambda x: x.split('/')[0])

    # Robust per-category sampling that preserves the 'category' column
    sampled_parts = []
    for cat, grp in all_labels_df.groupby('category'):
        n = min(len(grp), sample_size_per_category)
        sampled_parts.append(grp.sample(n=n, random_state=42))

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    # Safety: ensure 'category' exists (in case of unexpected behavior)
    if 'category' not in sampled_df.columns:
        sampled_df['category'] = sampled_df['label'].str.split('/').str[0]

    sampled_indices = sampled_df['original_index'].to_numpy()
    sampled_labels = sampled_df['label'].to_numpy()
    num_queries = len(sampled_labels)
    print(f"Sampled {num_queries} items for evaluation.")

    # Category counts within the sample for mAP
    sampled_category_counts = sampled_df['category'].value_counts().to_dict()

    # 3. Compute the distance matrix for the sampled items only
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    dist_matrix = compute_distance_matrix_parallel(searcher, metric, sampled_indices, max_workers)

    # 4. Evaluate mAP using the pre-computed matrix on the sample
    ap_scores = []
    for i in range(num_queries):
        query_label = sampled_labels[i]
        query_category = query_label.split('/')[0]

        distances = dist_matrix[i, :]
        sorted_indices = np.argsort(distances)

        result_indices = sorted_indices[1:top_n + 1]
        result_labels = sampled_labels[result_indices]

        # Total relevant items for this category within the sample
        total_relevant_in_sample = sampled_category_counts.get(query_category, 0) - 1  # Exclude self
        ap = calculate_average_precision(query_category, result_labels.tolist(), total_relevant_in_sample)
        ap_scores.append(ap)

        print(f"Processing query {i + 1}/{num_queries}: {query_label}", end='\r')

    print("\n" + "=" * 50)
    if not ap_scores:
        print("No scores were calculated.")
        return 0.0

    mean_ap = np.mean(ap_scores)
    print(f"Evaluation complete for [Weighting: {weighting_method}, Metric: {metric}]")
    print(f"Sampled Mean Average Precision (mAP) across {num_queries} queries: {mean_ap:.4f}")
    print("=" * 50)

    return mean_ap


if __name__ == '__main__':
    CSV_FILE = '../../Feature-matrix/OldFeatureSets/all_features_modified.csv'
    METRIC_TO_TEST = 'manhattan'

    # 1. Evaluate with your custom manual weights on a sample
    evaluate_performance(
        feature_csv_path=CSV_FILE,
        weights=MANUAL_WEIGHTS,
        weighting_method='feature',
        metric=METRIC_TO_TEST,
        sample_size_per_category=10
    )

    print('neutral = 0.1239')
    # 2. Evaluate with neutral (uniform) weights as a baseline on a sample
    evaluate_performance(
        feature_csv_path=CSV_FILE,
        weights=MANUAL_WEIGHTS,  # Weights are ignored when method is 'neutral'
        weighting_method='neutral',
        metric=METRIC_TO_TEST,
        sample_size_per_category=10
    )


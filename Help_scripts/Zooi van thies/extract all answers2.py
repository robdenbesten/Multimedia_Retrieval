"""
BATCH QUERY PROCESSOR (ALTERNATE VERSION)
This is another version of the batch query system.
It runs all shapes as queries with neutral weighting.
It saves results in a format ready for evaluation.
This version may use different settings than the main extractor.
"""

import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from Querying import ShapeSearcher, MANUAL_WEIGHTS


def generate_retrieval_results_wide_format():
     """
     Generates retrieval results in a wide CSV format, excluding the query itself.
     For each item, N = category_size - 1 (relevant items excluding the query).
     Writes retrieved categories into columns `retrieved_cat_rank_1` ... up to the
     maximum N across categories.
     """

     def category_from_label(label: str) -> str:
         """Extracts the category part from a 'Category/Object' label."""
         return label.split('/', 1)[0] if '/' in label else label

     # --- Configuration ---
     csv_path = '../../Feature-matrix/all_features.csv'
     out_path = 'results_neutral.csv'
     metrics = ['euclidean', 'manhattan', 'manhattan+chi-squared', 'manhattan+emd', 'manhattan+kullback-leibler', 'knn']
     # --- End of Configuration ---

     try:
         searcher = ShapeSearcher(
                feature_csv_path=csv_path,
                weights=MANUAL_WEIGHTS,
                weighting_method = 'neutral'  # or 'neutral'
            )
     except FileNotFoundError as e:
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)

     labels = searcher.get_available_labels()

     # Pre-calculate category counts
     category_counts = defaultdict(int)
     for lbl in labels:
         category_counts[category_from_label(lbl)] += 1

     # Maximum number of comparisons per query is max(category_size - 1, 0)
     max_comparisons = max((cnt - 1) for cnt in category_counts.values()) if category_counts else 0

     # Open CSV for writing
     with open(out_path, 'w', newline='', encoding='utf-8') as f:
         writer = csv.writer(f)
         header = ['query_label', 'query_category', 'metric']
         header.extend([f'retrieved_cat_rank_{i}' for i in range(1, max_comparisons + 1)])
         writer.writerow(header)

         for i, query_label in enumerate(labels):
             print(f"Processing item {i + 1}/{len(labels)}: {query_label}")
             query_category = category_from_label(query_label)

             # N = category size excluding the query itself
             category_size = category_counts[query_category]
             n_comparisons = max(0, category_size - 1)

             for metric in metrics:
                 # Request a few extra to ensure we can exclude the query if returned
                 request_n = n_comparisons + 1 if n_comparisons > 0 else 0
                 neighbors = []
                 if request_n > 0:
                     raw_neighbors = searcher.search(query_label, metric, top_n=request_n)
                     # Exclude the query itself if present and then trim to n_comparisons
                     neighbors = [lbl for lbl in raw_neighbors if lbl != query_label][:n_comparisons]

                 # Map neighbors to categories
                 retrieved_categories = [category_from_label(lbl) for lbl in neighbors]
                 # Pad to max_comparisons so all rows have same columns
                 retrieved_categories += [''] * (max_comparisons - len(retrieved_categories))

                 row = [query_label, query_category, metric] + retrieved_categories
                 writer.writerow(row)

     print(f'\nDetailed retrieval results have been written to `{out_path}`.')


if __name__ == '__main__':
    generate_retrieval_results_wide_format()

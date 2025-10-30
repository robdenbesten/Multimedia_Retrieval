# python
import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS


# def generate_retrieval_results_wide_format():
#     """
#     Generates retrieval results in a wide CSV format, excluding the query itself.
#     For each item, N = category_size - 1 (relevant items excluding the query).
#     Writes retrieved categories into columns `retrieved_cat_rank_1` ... up to the
#     maximum N across categories.
#     """
#
#     def category_from_label(label: str) -> str:
#         """Extracts the category part from a 'Category/Object' label."""
#         return label.split('/', 1)[0] if '/' in label else label
#
#     # --- Configuration ---
#     csv_path = 'Feature-matrix/all_features.csv'
#     out_path = 'raw_res_manhattanKull.csv'
#     metrics = ['manhattan+kullback-leibler']
#     # --- End of Configuration ---
#
#     try:
#         searcher = ShapeSearcher(csv_path, MANUAL_WEIGHTS)
#     except FileNotFoundError as e:
#         print(f"Error: {e}", file=sys.stderr)
#         sys.exit(1)
#
#     labels = searcher.get_available_labels()
#
#     # Pre-calculate category counts
#     category_counts = defaultdict(int)
#     for lbl in labels:
#         category_counts[category_from_label(lbl)] += 1
#
#     # Maximum number of comparisons per query is max(category_size - 1, 0)
#     max_comparisons = max((cnt - 1) for cnt in category_counts.values()) if category_counts else 0
#
#     # Open CSV for writing
#     with open(out_path, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         header = ['query_label', 'query_category', 'metric']
#         header.extend([f'retrieved_cat_rank_{i}' for i in range(1, max_comparisons + 1)])
#         writer.writerow(header)
#
#         for i, query_label in enumerate(labels):
#             print(f"Processing item {i + 1}/{len(labels)}: {query_label}")
#             query_category = category_from_label(query_label)
#
#             # N = category size excluding the query itself
#             category_size = category_counts[query_category]
#             n_comparisons = max(0, category_size - 1)
#
#             for metric in metrics:
#                 # Request a few extra to ensure we can exclude the query if returned
#                 request_n = n_comparisons + 1 if n_comparisons > 0 else 0
#                 neighbors = []
#                 if request_n > 0:
#                     raw_neighbors = searcher.search(query_label, metric, top_n=request_n)
#                     # Exclude the query itself if present and then trim to n_comparisons
#                     neighbors = [lbl for lbl in raw_neighbors if lbl != query_label][:n_comparisons]
#
#                 # Map neighbors to categories
#                 retrieved_categories = [category_from_label(lbl) for lbl in neighbors]
#                 # Pad to max_comparisons so all rows have same columns
#                 retrieved_categories += [''] * (max_comparisons - len(retrieved_categories))
#
#                 row = [query_label, query_category, metric] + retrieved_categories
#                 writer.writerow(row)
#
#     print(f'\nDetailed retrieval results have been written to `raw_res_manhattanKull.csv`.')


def calculate_statistics(csv_path: str, k: int = None):
    """
    Calculate performance statistics from a wide-format CSV.
    `k` specifies the number of top results to consider for each query.
    If `k` is None, it defaults to `N` (category_size - 1).
    """
    all_stats = []
    category_counts = defaultdict(int)
    query_data = []

    # ROC data
    y_true_all = []
    y_scores_all = []

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_data.append(row)
            category_counts[row['query_category']] += 1
    total_items = len(query_data)

    for row in query_data:
        query_category = row['query_category']
        retrieved_cols = [key for key in row.keys() if key.startswith('retrieved_cat_rank_')]
        retrieved_cols.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        retrieved_cats = [row[key] for key in retrieved_cols if row[key] != '']

        # N is the number of relevant items (category size - 1)
        N = max(0, category_counts[query_category] - 1)

        # Determine the number of items to evaluate (k or N)
        eval_k = k if k is not None else N
        retrieved_top_k = retrieved_cats[:eval_k]

        # --- ROC Data Generation ---
        # Create labels (1 for correct, 0 for incorrect) for all retrieved items
        y_true = [1 if cat == query_category else 0 for cat in retrieved_cats]
        # Create scores (higher score for higher rank)
        y_scores = [len(retrieved_cats) - i for i in range(len(retrieved_cats))]

        y_true_all.extend(y_true)
        y_scores_all.extend(y_scores)
        # --- End ROC Data Generation ---

        TP = sum(1 for cat in retrieved_top_k if cat == query_category)
        FP = len(retrieved_top_k) - TP
        FN = N - TP  # Relevant items not in the retrieved top-k
        population_excl_query = max(0, total_items - 1)
        # TN is non-relevant items not retrieved.
        # Total non-relevant = (population - 1) - N
        # Non-relevant retrieved = FP
        TN = (population_excl_query - N) - FP

        precision_at_k_vals = [
            sum(1 for cat in retrieved_top_k[:i] if cat == query_category) / i
            for i in range(1, len(retrieved_top_k) + 1)
        ]
        ap_numerator = sum(p for p, cat in zip(precision_at_k_vals, retrieved_top_k) if cat == query_category)
        # Average Precision is normalized by the number of relevant items (N)
        average_precision = ap_numerator / N if N > 0 else 0.0

        last_rank_consecutive = 0
        for cat in retrieved_top_k:
            if cat == query_category:
                last_rank_consecutive += 1
            else:
                break

        all_stats.append({
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'first_tier_correct': 1 if retrieved_top_k and retrieved_top_k[0] == query_category else 0,
            'average_precision': average_precision,
            'last_rank': last_rank_consecutive
        })

    num_queries = len(all_stats)
    if num_queries == 0:
        print("No data to process.")
        return

    # --- ROC Curve and AUC ---
    roc_auc = 0.0
    if y_true_all and y_scores_all:
        fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
        roc_auc = auc(fpr, tpr)
        specificity = 1 - fpr

        plt.figure()
        # Plotting Specificity vs. Sensitivity
        plt.plot(tpr, specificity, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        # The chance line goes from (0,1) to (1,0) in a Specificity vs Sensitivity plot
        #plt.plot([0, 1], [1, 0], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Sensitivity (True Positive Rate)')
        plt.ylabel('Specificity (True Negative Rate)')
        plt.title(f'ROC Curve for `{csv_path}`')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
    # --- End ROC Curve and AUC ---

    total_tp = sum(s['TP'] for s in all_stats)
    total_fp = sum(s['FP'] for s in all_stats)
    total_tn = sum(s['TN'] for s in all_stats)
    total_fn = sum(s['FN'] for s in all_stats)
    total_pop = total_tp + total_fp + total_tn + total_fn

    accuracy = (total_tp + total_tn) / total_pop if total_pop > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0 # Recall is the same as Sensitivity
    specificity_val = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    relative_error = (total_fp + total_fn) / (total_tp + total_tn) if (total_tp + total_tn) > 0 else 0.0
    first_tier_accuracy = sum(s['first_tier_correct'] for s in all_stats) / num_queries
    mean_avg_precision = sum(s['average_precision'] for s in all_stats) / num_queries
    fvalue = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_last_rank = sum(s['last_rank'] for s in all_stats) / num_queries

    k_str = f"k={k}" if k is not None else "k=N"
    print(f"--- Overall Performance Statistics for `{csv_path}` ({k_str}) ---")
    print(f"Processed {num_queries} queries.\n")
    print("--- Truth Table (Averages per query) ---")
    print(f"True Positives (TP):  {total_tp / num_queries:.2f}")
    print(f"False Positives (FP): {total_fp / num_queries:.2f}")
    print(f"True Negatives (TN):  {total_tn / num_queries:.2f}")
    print(f"False Negatives (FN): {total_fn / num_queries:.2f}\n")
    print("--- Retrieval Metrics ---")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score:             {fvalue:.4f}")
    print(f"Specificity:          {specificity_val:.4f}")
    print(f"Relative Error:       {relative_error:.4f}")
    print(f"First Tier Accuracy:  {first_tier_accuracy:.4f}")
    print(f"Mean Average Precision (MAP): {mean_avg_precision:.4f}")
    print(f"Area Under ROC (AUC): {roc_auc:.4f}")
    print(f"Average Last Rank (consecutive correct from top): {avg_last_rank:.2f}\n")


if __name__ == '__main__':
    # generate_retrieval_results_wide_format() # Uncomment to generate a results file first

    # --- Configuration for Statistics ---
    # Set k to an integer (e.g., 10) to evaluate top-k results.
    # Set k to None to evaluate top-N results (where N = category size - 1).
    K_VALUE = 10

    results_files = [
        'raw_res_manhattan.csv',
        'raw_res_manhattanKull.csv',
        'raw_res_KNN.csv',
        'raw_res_euclidean.csv'
    ]

    for file_path in results_files:
        try:
            calculate_statistics(file_path, k=K_VALUE)
        except FileNotFoundError:
            print(f"Error: The file `{file_path}` was not found. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred while processing `{file_path}`: {e}", file=sys.stderr)
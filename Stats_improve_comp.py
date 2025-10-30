# python
import os
import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def _calculate_stats_for_group(query_data: list, category_counts: dict, k: int = None):
    """
    Calculates performance statistics for a group of query results.

    Args:
        query_data (list): A list of dictionaries, where each dictionary is a row
                           from the input file for a specific metric.
        category_counts (dict): A dictionary mapping category names to their total
                                counts within the parent weight map.
        k (int, optional): The number of top results to consider. Defaults to N.

    Returns:
        A tuple containing:
        - dict: A dictionary of calculated statistics.
        - list: A list of true labels for ROC calculation.
        - list: A list of scores for ROC calculation.
    """
    total_items = sum(category_counts.values())
    all_stats_per_query = []
    y_true_all = []
    y_scores_all = []

    for row in query_data:
        query_category = row['query_category']
        retrieved_cols = sorted(
            [key for key in row.keys() if key.startswith('retrieved_cat_rank_')],
            key=lambda x: int(x.split('_')[-1])
        )
        retrieved_cats = [row[key] for key in retrieved_cols if row.get(key)]

        N = max(0, category_counts.get(query_category, 1) - 1)
        eval_k = k if k is not None else N
        if eval_k == 0 and N > 0: # If k is not set, but N is > 0, use N
             eval_k = N
        elif eval_k == 0 and N == 0: # Avoid division by zero if N is 0
             eval_k = 1

        retrieved_top_k = retrieved_cats[:eval_k]

        # ROC Data Generation
        y_true = [1 if cat == query_category else 0 for cat in retrieved_cats]
        y_scores = [len(retrieved_cats) - i for i in range(len(retrieved_cats))]
        y_true_all.extend(y_true)
        y_scores_all.extend(y_scores)

        TP = sum(1 for cat in retrieved_top_k if cat == query_category)
        FP = len(retrieved_top_k) - TP
        FN = N - TP
        TN = (total_items - 1 - N) - FP

        precision_at_k_vals = [
            sum(1 for cat in retrieved_top_k[:i] if cat == query_category) / i
            for i in range(1, len(retrieved_top_k) + 1)
        ]
        ap_numerator = sum(p for p, cat in zip(precision_at_k_vals, retrieved_top_k) if cat == query_category)
        average_precision = ap_numerator / N if N > 0 else 0.0

        last_rank_consecutive = 0
        for cat in retrieved_top_k:
            if cat == query_category:
                last_rank_consecutive += 1
            else:
                break

        all_stats_per_query.append({
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'first_tier_correct': 1 if retrieved_top_k and retrieved_top_k[0] == query_category else 0,
            'average_precision': average_precision,
            'last_rank': last_rank_consecutive
        })

    num_queries = len(all_stats_per_query)
    if num_queries == 0:
        return {}, [], []

    total_tp = sum(s['TP'] for s in all_stats_per_query)
    total_fp = sum(s['FP'] for s in all_stats_per_query)
    total_tn = sum(s['TN'] for s in all_stats_per_query)
    total_fn = sum(s['FN'] for s in all_stats_per_query)
    total_pop = total_tp + total_fp + total_tn + total_fn

    accuracy = (total_tp + total_tn) / total_pop if total_pop > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    specificity_val = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    fvalue = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    first_tier_accuracy = sum(s['first_tier_correct'] for s in all_stats_per_query) / num_queries
    mean_avg_precision = sum(s['average_precision'] for s in all_stats_per_query) / num_queries
    avg_last_rank = sum(s['last_rank'] for s in all_stats_per_query) / num_queries

    k_str = f"k={k}" if k is not None else "k=N"

    stats_row = {
        'metric': query_data[0]['metric'],
        'weights_map': query_data[0]['weights_map'],
        'k': k_str,
        'num_queries': num_queries,
        'total_TP': total_tp,
        'total_FP': total_fp,
        'total_TN': total_tn,
        'total_FN': total_fn,
        'accuracy': round(accuracy, 6),
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1_score': round(fvalue, 6),
        'specificity': round(specificity_val, 6),
        'first_tier_accuracy': round(first_tier_accuracy, 6),
        'mean_average_precision': round(mean_avg_precision, 6),
        'avg_last_rank': round(avg_last_rank, 6)
    }
    return stats_row, y_true_all, y_scores_all

def generate_report_for_weight_map(weight_map_name: str, data_by_metric: dict, category_counts: dict, k: int = None):
    """
    Generates a consolidated report for a single weight map.

    Args:
        weight_map_name (str): The name of the weight map being processed.
        data_by_metric (dict): Data grouped by metric for this weight map.
        category_counts (dict): Counts of each category for this weight map.
        k (int, optional): The 'k' value for top-k analysis.
    """
    all_stats_rows = []
    plt.figure(figsize=(10, 8))

    for metric_name, metric_data in data_by_metric.items():
        stats_row, y_true, y_scores = _calculate_stats_for_group(metric_data, category_counts, k=k)
        if not stats_row:
            print(f"Warning: No data for metric `{metric_name}` in `{weight_map_name}`. Skipping.", file=sys.stderr)
            continue

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        specificity = 1 - fpr
        roc_auc = auc(tpr, specificity) # AUC of Sensitivity vs. Specificity
        stats_row['roc_auc'] = round(auc(fpr, tpr), 6) # Store standard AUC
        all_stats_rows.append(stats_row)

        # Plot Sensitivity (TPR) vs. Specificity (1-FPR)
        plt.plot(tpr, specificity, lw=2, label=f'{metric_name} (AUC = {auc(fpr, tpr):.2f})')

    # Finalize and save the ROC plot for the weight map
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--') # Chance line for Sensitivity vs. Specificity
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensitivity (True Positive Rate)')
    plt.ylabel('Specificity (True Negative Rate)')
    plt.title(f'Consolidated ROC Curves for {weight_map_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plot_path = f"{weight_map_name}_ROC_curves.png"
    plt.savefig(plot_path)
    plt.close() # Close figure to free memory
    print(f"Report for `{weight_map_name}` saved to `{plot_path}`")

    # Write the statistics to a CSV file for the weight map
    if not all_stats_rows:
        return
    stats_csv_path = f"{weight_map_name}_stats_summary.csv"
    # Ensure roc_auc is the last column for consistency
    fieldnames = [f for f in all_stats_rows[0].keys() if f != 'roc_auc'] + ['roc_auc']
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats_rows)
    print(f"Statistics for `{weight_map_name}` saved to `{stats_csv_path}`")

def process_input_file(csv_path: str, k: int = None):
    """
    Reads the main input CSV and generates reports for each weight map.

    Args:
        csv_path (str): The path to the input CSV file.
        k (int, optional): The 'k' value for top-k analysis.
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            all_data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Error: The file `{csv_path}` was not found.", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred while reading `{csv_path}`: {e}", file=sys.stderr)
        return

    # Group all data by weight map
    data_by_weight_map = defaultdict(list)
    for row in all_data:
        data_by_weight_map[row['weights_map']].append(row)

    # Process each weight map group separately
    for weight_map_name, weight_map_data in data_by_weight_map.items():
        print(f"\nProcessing weight map: `{weight_map_name}`...")

        # Group this weight map's data by metric
        data_by_metric = defaultdict(list)
        # Calculate category counts *once* for the entire weight map
        category_counts = defaultdict(int)
        # Use a set to count unique queries for category counts
        processed_queries = set()

        for row in weight_map_data:
            data_by_metric[row['metric']].append(row)
            query_id = (row['query_label'], row['query_category'])
            if query_id not in processed_queries:
                category_counts[row['query_category']] += 1
                processed_queries.add(query_id)

        generate_report_for_weight_map(weight_map_name, data_by_metric, category_counts, k)

if __name__ == '__main__':
    # --- Configuration ---
    # Define the single input file containing all results.
    INPUT_FILE = 'raw_results_all.csv'

    # Set k to an integer for top-k, or None for top-N (category size - 1).
    K_VALUE = None

    # --- Execution ---
    process_input_file(INPUT_FILE, k=K_VALUE)
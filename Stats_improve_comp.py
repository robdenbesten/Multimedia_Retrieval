# python
import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def calculate_statistics_for_metric(metric_name: str, query_data: list, csv_path: str, k: int | None = None):
    """
    Calculate stats for a single metric group and return ROC data (fpr, tpr, auc).
    Uses macro-averaging for stratified evaluation.
    """
    all_stats = []
    category_counts = defaultdict(int)

    # count queries per class
    for row in query_data:
        category_counts[row['query_category']] += 1

    total_items = len(query_data)

    for row in query_data:
        query_category = row['query_category']
        retrieved_cols = [key for key in row.keys() if key.startswith('retrieved_cat_rank_')]
        retrieved_cols.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        retrieved_cats = [row[key] for key in retrieved_cols if row[key] != '']

        N = max(0, category_counts[query_category] - 1)
        eval_k = k if k is not None else N
        retrieved_top_k = retrieved_cats[:eval_k]

        TP = sum(1 for cat in retrieved_top_k if cat == query_category)
        FP = len(retrieved_top_k) - TP
        FN = N - TP
        population_excl_query = max(0, total_items - 1)
        TN = (population_excl_query - N) - FP

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

        all_stats.append({
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'first_tier_correct': 1 if retrieved_top_k and retrieved_top_k[0] == query_category else 0,
            'average_precision': average_precision,
            'last_rank': last_rank_consecutive,
            'query_category': query_category,
            'retrieved_cats': retrieved_cats,
        })

    num_queries = len(all_stats)
    if num_queries == 0:
        print(f"No data to process for metric '{metric_name}'.")
        return None, None, 0.0

    # --- Stratified Evaluation (Macro-Averaging) ---
    stats_by_category = defaultdict(list)
    for stat in all_stats:
        stats_by_category[stat['query_category']].append(stat)

    category_metrics = []
    all_tpr = {}
    mean_fpr = np.linspace(0, 1, 100)

    for category, cat_stats in stats_by_category.items():
        cat_tp = sum(s['TP'] for s in cat_stats)
        cat_fp = sum(s['FP'] for s in cat_stats)
        cat_tn = sum(s['TN'] for s in cat_stats)
        cat_fn = sum(s['FN'] for s in cat_stats)

        accuracy = (cat_tp + cat_tn) / (cat_tp + cat_fp + cat_tn + cat_fn) if (cat_tp + cat_fp + cat_tn + cat_fn) > 0 else 0.0
        precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        specificity = cat_tn / (cat_tn + cat_fp) if (cat_tn + cat_fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        first_tier_accuracy = sum(s['first_tier_correct'] for s in cat_stats) / len(cat_stats) if cat_stats else 0.0
        mean_avg_precision = sum(s['average_precision'] for s in cat_stats) / len(cat_stats) if cat_stats else 0.0
        avg_last_rank = sum(s['last_rank'] for s in cat_stats) / len(cat_stats) if cat_stats else 0.0

        category_metrics.append({
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1,
            'first_tier_accuracy': first_tier_accuracy, 'mean_avg_precision': mean_avg_precision,
            'avg_last_rank': avg_last_rank
        })

        # ROC for this category
        y_true_cat = []
        y_scores_cat = []
        for s in cat_stats:
            y_true_cat.extend([1 if cat == s['query_category'] else 0 for cat in s['retrieved_cats']])
            y_scores_cat.extend([len(s['retrieved_cats']) - i for i in range(len(s['retrieved_cats']))])

        if y_true_cat and y_scores_cat:
            fpr, tpr, _ = roc_curve(y_true_cat, y_scores_cat)
            all_tpr[category] = np.interp(mean_fpr, fpr, tpr)
            all_tpr[category][0] = 0.0

    # Macro-average metrics
    num_categories = len(category_metrics)
    if num_categories == 0:
        return None, None, 0.0

    macro_avg_accuracy = sum(m['accuracy'] for m in category_metrics) / num_categories
    macro_avg_precision = sum(m['precision'] for m in category_metrics) / num_categories
    macro_avg_recall = sum(m['recall'] for m in category_metrics) / num_categories
    macro_avg_specificity = sum(m['specificity'] for m in category_metrics) / num_categories
    macro_avg_f1 = sum(m['f1'] for m in category_metrics) / num_categories
    macro_avg_first_tier = sum(m['first_tier_accuracy'] for m in category_metrics) / num_categories
    macro_avg_map = sum(m['mean_avg_precision'] for m in category_metrics) / num_categories
    macro_avg_last_rank = sum(m['avg_last_rank'] for m in category_metrics) / num_categories

    # Macro-average ROC
    if all_tpr:
        mean_tpr = np.mean([tpr for tpr in all_tpr.values()], axis=0)
        mean_tpr[-1] = 1.0
        roc_auc = auc(mean_fpr, mean_tpr)
        final_tpr, final_fpr = mean_tpr, mean_fpr
    else:
        final_tpr, final_fpr, roc_auc = None, None, 0.0

    k_str = f"k={k}" if k is not None else "k=N"
    print(f"--- Macro-Averaged Stats for `{csv_path}` (Metric: {metric_name}, {k_str}) ---")
    print(f"Processed {num_queries} queries across {num_categories} categories.")
    print(f"Accuracy: {macro_avg_accuracy:.4f}, Precision: {macro_avg_precision:.4f}, Recall: {macro_avg_recall:.4f}, F1: {macro_avg_f1:.4f}, AUC: {roc_auc:.4f}\n")

    return final_fpr, final_tpr, roc_auc


def calculate_statistics_by_metric(csv_path: str, metric_column: str, k: int | None = None,
                                   weights_column: str | None = None, weights_value: str | None = None):
    """
    Calculate performance statistics from a wide-format CSV, grouped by a metric column.
    Optionally filters rows by a weights column/value (e.g., 'neutral', 'adjusted', 'distance').
    """
    grouped_data = defaultdict(list)

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if metric_column not in reader.fieldnames:
            print(f"Error: Metric column `{metric_column}` not found in `{csv_path}`.", file=sys.stderr)
            return

        chosen_wcol = None
        if weights_value is not None:
            candidates = []
            if weights_column:
                candidates.append(weights_column)
            # Auto-detect common names if explicit column not provided or missing
            candidates += ['weights_map', 'weights', 'weighting_method']
            chosen_wcol = next((c for c in candidates if c in reader.fieldnames), None)
            if not chosen_wcol:
                print(f"Error: No weights column found in `{csv_path}`. Tried: {candidates}.", file=sys.stderr)
                return

        for row in reader:
            if chosen_wcol is not None and row.get(chosen_wcol) != weights_value:
                continue
            grouped_data[row[metric_column]].append(row)

    if not grouped_data:
        sel = f" with `{weights_column or chosen_wcol}`=`{weights_value}`" if weights_value is not None else ""
        print(f"No data to process in the file `{csv_path}`{sel}.")
        return

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', max(1, len(grouped_data)))

    for i, (metric_name, query_data) in enumerate(grouped_data.items()):
        fpr, tpr, roc_auc = calculate_statistics_for_metric(metric_name, query_data, csv_path, k)
        if fpr is not None and tpr is not None:
            specificity = 1 - fpr
            plt.plot(tpr, specificity, color=colors(i), lw=2, label=f'{metric_name} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [1, 0], color='red', lw=1, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    suffix = f" | weights={weights_value}" if weights_value is not None else ""
    plt.xlabel('Sensitivity (TPR)')
    plt.ylabel('Specificity (1 - FPR)')
    plt.title(f'Combined ROC Curve for `{csv_path}` (Macro-Averaged){suffix}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Single combined results file; filter by a column that specifies weight method
    K_VALUE = None
    METRIC_COLUMN_NAME = 'metric'
    RESULTS_FILE = 'raw_results_all2.csv'  # your single file
    WEIGHTS_COLUMN = 'weights_map'        # set to the actual column name in your CSV
    WEIGHTS_FILTER = 'adjusted'           # e.g., 'neutral', 'adjusted', 'distance'

    try:
        calculate_statistics_by_metric(
            RESULTS_FILE,
            metric_column=METRIC_COLUMN_NAME,
            k=K_VALUE,
            weights_column=WEIGHTS_COLUMN,
            weights_value=WEIGHTS_FILTER
        )
    except FileNotFoundError:
        print(f"Error: The file `{RESULTS_FILE}` was not found. Skipping.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred while processing `{RESULTS_FILE}`: {e}", file=sys.stderr)
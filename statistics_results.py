import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import re
import os


def calculate_statistics_for_metric(metric_name: str, query_data: list, category_counts: dict, k: int = None):
    """
    Calculate statistics with proper per-category weighting.
    Each category contributes equally to the final metrics regardless of size.
    """
    rank_re = re.compile(r'retrieved_cat_rank_(\d+)$')

    # Group queries by category
    queries_by_category = defaultdict(list)
    for row in query_data:
        query_category = row.get('query_category', '').strip()
        if query_category:
            queries_by_category[query_category].append(row)

    if not queries_by_category:
        print(f"No valid queries found for metric '{metric_name}'.")
        return None, None, 0.0, {}

    # Store per-category results
    category_results = {}
    all_y_true = []
    all_y_scores = []

    # For accuracy calculation
    total_items_in_db = sum(category_counts.values())

    for category, category_queries in queries_by_category.items():
        N = max(0, category_counts.get(category, 0) - 1)  # Relevant items (excluding query itself)
        if N == 0:
            print(f"Warning: Category '{category}' has no other items. Skipping.")
            continue

        # If k is specified, use it. Otherwise, use all retrieved items (no truncation)
        eval_k = k

        category_stats = {
            'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
            'precisions': [],
            'recalls': [],
            'aps': [],
            'first_tier_correct': 0,
            'last_ranks': [],
            'y_true': [],
            'y_scores': []
        }

        for row in category_queries:
            # Parse retrieved columns
            retrieved_cols = [col for col in row.keys() if rank_re.match(col)]
            retrieved_cols.sort(key=lambda c: int(rank_re.match(c).group(1)))
            retrieved_cats = [row[col].strip() if row[col] else '' for col in retrieved_cols]

            # Filter non-empty results
            non_empty = [c for c in retrieved_cats if c != '']
            # Use eval_k if specified, otherwise use all retrieved items
            retrieved_top_k = non_empty[:eval_k] if eval_k is not None else non_empty

            if not retrieved_top_k:
                continue

            # Calculate TP and FP for this query
            TP = sum(1 for cat in retrieved_top_k if cat == category)
            FP = len(retrieved_top_k) - TP
            FN = max(0, N - TP)

            # Calculate TN (items in database not retrieved and not relevant)
            # Total negative items = total_items - 1 (query) - N (relevant items)
            total_negatives = max(0, total_items_in_db - 1 - N)
            TN = max(0, total_negatives - FP)

            category_stats['TP'] += TP
            category_stats['FP'] += FP
            category_stats['FN'] += FN
            category_stats['TN'] += TN

            # Precision and Recall for this query
            precision = TP / len(retrieved_top_k) if retrieved_top_k else 0.0
            recall = TP / N if N > 0 else 0.0
            category_stats['precisions'].append(precision)
            category_stats['recalls'].append(recall)

            # Average Precision (AP) for this query
            precision_at_i = []
            for i in range(1, len(retrieved_top_k) + 1):
                tp_at_i = sum(1 for cat in retrieved_top_k[:i] if cat == category)
                precision_at_i.append(tp_at_i / i)

            ap_numerator = sum(p for p, cat in zip(precision_at_i, retrieved_top_k) if cat == category)
            ap = ap_numerator / N if N > 0 else 0.0
            category_stats['aps'].append(ap)

            # First tier accuracy
            if retrieved_top_k[0] == category:
                category_stats['first_tier_correct'] += 1

            # Last consecutive rank
            last_rank = 0
            for cat in retrieved_top_k:
                if cat == category:
                    last_rank += 1
                else:
                    break
            category_stats['last_ranks'].append(last_rank)

            # ROC data (per-item scores using inverse rank)
            for i, cat in enumerate(retrieved_top_k):
                category_stats['y_true'].append(1 if cat == category else 0)
                category_stats['y_scores'].append(1.0 / (i + 1))

        # Aggregate per-category metrics
        num_queries = len(category_queries)
        if num_queries > 0:
            category_results[category] = {
                'num_queries': num_queries,
                'N': N,
                'precision': np.mean(category_stats['precisions']) if category_stats['precisions'] else 0.0,
                'recall': np.mean(category_stats['recalls']) if category_stats['recalls'] else 0.0,
                'map': np.mean(category_stats['aps']) if category_stats['aps'] else 0.0,
                'first_tier_accuracy': category_stats['first_tier_correct'] / num_queries,
                'avg_last_rank': np.mean(category_stats['last_ranks']) if category_stats['last_ranks'] else 0.0,
                'avg_TP': category_stats['TP'] / num_queries,
                'avg_FP': category_stats['FP'] / num_queries,
                'avg_TN': category_stats['TN'] / num_queries,
                'avg_FN': category_stats['FN'] / num_queries,
            }

            all_y_true.extend(category_stats['y_true'])
            all_y_scores.extend(category_stats['y_scores'])

    if not category_results:
        print(f"No valid category results for metric '{metric_name}'.")
        return None, None, 0.0, {}, {}

    # Calculate macro-averaged metrics (each category weighted equally)
    macro_precision = np.mean([res['precision'] for res in category_results.values()])
    macro_recall = np.mean([res['recall'] for res in category_results.values()])
    macro_map = np.mean([res['map'] for res in category_results.values()])
    macro_first_tier = np.mean([res['first_tier_accuracy'] for res in category_results.values()])
    macro_last_rank = np.mean([res['avg_last_rank'] for res in category_results.values()])

    macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall) \
        if (macro_precision + macro_recall) > 0 else 0.0

    # Calculate micro-averaged metrics (aggregate TP/FP/FN/TN across all queries)
    total_tp = sum(res['avg_TP'] * res['num_queries'] for res in category_results.values())
    total_fp = sum(res['avg_FP'] * res['num_queries'] for res in category_results.values())
    total_tn = sum(res['avg_TN'] * res['num_queries'] for res in category_results.values())
    total_fn = sum(res['avg_FN'] * res['num_queries'] for res in category_results.values())
    total_queries = sum(res['num_queries'] for res in category_results.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0.0

    # Accuracy and specificity
    total_population = total_tp + total_fp + total_tn + total_fn
    accuracy = (total_tp + total_tn) / total_population if total_population > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

    # ROC curve (using all retrieved items)
    fpr, tpr, roc_auc = None, None, 0.0
    if all_y_true and all_y_scores:
        try:
            fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            print(f"Warning: Could not compute ROC for '{metric_name}': {e}")

    # Print detailed results
    k_str = f"k={k}" if k is not None else "k=N"
    print(f"\n{'=' * 70}")
    print(f"Metric: {metric_name} ({k_str})")
    print(f"{'=' * 70}")
    print(f"Total queries processed: {total_queries}")
    print(f"Number of categories: {len(category_results)}\n")

    print(f"{'Category':<25} {'Queries':<8} {'N':<5} {'Precision':<10} {'Recall':<10} {'MAP':<10}")
    print(f"{'-' * 70}")
    for cat, res in sorted(category_results.items()):
        print(f"{cat:<25} {res['num_queries']:<8} {res['N']:<5} "
              f"{res['precision']:<10.4f} {res['recall']:<10.4f} {res['map']:<10.4f}")

    print(f"\n{'Macro-averaged metrics (equal category weight):'}")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-Score: {macro_f1:.4f}")
    print(f"  MAP: {macro_map:.4f}")
    print(f"  First Tier Accuracy: {macro_first_tier:.4f}")
    print(f"  Avg Last Rank: {macro_last_rank:.2f}")

    print(f"\n{'Micro-averaged metrics (query-level aggregate):'}")
    print(f"  Accuracy: {accuracy:.4f} (WARNING: Inflated by large TN)")
    print(f"  Precision: {micro_precision:.4f}")
    print(f"  Recall: {micro_recall:.4f}")
    print(f"  F1-Score: {micro_f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Avg TP: {total_tp / total_queries:.2f}")
    print(f"  Avg FP: {total_fp / total_queries:.2f}")
    print(f"  Avg TN: {total_tn / total_queries:.2f}")
    print(f"  Avg FN: {total_fn / total_queries:.2f}")

    if roc_auc > 0:
        print(f"\n  ROC AUC: {roc_auc:.4f}")

    print(f"{'=' * 70}\n")

    # Prepare summary dictionary
    stats_dict = {
        'Metric': metric_name,
        'k': k_str,
        'Num_Queries': total_queries,
        'Num_Categories': len(category_results),
        'Accuracy': f"{accuracy:.4f}",
        'Macro_Precision': f"{macro_precision:.4f}",
        'Macro_Recall': f"{macro_recall:.4f}",
        'Macro_F1': f"{macro_f1:.4f}",
        'Macro_MAP': f"{macro_map:.4f}",
        'Macro_First_Tier': f"{macro_first_tier:.4f}",
        'Macro_Avg_Last_Rank': f"{macro_last_rank:.2f}",
        'Micro_Precision': f"{micro_precision:.4f}",
        'Micro_Recall': f"{micro_recall:.4f}",
        'Micro_F1': f"{micro_f1:.4f}",
        'Specificity': f"{specificity:.4f}",
        'ROC_AUC': f"{roc_auc:.4f}",
        'Avg_TP': f"{total_tp / total_queries:.2f}",
        'Avg_FP': f"{total_fp / total_queries:.2f}",
        'Avg_TN': f"{total_tn / total_queries:.2f}",
        'Avg_FN': f"{total_fn / total_queries:.2f}",
    }

    return fpr, tpr, roc_auc, stats_dict, category_results


def calculate_statistics(csv_path: str, k: int = None):
    """
    Main driver for statistics calculation with proper category balancing.
    """
    # Read all rows
    all_rows = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    except FileNotFoundError:
        print(f"Error: The file `{csv_path}` was not found.", file=sys.stderr)
        return

    if not all_rows:
        print("No data to process in the CSV file.")
        return

    # Build unique queries and category counts
    unique_queries = {}
    for row in all_rows:
        qlabel = row.get('query_label', '').strip()
        qcat = row.get('query_category', '').strip()
        if qlabel and qlabel not in unique_queries:
            unique_queries[qlabel] = qcat

    category_counts = defaultdict(int)
    for cat in unique_queries.values():
        category_counts[cat] += 1

    print(f"\nDataset Overview:")
    print(f"  Total unique queries: {len(unique_queries)}")
    print(f"  Number of categories: {len(category_counts)}")
    print(f"  Category sizes: {dict(category_counts)}\n")

    # Group by metric
    metrics = sorted(list(set(row.get('metric', '').strip() for row in all_rows if row.get('metric', '').strip())))
    metric_data = {metric: [row for row in all_rows if row.get('metric', '').strip() == metric] for metric in metrics}

    # Calculate statistics for each metric
    plt.figure(figsize=(10, 10))
    all_metric_stats = []

    # Store per-category results for heatmap
    category_performance = {}  # {metric: {category: {'map': x, 'f1': y, 'precision': z, 'recall': w}}}

    for metric in metrics:
        fpr, tpr, roc_auc, stats_dict, category_results = calculate_statistics_for_metric(
            metric_name=metric,
            query_data=metric_data[metric],
            category_counts=dict(category_counts),
            k=k
        )
        if stats_dict:
            all_metric_stats.append(stats_dict)

        # Store category-level results for heatmap
        if category_results:
            category_performance[metric] = category_results

        if fpr is not None and tpr is not None and roc_auc > 0:
            # Plot a standard ROC curve: TPR vs FPR
            plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {roc_auc:.3f})')

    # Write summary stats to CSV
    if all_metric_stats:
        output_filename = f"stats_{os.path.basename(csv_path)}"
        fieldnames = all_metric_stats[0].keys()
        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_metric_stats)
            print(f"\nSummary statistics saved to `{output_filename}`\n")
        except IOError as e:
            print(f"Error writing to file `{output_filename}`: {e}", file=sys.stderr)

    # Finalize the standard ROC plot
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title(f'ROC Curves - {os.path.basename(csv_path)}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Generate heatmap for per-category MAP performance
    if category_performance:
        # Collect all categories across all metrics
        all_categories = sorted(list(set(
            cat for metric_cats in category_performance.values()
            for cat in metric_cats.keys()
        )))

        # Build MAP matrix (rows = categories, columns = metrics)
        map_matrix = []
        for category in all_categories:
            map_row = []
            for metric in metrics:
                if category in category_performance.get(metric, {}):
                    cat_data = category_performance[metric][category]
                    map_row.append(cat_data['map'])
                else:
                    map_row.append(0.0)
            map_matrix.append(map_row)

        # Calculate figure size based on content
        fig_width = max(12, len(metrics) * 1.5)
        fig_height = max(8, len(all_categories) * 0.5)

        # Create single MAP heatmap with better layout
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # MAP heatmap with red-to-green color scheme
        # Low scores = red (bad), Medium = yellow, High scores = green (good)
        im = ax.imshow(map_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels with better formatting
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(all_categories)))
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(all_categories, fontsize=10)

        # Add title and labels
        ax.set_title(f'Mean Average Precision (MAP) per Category\n{os.path.basename(csv_path)}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Distance Metric', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Category', fontsize=13, fontweight='bold', labelpad=10)

        # Add text annotations with better visibility
        for i in range(len(all_categories)):
            for j in range(len(metrics)):
                value = map_matrix[i][j]
                # Choose text color based on background brightness
                # Black text works best for RdYlGn across most values
                text_color = "black"
                text = ax.text(j, i, f'{value:.3f}',
                              ha="center", va="center",
                              color=text_color,
                              fontsize=9,
                              fontweight='bold')

        # Add colorbar with better formatting
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('MAP Score', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Add grid for better readability
        ax.set_xticks(np.arange(len(metrics)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(all_categories)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    K_VALUE = None  # Set to integer for top-k evaluation, None for top-N

    results_files = [
        'results_neutral.csv',
        'results_adjusted.csv'
    ]

    for file_path in results_files:
        if os.path.exists(file_path):
            print(f"\n{'#' * 70}")
            print(f"# Processing: {file_path}")
            print(f"{'#' * 70}")
            calculate_statistics(file_path, k=K_VALUE)
        else:
            print(f"Warning: File `{file_path}` not found. Skipping.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from math import pi

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_neutral_euclidean_vs_manhattan(neutral_df, folder):
    """A. Plot comparison between neutral Euclidean and Manhattan."""
    metrics_to_compare = ['euclidean', 'manhattan']
    df = neutral_df[neutral_df['Metric'].isin(metrics_to_compare)]
    if len(df) < 2:
        print("! Skipping plot A: Not enough data for Euclidean vs Manhattan.")
        return

    f1_scores = df.set_index('Metric').loc[metrics_to_compare, 'Macro_F1']

    x = np.arange(len(metrics_to_compare))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, f1_scores.values, width, color=['skyblue', 'lightcoral'])

    ax.set_ylabel('F1 Score')
    ax.set_title('Neutral Weights: Euclidean vs. Manhattan (F1)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Euclidean', 'Manhattan'])
    ax.grid(axis='y', alpha=0.5)
    ax.set_ylim(0, f1_scores.max() * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(folder / 'A_neutral_euclidean_vs_manhattan.png', dpi=300)
    plt.close()


def plot_manhattan_variations(neutral_df, folder):
    """B. Plot comparison between neutral Manhattan and its composite variations."""
    metrics_to_compare = ['manhattan', 'manhattan+chi-squared', 'manhattan+emd', 'manhattan+kullback-leibler']
    df = neutral_df[neutral_df['Metric'].isin(metrics_to_compare)]

    if len(df) < len(metrics_to_compare):
        print(f"! Skipping plot B: Not enough data for Manhattan variations. Found {len(df)} of {len(metrics_to_compare)} metrics.")
        return

    df = df.set_index('Metric').loc[metrics_to_compare]

    f1_scores = df['Macro_F1']

    labels = [label.replace('+', '\n+') for label in df.index]
    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, f1_scores, width, color='teal')

    ax.set_ylabel('F1 Score')
    ax.set_title('Neutral Weights: Manhattan vs. Manhattan Variations (F1)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.5)
    ax.set_ylim(0, f1_scores.max() * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(folder / 'B_manhattan_variations.png', dpi=300)
    plt.close()


def prove_knn_euclidean_equivalence(neutral_df, folder):
    """E. Prove that k-NN and Euclidean with neutral weights are equivalent."""

    # Extract the two metrics
    knn_data = neutral_df[neutral_df['Metric'] == 'knn']
    euclidean_data = neutral_df[neutral_df['Metric'] == 'euclidean']

    if len(knn_data) == 0 or len(euclidean_data) == 0:
        print("! Skipping proof: k-NN or Euclidean data not found in neutral results.")
        return

    knn_row = knn_data.iloc[0]
    euclidean_row = euclidean_data.iloc[0]

    # Metrics to compare
    metrics_to_compare = ['Macro_Precision', 'Macro_Recall', 'Macro_F1',
                          'Macro_MAP', 'Macro_First_Tier', 'ROC_AUC']

    # Calculate differences
    differences = []
    for metric in metrics_to_compare:
        knn_val = knn_row[metric]
        euclidean_val = euclidean_row[metric]
        diff = abs(knn_val - euclidean_val)
        differences.append({
            'Metric': metric.replace('_', ' '),
            'k-NN': f'{knn_val:.6f}',
            'Euclidean': f'{euclidean_val:.6f}',
            'Absolute Difference': f'{diff:.6f}'
        })

    df = pd.DataFrame(differences)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Colors
    ORANGE_HEADER = '#cf5230'
    WHITE = '#FFFFFF'
    LIGHT_GREEN = '#D4EDDA'

    # Create cell colors - highlight if difference is essentially zero
    cell_colors = []
    for idx in range(len(df)):
        diff_val = float(df.iloc[idx]['Absolute Difference'])
        # If difference is < 0.0001, highlight in green
        if diff_val < 0.0001:
            cell_colors.append([WHITE, WHITE, WHITE, LIGHT_GREEN])
        else:
            cell_colors.append([WHITE, WHITE, WHITE, WHITE])

    # Column header colors
    col_colors = [ORANGE_HEADER] * 4

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     cellColours=cell_colors,
                     colColours=col_colors)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white', size=12)
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.5)

    # Style data cells
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_edgecolor('#333333')
            cell.set_linewidth(1.5)
            cell.set_text_props(weight='bold', color='#333333', size=11)

            if j == 0:
                # Metric names - left aligned
                cell.set_text_props(ha='left')

    plt.title('Proof: k-NN ≡ Euclidean Distance (Neutral Weights)',
              fontsize=15, fontweight='bold', pad=20, color='#333333')

    # Add annotation
    max_diff = df['Absolute Difference'].astype(float).max()
    annotation = f"Maximum absolute difference: {max_diff:.6f}\n"
    annotation += "Green highlighting indicates differences < 0.0001 (effectively identical)"

    plt.figtext(0.5, 0.02, annotation, ha='center', fontsize=10,
                style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(folder / 'E_knn_euclidean_proof.png', dpi=300, bbox_inches='tight')
    print("✓ Proof table saved as 'E_knn_euclidean_proof.png'")
    plt.close()

    # Print summary to console
    print("\n" + "=" * 60)
    print("K-NN VS EUCLIDEAN COMPARISON (NEUTRAL WEIGHTS)")
    print("=" * 60)
    for _, row in df.iterrows():
        diff = float(row['Absolute Difference'])
        status = "✓ IDENTICAL" if diff < 0.0001 else "⚠ DIFFERENT"
        print(f"{row['Metric']:20s} | Diff: {row['Absolute Difference']:10s} | {status}")
    print("=" * 60)

def plot_feature_weights(folder):
    """C. Plot the normalized feature weights as a table."""
    weights = {
        'A3': 0.31, 'D1': 0.21, 'D2': 0.11, 'D3': 0.09, 'D4': 0.02,
        'Surface area': 0.01, 'Sphericity': 0.02, 'Rectangularity': 0.01,
        'Diameter': 0.01, 'Convexity': 0.06, 'Eccentricity': 0.15
    }

    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Sort by weight descending for the table
    sorted_weights = sorted(normalized_weights.items(), key=lambda item: item[1], reverse=True)

    # Display values as percentages
    cell_data = [[feature, f'{weight * 100:.2f}%'] for feature, weight in sorted_weights]
    columns = ['Feature', 'Weight (%)']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=cell_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('white')  # Purple header
        table[(0, i)].set_text_props(weight='bold', color='black')

    # Style cells
    for i in range(len(cell_data)):
        table[(i + 1, 0)].set_text_props(ha='left')
        table[(i + 1, 1)].set_text_props(ha='right')

    plt.title('Feature Weights for Metrics', fontweight='bold', fontsize=16, pad=20)
    plt.savefig(folder / 'C_feature_weights.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_weighted_results_table(folder):
    """F. Display weighted results in a clean table format."""
    # Read data from CSV file
    csv_path = Path('stats_results_weighted.csv')
    if not csv_path.exists():
        print(f"! Skipping weighted results table: {csv_path} not found.")
        return

    weighted_df = pd.read_csv(csv_path)

    # Define metric order and display names
    metric_order = ['manhattan', 'manhattan+kullback-leibler', 'manhattan+chi-squared', 'manhattan+emd', 'euclidean']
    metric_display_names = {
        'manhattan': 'Manhattan',
        'manhattan+kullback-leibler': '+Kullback-Leibler',
        'manhattan+chi-squared': '+Chi-Squared',
        'manhattan+emd': '+EMD',
        'euclidean': 'Euclidean'
    }

    # Filter and order the data
    weighted_df = weighted_df[weighted_df['Metric'].isin(metric_order)]
    weighted_df['Metric'] = pd.Categorical(weighted_df['Metric'], categories=metric_order, ordered=True)
    weighted_df = weighted_df.sort_values('Metric')

    # Extract the metrics we want to display
    data = {
        'Metric': ['MAP', 'F1', 'AUC', '1st Tier Acc.', 'Avg. Last Rank']
    }

    for _, row in weighted_df.iterrows():
        metric_name = metric_display_names[row['Metric']]
        data[metric_name] = [
            float(row['Macro_MAP']),
            float(row['Macro_F1']),
            float(row['ROC_AUC']),
            float(row['Macro_First_Tier']),
            float(row['Macro_Avg_Last_Rank'])
        ]

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Colors
    ORANGE_HEADER = '#cf5230'
    WHITE = '#FFFFFF'
    LIGHT_ORANGE = '#FFFFFF'

    # Format the data to 4 decimal places for display
    formatted_data = df.copy()
    for col in df.columns[1:]:
        formatted_data[col] = df[col].apply(lambda x: f'{x:.4f}')

    # Define soft colors for highlighting
    SOFT_GREEN = '#D4EDDA'
    SOFT_RED = '#F8D7DA'

    # Create cell colors - highlight min/max per row
    cell_colors = []
    for idx in range(len(df)):
        row_colors = []
        # First column (Metric name) - same orange as header
        row_colors.append(ORANGE_HEADER)

        # For value columns, find min and max
        row_values = df.iloc[idx, 1:].values  # Skip first column (Metric names)
        max_val = max(row_values)
        min_val = min(row_values)

        for val in row_values:
            if val == max_val:
                row_colors.append(SOFT_GREEN)  # Highest value = green
            elif val == min_val:
                row_colors.append(SOFT_RED)    # Lowest value = red
            else:
                # Alternating background for middle values
                if idx % 2 == 0:
                    row_colors.append(WHITE)
                else:
                    row_colors.append(LIGHT_ORANGE)

        cell_colors.append(row_colors)

    # Column header colors
    col_colors = [ORANGE_HEADER] * len(df.columns)

    # Create the table
    table = ax.table(cellText=formatted_data.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     cellColours=cell_colors,
                     colColours=col_colors)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white', size=12)
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.5)

    # Style data cells
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_edgecolor('#333333')
            cell.set_linewidth(1.5)

            if j == 0:
                # Metric names - left aligned and bold with white text on orange background
                cell.set_text_props(weight='bold', ha='left', color='white', size=11)
            else:
                # Values - centered and bold
                cell.set_text_props(weight='bold', color='#333333', size=11)

    plt.title('Weighted Results Comparison',
              fontsize=15, fontweight='bold', pad=20, color='#333333')

    # Add annotation
    annotation = "Performance metrics for different distance functions with adjusted feature weights"
    plt.figtext(0.5, 0.02, annotation, ha='center', fontsize=10,
                style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(folder / 'F_weighted_results_table.png', dpi=300, bbox_inches='tight')
    print("✓ Weighted results table saved as 'F_weighted_results_table.png'")
    plt.close()

def plot_adjusted_metrics_comparison(neutral_df, adjusted_df, folder):
    """D. Plot comparison of neutral vs adjusted Manhattan and composite metrics.

    Shows only Macro F1 comparison.
    """
    metrics_to_compare = ['manhattan', 'manhattan+chi-squared', 'manhattan+emd', 'manhattan+kullback-leibler']
    df_neutral = neutral_df[neutral_df['Metric'].isin(metrics_to_compare)]
    df_adjusted = adjusted_df[adjusted_df['Metric'].isin(metrics_to_compare)]

    if len(df_neutral) < len(metrics_to_compare) or len(df_adjusted) < len(metrics_to_compare):
        print(f"! Skipping plot D: Not enough data for Manhattan variations. Found Neutral: {len(df_neutral)} , Adjusted: {len(df_adjusted)} of {len(metrics_to_compare)} metrics.")
        return

    df_neutral = df_neutral.set_index('Metric').loc[metrics_to_compare]
    df_adjusted = df_adjusted.set_index('Metric').loc[metrics_to_compare]

    labels = [label.replace('+', '\n+') for label in df_neutral.index]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Macro F1 comparison ---
    bars_n = ax.bar(x - width/2, df_neutral['Macro_F1'], width, label='Neutral', color='lightblue')
    bars_a = ax.bar(x + width/2, df_adjusted['Macro_F1'], width, label='Weighted', color='salmon')
    ax.set_title('F1 — Neutral vs Weighted', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, max(df_neutral['Macro_F1'].max(), df_adjusted['Macro_F1'].max()) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # annotate
    for bar in list(bars_n) + list(bars_a):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(folder / 'D_adjusted_metrics_comparison.png', dpi=300)
    plt.close()


def main():
    # Create presentation folder
    presentation_folder = Path("Presentation_Results")
    presentation_folder.mkdir(exist_ok=True)

    # Load the data
    try:
        neutral_df_raw = pd.read_csv('stats_results_neutral.csv')
        adjusted_df_raw = pd.read_csv('stats_results_adjusted.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the results CSV files exist.")
        return

    # Merge dataframes to ensure metrics are aligned and only common metrics are used
    merged_df = pd.merge(neutral_df_raw, adjusted_df_raw, on='Metric', suffixes=('_neutral', '_adjusted'))

    # For operations that need separate dataframes, create them from the merged set
    common_metrics = merged_df['Metric']
    neutral_df = neutral_df_raw[neutral_df_raw['Metric'].isin(common_metrics)].reset_index(drop=True)
    adjusted_df = adjusted_df_raw[adjusted_df_raw['Metric'].isin(common_metrics)].reset_index(drop=True)

    # Add a column to distinguish datasets
    neutral_df['Dataset'] = 'Neutral'
    adjusted_df['Dataset'] = 'Adjusted'

    # Combine for comparison
    combined_df = pd.concat([neutral_df, adjusted_df])

    # Key metrics to highlight
    key_metrics = [
        'Macro_Precision', 'Macro_Recall', 'Macro_F1',
        'Macro_MAP', 'Macro_First_Tier',
        'Micro_Precision', 'Micro_Recall',
        'ROC_AUC', 'Accuracy'
    ]

    # ============================================================================
    # A-E. NEW PLOTS FOR PRESENTATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("GENERATING NEW PRESENTATION PLOTS...")
    print("=" * 80)
    plot_neutral_euclidean_vs_manhattan(neutral_df, presentation_folder)
    plot_manhattan_variations(neutral_df_raw, presentation_folder)
    plot_feature_weights(presentation_folder)
    plot_adjusted_metrics_comparison(neutral_df_raw, adjusted_df_raw, presentation_folder)
    prove_knn_euclidean_equivalence(neutral_df_raw, presentation_folder)  # ← MOVED HERE
    print("-" * 80)

    # ============================================================================
    # 1. Overall Performance Comparison - Key Metrics
    # ============================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Performance Comparison: Neutral vs Adjusted Weights', fontsize=16, fontweight='bold')

    metrics_to_plot = ['Macro_Precision', 'Macro_Recall', 'Macro_F1',
                       'Macro_MAP', 'Macro_First_Tier', 'ROC_AUC']

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Create bar plot
        x = np.arange(len(neutral_df))
        width = 0.35

        neutral_vals = neutral_df[metric].values
        adjusted_vals = adjusted_df[metric].values

        bars1 = ax.bar(x - width / 2, neutral_vals, width, label='Neutral', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width / 2, adjusted_vals, width, label='Adjusted', alpha=0.8, color='coral')

        ax.set_xlabel('Distance Metric')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(neutral_df['Metric'].values, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(presentation_folder / '1_overall_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # 2. Improvement Analysis - Percentage Change
    # ============================================================================
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics_for_improvement = ['Macro_Precision', 'Macro_Recall', 'Macro_F1',
                               'Macro_MAP', 'Macro_First_Tier', 'ROC_AUC']

    improvement_data = []
    for metric in metrics_for_improvement:
        for i, distance_metric in enumerate(neutral_df['Metric']):
            neutral_val = neutral_df.iloc[i][metric]
            adjusted_val = adjusted_df.iloc[i][metric]
            improvement = ((adjusted_val - neutral_val) / neutral_val) * 100 if neutral_val != 0 else 0
            improvement_data.append({
                'Distance Metric': distance_metric,
                'Performance Metric': metric.replace('_', ' '),
                'Improvement (%)': improvement
            })

    improvement_df = pd.DataFrame(improvement_data)
    pivot_improvement = improvement_df.pivot(index='Performance Metric',
                                             columns='Distance Metric',
                                             values='Improvement (%)')

    sns.heatmap(pivot_improvement, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement (%)'}, linewidths=0.5)
    plt.title('Percentage Improvement: Adjusted vs Neutral Weights', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Distance Metric', fontweight='bold')
    plt.ylabel('Performance Metric', fontweight='bold')
    plt.tight_layout()
    plt.savefig(presentation_folder / '2_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # 3. Best Performing Metrics Summary
    # ============================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Find best performing distance metric for each dataset
    best_neutral = neutral_df.loc[neutral_df['Macro_F1'].idxmax()]
    best_adjusted = adjusted_df.loc[adjusted_df['Macro_F1'].idxmax()]

    metrics_to_show = ['Macro_Precision', 'Macro_Recall', 'Macro_F1',
                       'Macro_MAP', 'Macro_First_Tier', 'ROC_AUC']

    # Neutral best
    ax1.barh(range(len(metrics_to_show)),
             [best_neutral[m] for m in metrics_to_show],
             color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(metrics_to_show)))
    ax1.set_yticklabels([m.replace('_', ' ') for m in metrics_to_show])
    ax1.set_xlabel('Score')
    ax1.set_title(f'Best Neutral: {best_neutral["Metric"]}', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, max(best_neutral[metrics_to_show].max(), best_adjusted[metrics_to_show].max()) * 1.1)
    for i, v in enumerate([best_neutral[m] for m in metrics_to_show]):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

    # Adjusted best
    ax2.barh(range(len(metrics_to_show)),
             [best_adjusted[m] for m in metrics_to_show],
             color='coral', alpha=0.8)
    ax2.set_yticks(range(len(metrics_to_show)))
    ax2.set_yticklabels([m.replace('_', ' ') for m in metrics_to_show])
    ax2.set_xlabel('Score')
    ax2.set_title(f'Best Adjusted: {best_adjusted["Metric"]}', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, max(best_neutral[metrics_to_show].max(), best_adjusted[metrics_to_show].max()) * 1.1)
    for i, v in enumerate([best_adjusted[m] for m in metrics_to_show]):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

    plt.suptitle('Best Performing Distance Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(presentation_folder / '3_best_performers.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # 4. Radar Chart Comparison (Best Methods)
    # ============================================================================
    from math import pi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(projection='polar'))

    categories = ['Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1',
                  'Macro\nMAP', 'First\nTier', 'ROC\nAUC']
    metrics_radar = ['Macro_Precision', 'Macro_Recall', 'Macro_F1',
                     'Macro_MAP', 'Macro_First_Tier', 'ROC_AUC']

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Determine a shared y-axis limit for better comparison
    max_val = combined_df[metrics_radar].max().max()
    y_limit = (max_val // 0.1 + 1) * 0.1  # Round up to next 0.1

    # Plot all methods for neutral
    ax1.set_theta_offset(pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=9)
    ax1.set_ylim(0, y_limit)
    ax1.set_title('Neutral Weights - All Methods', fontweight='bold', pad=20)
    ax1.grid(True)

    colors_neutral = plt.cm.Blues(np.linspace(0.4, 0.8, len(neutral_df)))
    for idx, row in neutral_df.iterrows():
        values = [row[m] for m in metrics_radar]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=row['Metric'], color=colors_neutral[idx])
        ax1.fill(angles, values, alpha=0.15, color=colors_neutral[idx])

    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # Plot all methods for adjusted
    ax2.set_theta_offset(pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=9)
    ax2.set_ylim(0, y_limit)
    ax2.set_title('Adjusted Weights - All Methods', fontweight='bold', pad=20)
    ax2.grid(True)

    colors_adjusted = plt.cm.Reds(np.linspace(0.4, 0.8, len(adjusted_df)))
    for idx, row in adjusted_df.iterrows():
        values = [row[m] for m in metrics_radar]
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=row['Metric'], color=colors_adjusted[idx])
        ax2.fill(angles, values, alpha=0.15, color=colors_adjusted[idx])

    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.suptitle('Radar Chart: Performance Across Key Metrics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(presentation_folder / '4_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # 5. Key Statistics Summary Table
    # ============================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create summary statistics
    summary_data = []
    for metric in neutral_df['Metric']:
        neutral_row = neutral_df[neutral_df['Metric'] == metric].iloc[0]
        adjusted_row = adjusted_df[adjusted_df['Metric'] == metric].iloc[0]

        f1_change = ((adjusted_row['Macro_F1'] - neutral_row['Macro_F1']) / neutral_row['Macro_F1'] * 100) if \
        neutral_row['Macro_F1'] != 0 else 0
        map_change = ((adjusted_row['Macro_MAP'] - neutral_row['Macro_MAP']) / neutral_row['Macro_MAP'] * 100) if \
        neutral_row['Macro_MAP'] != 0 else 0
        auc_change = ((adjusted_row['ROC_AUC'] - neutral_row['ROC_AUC']) / neutral_row['ROC_AUC'] * 100) if neutral_row[
                                                                                                                'ROC_AUC'] != 0 else 0

        summary_data.append([
            metric,
            f"{neutral_row['Macro_F1']:.4f}", f"{adjusted_row['Macro_F1']:.4f}", f"{f1_change:+.2f}%",
            f"{neutral_row['Macro_MAP']:.4f}", f"{adjusted_row['Macro_MAP']:.4f}", f"{map_change:+.2f}%",
            f"{neutral_row['ROC_AUC']:.4f}", f"{adjusted_row['ROC_AUC']:.4f}", f"{auc_change:+.2f}%"
        ])

    columns = ['Distance\nMetric',
               'Neutral\nMacro F1', 'Adjusted\nMacro F1', 'F1\nChange',
               'Neutral\nMAP', 'Adjusted\nMAP', 'MAP\nChange',
               'Neutral\nROC AUC', 'Adjusted\nROC AUC', 'AUC\nChange']

    table = ax.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code the headers
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code improvement columns
    for i in range(1, len(summary_data) + 1):
        # F1 Change
        change_val = float(summary_data[i - 1][3].strip('%+'))
        if change_val > 0:
            table[(i, 3)].set_facecolor('#D4EDDA')  # Light green
        elif change_val < 0:
            table[(i, 3)].set_facecolor('#F8D7DA')  # Light red

        # MAP Change
        change_val = float(summary_data[i - 1][6].strip('%+'))
        if change_val > 0:
            table[(i, 6)].set_facecolor('#D4EDDA')
        elif change_val < 0:
            table[(i, 6)].set_facecolor('#F8D7DA')

        # AUC Change
        change_val = float(summary_data[i - 1][9].strip('%+'))
        if change_val > 0:
            table[(i, 9)].set_facecolor('#D4EDDA')
        elif change_val < 0:
            table[(i, 9)].set_facecolor('#F8D7DA')

    plt.title('Performance Summary: Neutral vs Adjusted Weights',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(presentation_folder / '5_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # 6. Average Performance Across All Metrics
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    avg_neutral = neutral_df[key_metrics].mean()
    avg_adjusted = adjusted_df[key_metrics].mean()

    x = np.arange(len(key_metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, avg_neutral, width, label='Neutral', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width / 2, avg_adjusted, width, label='Adjusted', alpha=0.8, color='coral')

    ax.set_ylabel('Average Score', fontweight='bold')
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_title('Average Performance Across All Distance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in key_metrics], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}',
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(presentation_folder / '6_average_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # 7. Generate Summary Report
    # ============================================================================
    report_path = presentation_folder / 'PRESENTATION_SUMMARY.txt'
    # Specify UTF-8 encoding to handle special characters like '→' on Windows
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIMEDIA RETRIEVAL: NEUTRAL VS ADJUSTED WEIGHTS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n\n")

        # Overall improvement
        avg_improvement = {}
        for metric in key_metrics:
            neutral_avg = neutral_df[metric].mean()
            adjusted_avg = adjusted_df[metric].mean()
            improvement = ((adjusted_avg - neutral_avg) / neutral_avg) * 100 if neutral_avg != 0 else 0
            avg_improvement[metric] = improvement

        f.write("1. OVERALL PERFORMANCE IMPROVEMENT (ON COMMON METRICS):\n\n")
        for metric, improvement in avg_improvement.items():
            f.write(f"   {metric:25s}: {improvement:+6.2f}%\n")

        f.write("\n\n2. BEST PERFORMING METHODS:\n\n")
        f.write("   NEUTRAL WEIGHTS:\n")
        best_neutral_idx = neutral_df['Macro_F1'].idxmax()
        best_neutral = neutral_df.iloc[best_neutral_idx]
        f.write(f"   - Method: {best_neutral['Metric']}\n")
        f.write(f"   - Macro F1: {best_neutral['Macro_F1']:.4f}\n")
        f.write(f"   - Macro MAP: {best_neutral['Macro_MAP']:.4f}\n")
        f.write(f"   - ROC AUC: {best_neutral['ROC_AUC']:.4f}\n\n")

        f.write("   ADJUSTED WEIGHTS:\n")
        best_adjusted_idx = adjusted_df['Macro_F1'].idxmax()
        best_adjusted = adjusted_df.iloc[best_adjusted_idx]
        f.write(f"   - Method: {best_adjusted['Metric']}\n")
        f.write(f"   - Macro F1: {best_adjusted['Macro_F1']:.4f}\n")
        f.write(f"   - Macro MAP: {best_adjusted['Macro_MAP']:.4f}\n")
        f.write(f"   - ROC AUC: {best_adjusted['ROC_AUC']:.4f}\n\n")

        f.write("\n3. BIGGEST IMPROVEMENTS BY DISTANCE METRIC:\n\n")
        for i, metric in enumerate(neutral_df['Metric']):
            neutral_f1 = neutral_df.iloc[i]['Macro_F1']
            adjusted_f1 = adjusted_df.iloc[i]['Macro_F1']
            improvement = ((adjusted_f1 - neutral_f1) / neutral_f1) * 100 if neutral_f1 != 0 else 0
            f.write(f"   {metric:30s}: {improvement:+6.2f}% (F1: {neutral_f1:.4f} → {adjusted_f1:.4f})\n")

        f.write("\n\n4. STATISTICAL SUMMARY:\n\n")
        f.write(f"   Number of Queries: {neutral_df['Num_Queries'].iloc[0]}\n")
        f.write(f"   Number of Categories: {neutral_df['Num_Categories'].iloc[0]}\n")
        f.write(f"   Distance Metrics Tested: {len(neutral_df)}\n\n")

        f.write("\n5. KEY RECOMMENDATIONS:\n\n")
        best_overall = adjusted_df.loc[adjusted_df['Macro_F1'].idxmax(), 'Metric']
        f.write(f"   - Use ADJUSTED WEIGHTS for better retrieval performance\n")
        f.write(f"   - Best overall method: {best_overall}\n")
        f.write(f"   - Average improvement across all metrics: {np.mean(list(avg_improvement.values())):+.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("VISUALIZATION FILES GENERATED:\n")
        f.write("-" * 80 + "\n")
        f.write("   • A_neutral_euclidean_vs_manhattan.png - Comparison of neutral Euclidean and Manhattan.\n")
        f.write("   • B_manhattan_variations.png - Comparison of neutral and adjusted Manhattan metrics.\n")
        f.write("   • C_feature_weights.png - Overview of normalized feature weights.\n")
        f.write("   • D_adjusted_metrics_comparison.png - Comparison of all adjusted metrics.\n")
        f.write("   1. 1_overall_performance_comparison.png - Bar charts of all metrics\n")
        f.write("   2. 2_improvement_heatmap.png - Percentage improvements heatmap\n")
        f.write("   3. 3_best_performers.png - Best methods comparison\n")
        f.write("   4. 4_radar_comparison.png - Radar charts for all methods\n")
        f.write("   5. 5_summary_table.png - Detailed comparison table\n")
        f.write("   6. 6_average_performance.png - Average scores across methods\n")
        f.write("=" * 80 + "\n")

    prove_knn_euclidean_equivalence(neutral_df, presentation_folder)

    print("\n" + "=" * 80)
    print("GENERATING NEW PRESENTATION PLOTS...")
    print("=" * 80)
    plot_neutral_euclidean_vs_manhattan(neutral_df, presentation_folder)
    plot_manhattan_variations(neutral_df_raw, presentation_folder)
    plot_feature_weights(presentation_folder)
    plot_adjusted_metrics_comparison(neutral_df_raw, adjusted_df_raw, presentation_folder)
    prove_knn_euclidean_equivalence(neutral_df_raw, presentation_folder)
    plot_weighted_results_table(presentation_folder)  # ← ADD THIS LINE
    print("-" * 80)

    print("\n" + "=" * 80)
    print("ALL PRESENTATION MATERIALS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nLocation: {presentation_folder.absolute()}")
    print("\nFiles created:")
    print("  • 10 visualization PNG files (300 DPI, print-ready)")
    print("  • 1 comprehensive summary report")
    print("\nThese materials are ready for your presentation!")


if __name__ == "__main__":
    main()
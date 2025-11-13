"""
RESULTS TABLE GENERATOR
This file creates formatted tables showing performance metrics.
It highlights best values in green and worst values in red.
It uses color coding to make results easy to understand.
The tables are publication-ready for reports and presentations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
# Data
data = {
    'Metric': ['MAP', 'F1', 'AUC', '1st Tier Acc.', 'Avg. Last Rank'],
    'Manhattan': [0.1187, 0.1954, 0.5553, 0.5079, 1.68],
    '+Kullback-Leibler': [0.1151, 0.1896, 0.5665, 0.4846, 1.63],
    '+Chi-Squared': [0.1128, 0.1884, 0.5435, 0.4814, 1.61],
    '+EMD': [0.0928, 0.1673, 0.5383, 0.4516, 1.20],
    'Euclidean': [0.0918, 0.1627, 0.5385, 0.4571, 1.27]
}

df = pd.DataFrame(data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

# Orange/brown color scheme like the reference
ORANGE_HEADER = '#cf5230'  # Orange-brown for headers
WHITE = '#FFFFFF'
LIGHT_GREEN = '#D4EDDA'    # Soft green for best values
LIGHT_RED = '#F8D7DA'      # Soft red for worst values

# Create color mapping for data rows only (not header)
cell_colors = []

for idx in range(len(df)):
    row_colors = []

    # First column (metric name) - orange header color
    row_colors.append(ORANGE_HEADER)

    # Get values for this metric (excluding the metric name)
    values = df.iloc[idx, 1:].values.astype(float)

    # Find min and max for this metric
    min_val = values.min()
    max_val = values.max()

    # Create colors for each value cell
    for val in values:
        if val == max_val:
            row_colors.append(LIGHT_GREEN)  # Best value: soft green
        elif val == min_val:
            row_colors.append(LIGHT_RED)    # Worst value: soft red
        else:
            row_colors.append(WHITE)

    cell_colors.append(row_colors)

# Create column header colors
col_colors = [ORANGE_HEADER] * len(df.columns)

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns,
                cellLoc='center', loc='center',
                cellColours=cell_colors,
                colColours=col_colors)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Style header row - white text on orange
for i in range(len(df.columns)):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white', size=13)
    cell.set_edgecolor('white')
    cell.set_linewidth(2)

# Style all data cells
for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        cell = table[(i, j)]
        cell.set_edgecolor('white')
        cell.set_linewidth(2)

        if j == 0:
            # Metric names (first column) - white text, left aligned
            cell.set_text_props(weight='bold', color='white', ha='left', size=12)
        else:
            # Value cells - dark text, bold
            cell.set_text_props(weight='bold', color='#333333', size=12)

plt.title('Performance Comparison of Distance Metrics',
          fontsize=15, fontweight='bold', pad=20, color='#333333')

plt.tight_layout()
plt.savefig('performance_comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Table saved as 'performance_comparison_table.png'")
plt.show()

# Also print a summary
print("\n" + "="*80)
print("SUMMARY - Best Performing Metric per Evaluation Criterion")
print("="*80)
for idx in range(len(df)):
    metric_name = df.iloc[idx, 0]
    values = df.iloc[idx, 1:].values.astype(float)
    best_idx = values.argmax()
    best_method = df.columns[best_idx + 1]
    best_value = values[best_idx]
    print(f"{metric_name:20s} → {best_method:35s} ({best_value:.4f})")
print("="*80)

'''
import pandas as pd
import matplotlib.pyplot as plt

# Data - sorted by weight (descending)
data = {
    'Feature': ['A3', 'D1', 'Eccentricity', 'D2', 'D3', 'Convexity',
                'D4', 'Sphericity', 'Surface area', 'Rectangularity', 'Diameter'],
    'Weight': [0.31, 0.21, 0.15, 0.11, 0.09, 0.06, 0.02, 0.02, 0.01, 0.01, 0.01]
}

df = pd.DataFrame(data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 7))
ax.axis('tight')
ax.axis('off')

# Orange/brown color scheme
ORANGE_HEADER = '#cf5230'
WHITE = '#FFFFFF'

# Create color mapping - all white cells
cell_colors = [[WHITE, WHITE] for _ in range(len(df))]

# Column header colors
col_colors = [ORANGE_HEADER, ORANGE_HEADER]

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns,
                 cellLoc='center', loc='center',
                 cellColours=cell_colors,
                 colColours=col_colors)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header row
for i in range(len(df.columns)):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white', size=13)
    cell.set_edgecolor('#333333')
    cell.set_linewidth(1.5)

# Style data cells
for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        cell = table[(i, j)]
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.5)

        if j == 0:
            # Feature names - left aligned
            cell.set_text_props(weight='bold', color='#333333', ha='left', size=12)
        else:
            # Weight values - bold
            cell.set_text_props(weight='bold', color='#333333', size=12)

plt.title('Feature Weights for Distance Calculation',
          fontsize=15, fontweight='bold', pad=20, color='#333333')

plt.tight_layout()
plt.savefig('feature_weights_table.png', dpi=300, bbox_inches='tight')
print("✓ Table saved as 'feature_weights_table.png'")
plt.show()

# Summary
print("\n" + "=" * 60)
print("FEATURE WEIGHTS SUMMARY")
print("=" * 60)
print(f"Highest weight: {df.iloc[0, 0]:15s} → {df.iloc[0, 1]:.2f}")
print(f"Lowest weight:  {df.iloc[-1, 0]:15s} → {df.iloc[-1, 1]:.2f}")
print(f"Total weight:   {df['Weight'].sum():.2f}")
print("=" * 60)
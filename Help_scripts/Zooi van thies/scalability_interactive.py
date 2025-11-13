"""
INTERACTIVE T-SNE DATA EXPORTER
This file creates t-SNE visualizations and saves them in multiple formats.
It generates 2D embeddings of shape features for visualization.
It exports data as CSV and JSON for use in interactive tools.
This creates the data files used by the t-SNE GUI viewer.
"""

import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def compute_tsne(X_scaled, n_components=2, perplexity=10, max_iter=3000, random_state=42):
    """Compute t-SNE embedding for the given scaled features."""
    print(f"Running t-SNE for 2D embedding with perplexity={perplexity} and {max_iter} iterations... (this may take a moment)")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    X_2d = tsne.fit_transform(X_scaled)
    print("t-SNE completed.")
    return X_2d

def save_plot_data(features_df, X_2d, output_base_name):
    """Save plot data in structured formats for interactive GUI development."""
    print(f"Saving plot data to structured formats...")
    
    categories = features_df['Category'].astype(str)
    objects = features_df['Object'].astype(str)
    
    # Create structured data
    plot_data = []
    for i in range(len(features_df)):
        plot_data.append({
            'category': categories.iloc[i],
            'object_name': objects.iloc[i],
            'x': float(X_2d[i, 0]),
            'y': float(X_2d[i, 1])
        })
    
    # Save as CSV for easy loading
    csv_filename = f"{output_base_name}.csv"
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(csv_filename, index=False)
    
    print(f"Plot data saved to:")
    print(f"  - CSV format: {csv_filename}")
    
    return plot_data

def save_full_tsne_map(features_df, X_2d, output_filename):
    """Saves the complete t-SNE map to a PNG file."""
    print(f"Saving full t-SNE map to `{output_filename}`...")
    categories = features_df['Category'].astype(str)
    unique_categories = sorted(list(dict.fromkeys(categories.tolist())))
    n_cat = len(unique_categories)

    try:
        import seaborn as sns
        palette = sns.color_palette(n_colors=n_cat)
    except ImportError:
        cmap_name = 'tab20' if n_cat > 10 else 'tab10'
        cmap = plt.cm.get_cmap(cmap_name, n_cat)
        palette = [cmap(i) for i in range(n_cat)]

    category_color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}

    plt.figure(figsize=(14, 8))

    for cat in unique_categories:
        mask = (categories == cat).values
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cat, color=category_color_map[cat], alpha=0.7, s=50, edgecolors='none')

    plt.title('Full 2D t-SNE Map of All Items')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=1.2, ncol=2, fontsize='small')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Map saved to `{output_filename}`.")


if __name__ == '__main__':
    csv_file = '../../Feature-matrix/OldFeatureSets/all_features_modified.csv'

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"`{csv_file}` not found. Ensure the CSV file is in the directory.")

    all_features_df = pd.read_csv(csv_file)
    if 'Object' not in all_features_df.columns or 'Category' not in all_features_df.columns:
        raise ValueError("CSV must contain `Object` and `Category` columns.")

    print(f"Loaded {len(all_features_df)} rows from `{csv_file}`.")

    feat_cols = [c for c in all_features_df.columns if c not in ('Object', 'Category')]
    if not feat_cols:
        raise ValueError("No feature columns found in DataFrame.")

    X = all_features_df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Scale high-dimensional features
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Set t-SNE perplexity
    perplexity_val = 20

    # 3. Compute t-SNE for visualization
    X_2d = compute_tsne(X_scaled, perplexity=perplexity_val, max_iter=100000)

    # 4. Save the plot data in structured formats for interactive GUI
    plot_data = save_plot_data(all_features_df, X_2d, 'tsne_plot_data')
    
    # 5. Save the full t-SNE map to a file
    save_full_tsne_map(all_features_df, X_2d, 'tsne_full_map.png')
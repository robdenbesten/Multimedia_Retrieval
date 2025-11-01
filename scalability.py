import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

def compute_tsne(X_scaled, n_components=2, perplexity=10, max_iter=3000, random_state=42):
    """Compute t-SNE embedding for the given scaled features."""
    print(f"Running t-SNE for 2D embedding with perplexity={perplexity} and {max_iter} iterations... (this may take a moment)")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    X_2d = tsne.fit_transform(X_scaled)
    print("t-SNE completed.")
    return X_2d

def find_nearest_neighbours_knn(features_df, nn_model, query_index, k, X_scaled):
    """Finds nearest neighbors using a k-NN model on high-dimensional data."""
    # Query for k+1 neighbors to account for the query item itself
    n = min(k + 1, len(features_df))
    query_vec = X_scaled[query_index].reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=n)

    # Exclude the first result (the query item)
    indices = indices.flatten()[1:]
    distances = distances.flatten()[1:]

    nearest_df = features_df.iloc[indices][['Object', 'Category']].copy()
    nearest_df = nearest_df.reset_index().rename(columns={'index': 'Index'})
    nearest_df['Distance'] = distances
    return nearest_df, indices


def save_full_tsne_map(features_df, X_2d, output_filename):
    """Saves the complete t-SNE map to a PNG file."""
    print(f"Saving full t-SNE map to `{output_filename}`...")
    categories = features_df['Category'].astype(str)
    unique_categories = list(dict.fromkeys(categories.tolist()))
    n_cat = len(unique_categories)

    try:
        import seaborn as sns
        palette = sns.color_palette(n_colors=n_cat)
    except Exception:
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
    print("Map saved.")


def visualize_neighbourhoods_2d(features_df, query_index, neighbor_indices, X_2d, show=True):
    """Visualizes the query and its neighbors on a 2D embedding."""
    categories = features_df['Category'].astype(str)
    unique_categories = list(dict.fromkeys(categories.tolist()))
    n_cat = len(unique_categories)

    try:
        import seaborn as sns
        palette = sns.color_palette(n_colors=n_cat)
    except Exception:
        cmap_name = 'tab20' if n_cat > 10 else 'tab10'
        cmap = plt.cm.get_cmap(cmap_name, n_cat)
        palette = [cmap(i) for i in range(n_cat)]

    category_color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}

    plt.figure(figsize=(14, 8))  # Increased width to accommodate legend

    for cat in unique_categories:
        mask = (categories == cat).values
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cat, color=category_color_map[cat], alpha=0.7, s=50, edgecolors='none')

    for ni in neighbor_indices:
        cat = str(features_df.iloc[ni]['Category'])
        pt = X_2d[ni]
        plt.scatter(pt[0], pt[1], color=category_color_map.get(cat, 'gray'), s=140, edgecolor='black', linewidth=1.2, zorder=5)
        plt.annotate(f"{features_df.iloc[ni]['Object']} ({cat})", (pt[0], pt[1]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=category_color_map.get(cat, 'black'))

    q_pt = X_2d[query_index]
    q_name = features_df.iloc[query_index]['Object']
    q_cat = str(features_df.iloc[query_index]['Category'])
    plt.scatter(q_pt[0], q_pt[1], color='red', s=200, edgecolor='black', marker='*', zorder=6, label='Query')
    plt.annotate(f"{q_name} ({q_cat})", (q_pt[0], q_pt[1]), textcoords="offset points", xytext=(8, 8), fontsize=9, weight='bold', color='red')

    plt.title('2D t-SNE Visualization of Item Neighbourhoods')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=1.2, ncol=2, fontsize='small')  # Legend in 2 columns with smaller font
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == '__main__':
    csv_file = 'Feature-matrix/all_features_modified.csv'

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"`{csv_file}` not found. Ensure the CSV file is in the directory.")

    all_features_df = pd.read_csv(csv_file)
    if 'Object' not in all_features_df.columns or 'Category' not in all_features_df.columns:
        raise ValueError("CSV must contain `Object` and `Category` columns.")

    print(f"Loaded {len(all_features_df)} rows from `{csv_file}`.")
    print("Sample (index, Object, Category):")
    print(all_features_df[['Object', 'Category']].head(20).reset_index().to_string(index=False))

    feat_cols = [c for c in all_features_df.columns if c not in ('Object', 'Category')]
    if not feat_cols:
        raise ValueError("No feature columns found in DataFrame.")

    X = all_features_df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Scale high-dimensional features
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Build k-NN model on the high-dimensional scaled data
    nn_model = NearestNeighbors(n_neighbors=11, algorithm='ball_tree')
    nn_model.fit(X_scaled)
    print("k-NN model built on high-dimensional scaled features.")

    # Get t-SNE perplexity from user
    perplexity_val = 20  # Default value
    while True:
        try:
            p_input = input(f"Enter t-SNE perplexity (default: {perplexity_val}, recommended 5-50): ").strip()
            if not p_input:
                break  # Use default
            perplexity_val = int(p_input)
            if perplexity_val > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 3. Compute t-SNE for visualization only
    X_2d = compute_tsne(X_scaled, perplexity=perplexity_val, max_iter=100000)

    # Ask user if they want to save the full map
    save_choice = input("Save the full t-SNE map to a PNG file? (y/n): ").strip().lower()
    if save_choice in ('y', 'yes'):
        save_full_tsne_map(all_features_df, X_2d, 'tsne_full_map40.png')

    while True:
        user_input = input("\nEnter object index or object name (or `q` to quit): ").strip()
        if user_input.lower() in ('q', 'quit', 'exit'):
            print("Exiting.")
            break

        query_index = None
        if user_input.isdigit():
            idx = int(user_input)
            if 0 <= idx < len(all_features_df):
                query_index = idx
            else:
                print("Index out of range.")
                continue
        else:
            matches = all_features_df.index[all_features_df['Object'] == user_input].tolist()
            if len(matches) == 0:
                print("Object name not found. Try exact name or use an index.")
                continue
            query_index = matches[0]

        q_name = all_features_df.iloc[query_index]['Object']
        q_cat = all_features_df.iloc[query_index]['Category']
        print(f"Querying `{q_name}` (index {query_index}) in category `{q_cat}`...")

        # 4. Find neighbors using the k-NN model on high-dimensional data
        k_neighbors = 10
        nearest_df, neighbor_indices = find_nearest_neighbours_knn(all_features_df, nn_model, query_index, k=k_neighbors, X_scaled=X_scaled)
        print(f"\nTop {k_neighbors} nearest neighbours (index, Object, Category, Distance):")
        print(nearest_df.to_string(index=False))

        viz_choice = input("Show visualization? (y/n): ").strip().lower()
        if viz_choice in ('y', 'yes'):
            print("Opening visualization window (close it to continue)...")
            # 5. Visualize the results using the pre-computed 2D t-SNE data
            visualize_neighbourhoods_2d(all_features_df, query_index, neighbor_indices, X_2d=X_2d, show=True)
        else:
            print("Skipping visualization.")
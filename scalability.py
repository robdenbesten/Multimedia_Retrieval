import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

def compute_tsne(X_scaled, n_components=2, perplexity=30, max_iter=3000, random_state=42):
    """Compute t-SNE embedding for the given scaled features."""
    print("Running t-SNE for 2D embedding... (this may take a moment)")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    X_2d = tsne.fit_transform(X_scaled)
    print("t-SNE completed.")
    return X_2d

def compute_isomap(X_scaled, n_components=2, n_neighbors=5, max_iter=300, random_state=42):
    """Compute Isomap embedding for the given scaled features."""
    print("Running Isomap for 2D embedding... (this may take a moment)")
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, max_iter=max_iter, random_state=random_state)
    X_2d = isomap.fit_transform(X_scaled)
    print("Isomap completed.")
    return X_2d

def find_nearest_neighbours_knn(features_df, nn_model, query_index, k, X_2d):
    n = min(k + 1, len(features_df))
    query_vec = X_2d[query_index].reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=n)
    indices = indices.flatten()[1:]
    distances = distances.flatten()[1:]
    nearest_df = features_df.iloc[indices][['Object', 'Category']].copy()
    nearest_df = nearest_df.reset_index().rename(columns={'index': 'Index'})
    nearest_df['Distance'] = distances
    return nearest_df, indices


def visualize_neighbourhoods_2d(features_df, query_index, neighbor_indices, X_2d, show=True):
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
    csv_file = 'ShapeDatabase_INFOMR-master/all_features.csv'

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

    X_scaled = StandardScaler().fit_transform(X)

    X_2d = compute_tsne(X_scaled)

    nn_model = NearestNeighbors(n_neighbors=11, algorithm='ball_tree')
    nn_model.fit(X_2d)
    print("k-NN model built on t-SNE map.")

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

        nearest_df, neighbor_indices = find_nearest_neighbours_knn(all_features_df, nn_model, query_index, k=10, X_2d=X_2d)
        print("\nNearest neighbours (index, Object, Category, Distance):")
        print(nearest_df.to_string(index=False))

        viz_choice = input("Show visualization? (y/n): ").strip().lower()
        if viz_choice in ('y', 'yes'):
            print("Opening visualization window (close it to continue)...")
            visualize_neighbourhoods_2d(all_features_df, query_index, neighbor_indices, X_2d=X_2d, show=True)
        else:
            print("Skipping visualization.")
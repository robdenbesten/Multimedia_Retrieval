import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px


def compute_tsne(X_scaled, perplexity, max_iter):
    """Compute t-SNE embedding for the given scaled features."""
    print(f"Running t-SNE with perplexity={perplexity} and {max_iter} iterations...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X_scaled)
    print("t-SNE completed.")
    return X_2d


def create_interactive_plot(features_df, X_2d):
    """Creates and shows an interactive t-SNE plot using Plotly."""
    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame(X_2d, columns=['tsne_1', 'tsne_2'])
    plot_df['Object'] = features_df['Object']
    plot_df['Category'] = features_df['Category'].astype(str)

    print("Generating interactive plot...")
    fig = px.scatter(
        plot_df,
        x='tsne_1',
        y='tsne_2',
        color='Category',
        hover_name='Object',
        hover_data={'Category': True, 'tsne_1': False, 'tsne_2': False},
        title='Interactive 2D t-SNE Map'
    )

    fig.update_traces(marker=dict(size=8,
                                  opacity=0.8,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(
        legend_title_text='Click to toggle categories'
    )

    # Show the plot (opens in a browser)
    fig.show()
    print("Plot has been opened in your browser.")


if __name__ == '__main__':
    csv_file = 'Feature-matrix/all_features_modified.csv'

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"`{csv_file}` not found. Ensure the CSV file is in the correct directory.")

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

    # 2. Get t-SNE perplexity from user
    perplexity_val = 20  # Default value

    # 3. Compute t-SNE for visualization
    X_2d = compute_tsne(X_scaled, perplexity=perplexity_val, max_iter=5000)

    # 4. Create and display the interactive plot
    create_interactive_plot(all_features_df, X_2d)
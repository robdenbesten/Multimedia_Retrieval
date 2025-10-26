import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
    from annoy import AnnoyIndex
except Exception as e:
    raise ImportError("Annoy is required. Install with: pip install annoy") from e


def convert_json_to_csv(json_path, csv_path):
    def sanitize_name(name: str) -> str:
        return str(name).replace(' ', '_').replace('.', '__').replace('-', '_')

    def parse_object_key(key: str):
        parts = re.split(r'[\\/]', key, maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, key

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"convert_json_to_csv: `{json_path}` not found.")
        return False
    except Exception as e:
        print(f"convert_json_to_csv: failed to load JSON: {e}")
        return False

    if not isinstance(data, dict) or not data:
        print("convert_json_to_csv: JSON is empty or not a dict of objects.")
        return False

    descriptor_max_len = {}
    rows = []

    def collect_descriptors(prefix, val, out):
        if isinstance(val, list):
            out[prefix] = val
        elif isinstance(val, dict):
            for k, v in val.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                collect_descriptors(new_prefix, v, out)

    for obj_key, details in data.items():
        parsed_cat, parsed_name = parse_object_key(obj_key)
        descriptors_map = {}
        category = ''
        if isinstance(details, dict):
            category = str(details.get('Category', details.get('category', '') or '')).strip()
            for key, val in details.items():
                if key.lower() in ('category', 'class', 'label'):
                    continue
                collect_descriptors(key, val, descriptors_map)
            if not descriptors_map and 'features' in details and isinstance(details['features'], list):
                descriptors_map['features'] = details['features']
        else:
            if isinstance(details, list):
                descriptors_map['features'] = details

        if (not category) and parsed_cat:
            category = parsed_cat

        object_name = parsed_name if parsed_name else obj_key

        for desc_name, vec in descriptors_map.items():
            if isinstance(vec, list):
                descriptor_max_len[desc_name] = max(descriptor_max_len.get(desc_name, 0), len(vec))

        rows.append({'ObjectName': object_name, 'Category': category, '_descriptors': descriptors_map})

    if not rows:
        print("convert_json_to_csv: no objects with descriptors found.")
        return False

    all_descriptor_names = sorted(descriptor_max_len.keys())
    col_names = ['ObjectName', 'Category']
    descriptor_columns = []
    for desc in all_descriptor_names:
        sanitized = sanitize_name(desc)
        dim_count = descriptor_max_len[desc]
        for i in range(1, dim_count + 1):
            col = f"{sanitized}__{i}"
            descriptor_columns.append((desc, i - 1, col))
            col_names.append(col)

    df_rows = []
    for r in rows:
        base = {'ObjectName': r['ObjectName'], 'Category': r['Category']}
        descs = r.get('_descriptors', {})
        for desc_name, idx, col in descriptor_columns:
            vec = descs.get(desc_name)
            if isinstance(vec, list) and idx < len(vec):
                try:
                    base[col] = float(vec[idx])
                except Exception:
                    base[col] = np.nan
            else:
                base[col] = np.nan
        df_rows.append(base)

    try:
        df = pd.DataFrame(df_rows, columns=col_names)
        df.to_csv(csv_path, index=False)
        if os.path.exists(csv_path):
            print(f"convert_json_to_csv: created `{csv_path}` with {len(df)} rows and {len(df.columns)} columns.")
            return True
        else:
            print(f"convert_json_to_csv: failed to write `{csv_path}`.")
            return False
    except Exception as e:
        print(f"convert_json_to_csv: error writing CSV: {e}")
        return False


def build_annoy_index(features_df, n_trees=10, metric='euclidean'):
    feat_cols = [c for c in features_df.columns if c not in ('ObjectName', 'Category')]
    if not feat_cols:
        raise ValueError("No feature columns found in DataFrame (need columns besides `ObjectName` and `Category`).")

    X = features_df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    dim = X_scaled.shape[1]
    annoy_index = AnnoyIndex(dim, metric)
    for i, vec in enumerate(X_scaled):
        annoy_index.add_item(i, vec.tolist())

    annoy_index.build(n_trees)
    return annoy_index, X_scaled, scaler, feat_cols


def find_nearest_neighbours_ann(features_df, annoy_index, query_index, k=10, X_scaled=None):
    n = min(k + 1, len(features_df))
    indices, distances = annoy_index.get_nns_by_item(query_index, n, include_distances=True)
    # remove query itself (first result)
    if len(indices) > 0 and indices[0] == query_index:
        indices = indices[1:]
        distances = distances[1:]
    neighbor_indices = indices
    neighbor_distances = distances

    nearest_df = features_df.iloc[neighbor_indices][['ObjectName', 'Category']].copy()
    nearest_df = nearest_df.reset_index().rename(columns={'index': 'Index'})
    nearest_df['Distance'] = neighbor_distances

    return nearest_df, neighbor_indices, X_scaled


def visualize_neighbourhoods(features_df, query_index, neighbor_indices, X_scaled=None, show=True):
    feat_cols = [c for c in features_df.columns if c not in ('ObjectName', 'Category')]
    if X_scaled is None:
        X = features_df[feat_cols].values.astype(float)
        X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

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

    plt.figure(figsize=(12, 8))

    for cat in unique_categories:
        mask = (categories == cat).values
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=cat,
            color=category_color_map[cat],
            alpha=0.7,
            s=50,
            edgecolors='none'
        )

    for ni in neighbor_indices:
        cat = str(features_df.iloc[ni]['Category'])
        pt = X_2d[ni]
        plt.scatter(pt[0], pt[1],
                    color=category_color_map.get(cat, 'gray'),
                    s=140,
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=5)
        plt.annotate(f"{features_df.iloc[ni]['ObjectName']} ({cat})",
                     (pt[0], pt[1]),
                     textcoords="offset points",
                     xytext=(6, 4),
                     fontsize=8,
                     color=category_color_map.get(cat, 'black'))

    q_pt = X_2d[query_index]
    q_name = features_df.iloc[query_index]['ObjectName']
    q_cat = str(features_df.iloc[query_index]['Category'])
    plt.scatter(q_pt[0], q_pt[1],
                color='red',
                s=200,
                edgecolor='black',
                marker='*',
                zorder=6,
                label='Query')
    plt.annotate(f"{q_name} ({q_cat})",
                 (q_pt[0], q_pt[1]),
                 textcoords="offset points",
                 xytext=(8, 8),
                 fontsize=9,
                 weight='bold',
                 color='red')

    plt.title('2D Visualization of Item Neighbourhoods (Annoy ANN)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(markerscale=1.2)
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == '__main__':
    json_file = 'ShapeDatabase_INFOMR-master/features.json'
    csv_file = 'features2.csv'

    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for `{json_file}` and `{csv_file}` in the current directory.")

    if not os.path.exists(csv_file):
        print(f"`{csv_file}` not found.")
        if os.path.exists(json_file):
            print(f"Found `{json_file}`, attempting conversion...")
            ok = convert_json_to_csv(json_file, csv_file)
            if not ok:
                raise FileNotFoundError(f"Conversion failed; `{csv_file}` was not created.")
        else:
            raise FileNotFoundError(f"Neither `{csv_file}` nor `{json_file}` were found. Place one of them in the directory shown above.")

    all_features_df = pd.read_csv(csv_file)
    if 'ObjectName' not in all_features_df.columns or 'Category' not in all_features_df.columns:
        raise ValueError("`features.csv` must contain `ObjectName` and `Category` columns.")

    print(f"Loaded {len(all_features_df)} rows from `{csv_file}`.")
    print("Sample (index, ObjectName, Category):")
    print(all_features_df[['ObjectName', 'Category']].head(20).reset_index().to_string(index=False))

    annoy_index, X_scaled, scaler, feat_cols = build_annoy_index(all_features_df, n_trees=20, metric='euclidean')

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
            matches = all_features_df.index[all_features_df['ObjectName'] == user_input].tolist()
            if len(matches) == 0:
                print("Object name not found. Try exact name or use an index.")
                continue
            query_index = matches[0]

        q_name = all_features_df.iloc[query_index]['ObjectName']
        q_cat = all_features_df.iloc[query_index]['Category']
        print(f"Querying `{q_name}` (index {query_index}) in category `{q_cat}`...")

        nearest_df, neighbor_indices, _ = find_nearest_neighbours_ann(all_features_df, annoy_index, query_index, k=10, X_scaled=X_scaled)
        print("\nNearest neighbours (index, ObjectName, Category, Distance):")
        print(nearest_df.to_string(index=False))

        viz_choice = input("Show visualization? (y/n): ").strip().lower()
        if viz_choice in ('y', 'yes'):
            print("Opening visualization window (close it to continue)...")
            visualize_neighbourhoods(all_features_df, query_index, neighbor_indices, X_scaled=X_scaled, show=True)
        else:
            print("Skipping visualization.")
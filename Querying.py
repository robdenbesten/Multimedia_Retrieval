"""
SHAPE SEARCH ENGINE
This file finds similar 3D shapes based on their features.
It uses different distance measures (Euclidean, Manhattan, Chi-squared, etc.) to compare shapes.
It applies weights to different features to prioritize important characteristics.
It can use KNN (k-nearest neighbors) to quickly find the most similar shapes.
The search results are ranked by similarity from most to least similar.
"""

import os
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Feature & Distance Constants
# -----------------------------
HIST_KEYS = ['A3', 'D1', 'D2', 'D3', 'D4']
SCALAR_KEYS = ['Surface area', 'Sphericity', 'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']
FEATURE_GROUP_ORDER = HIST_KEYS + SCALAR_KEYS
HIST_BINS = 20
EPS = 1e-10

WEIGHTING_METHOD = 'feature'  # Options: 'feature' or 'neutral'

# -----------------------------
# Default Configuration
# -----------------------------
MANUAL_WEIGHTS = {
    'A3': 9, 'D1': 6.0, 'D2': 3.1, 'D3': 2.5, 'D4': 0.5,
    'Surface area': 0.3, 'Sphericity': 0.5, 'Rectangularity': 0.5,
    'Diameter': 0.5, 'Convexity': 1.7, 'Eccentricity': 4.5,
}


# -----------------------------
# Helper Functions
# -----------------------------
def _compute_group_slices() -> Dict[str, slice]:
    slices = {}
    idx = 0
    for key in HIST_KEYS:
        slices[key] = slice(idx, idx + HIST_BINS)
        idx += HIST_BINS
    for key in SCALAR_KEYS:
        slices[key] = slice(idx, idx + 1)
        idx += 1
    return slices


GROUP_SLICES = _compute_group_slices()


def _load_dataset(csv_path: str):
    df = pd.read_csv(csv_path, header=0, comment="#", engine="python")
    feat_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)
    y = df["Category"].astype(str).to_numpy()
    obj_ids = df["Object"].astype(str).to_numpy()
    labels = [os.path.join(cat, obj).replace('\\', '/') for cat, obj in zip(y, obj_ids)]

    return X, labels

def _normalize_features(features: np.ndarray) -> np.ndarray:
    norm_features = features.copy()
    # Normalize Histograms by sum
    for i in range(norm_features.shape[0]):
        for key in HIST_KEYS:
            sl = GROUP_SLICES[key]
            hist_slice = norm_features[i, sl]
            hist_sum = np.sum(hist_slice)
            if hist_sum > 0:
                norm_features[i, sl] = hist_slice / hist_sum

    # Standardize Scalars
    for key in SCALAR_KEYS:
        sl = GROUP_SLICES[key]
        scalar_column = norm_features[:, sl]
        mean, std = np.mean(scalar_column), np.std(scalar_column)
        norm_features[:, sl] = (scalar_column - mean) / std if std > 0 else 0.0

    return norm_features


# -----------------------------
# Distance Functions
# -----------------------------
def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1.0 - (dot_product / ((norm_a * norm_b) + EPS))


def _chi_squared_hist(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + EPS)))


def _kl_hist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + EPS) / (q + EPS))))


def _kl_sym_hist(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (_kl_hist(a, b) + _kl_hist(b, a))


def _emd_1d_hist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    'euclidean': _l2,
    'manhattan': _l1,
}
HIST_ONLY_METRICS = {'chi-squared', 'kullback-leibler', 'emd'}

COMPOSITE_METRICS: Dict[str, Dict[str, Callable]] = {
    'manhattan+chi-squared': {'scalar_fn': _l1, 'hist_fn': _chi_squared_hist},
    'manhattan+emd': {'scalar_fn': _l1, 'hist_fn': _emd_1d_hist},
    'manhattan+kullback-leibler': {'scalar_fn': _l1, 'hist_fn': _kl_sym_hist},
}

def _normalize_weights(weights: Dict[str, float], allowed: List[str], method: str = WEIGHTING_METHOD) -> Dict[str, float]:
    """
    Normalize weights for the given allowed keys.
    If method == 'neutral' then return uniform weights (bypass any provided weights).
    """
    if method == 'neutral':
        return {k: 1.0 / len(allowed) for k in allowed}

    # existing behavior: clamp to >=0 and normalize
    w = {k: max(0.0, weights.get(k, 0.0)) for k in allowed}
    s = sum(w.values())
    if s <= 0:
        return {k: 1.0 / len(allowed) for k in allowed}
    return {k: v / s for k, v in w.items()}


def _expand_group_weights_to_dims(sub_weights: Dict[str, float], groups: List[str], total_dim: int) -> np.ndarray:
    """
    Given normalized group weights, return a per-dimension weight array.
    """
    per_dim_weights = np.ones(total_dim, dtype=float)
    for g in groups:
        sl = GROUP_SLICES[g]
        per_dim_weights[sl] = sub_weights[g]
    return per_dim_weights


# -----------------------------
# Core Search Logic
# -----------------------------
class ShapeSearcher:
    """
    Manages loading, processing, and searching for 3D shapes based on feature vectors.
    Uses pure k-NN on the high-dimensional normalized features for the 'knn' metric.
    """

    def __init__(self, feature_csv_path: str, weights: Dict[str, float], weighting_method: str = 'feature'):
        if not os.path.exists(feature_csv_path):
            raise FileNotFoundError(f"Feature CSV file not found at '{feature_csv_path}'")

        self.weights = weights
        self.weighting_method = weighting_method
        # include 'knn' as the high-dimensional k-NN metric
        self.metrics = list(DISTANCE_FUNCTIONS.keys()) + list(COMPOSITE_METRICS.keys()) + ['knn']

        # Load raw features for display purposes
        df = pd.read_csv(feature_csv_path, header=0, comment="#", engine="python")
        feat_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Create labels from category and object columns
        categories = df["Category"].astype(str).to_numpy()
        obj_ids = df["Object"].astype(str).to_numpy()
        self.labels = [os.path.join(cat, obj).replace('\\', '/') for cat, obj in zip(categories, obj_ids)]

        # Store features as DataFrame indexed by label for easy lookup
        self.features_df = feat_df.copy()
        self.features_df.index = self.labels

        # Get normalized numpy features for distance calculations
        raw_features = feat_df.to_numpy(dtype=np.float32)
        self.features = _normalize_features(raw_features)

        # --- k-NN pre-computation on high-dimensional features ---
        self.knn_model = None
        self._setup_knn()

    def _setup_knn(self, n_neighbors=11):
        """Builds a k-NN model on the normalized high-dimensional features."""
        print("Building k-NN model on high-dimensional normalized features...")
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        self.knn_model.fit(self.features)
        print("k-NN model built.")

    def get_available_labels(self) -> List[str]:
        return self.labels

    def calculate_distance(self, index_a: int, index_b: int, metric: str) -> float:
        """
        Calculates the distance between two items specified by their indices.
        This method is designed to be called from an external process, like ProcessPoolExecutor.
        """
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")
        if metric == 'knn':
            raise NotImplementedError("Distance matrix calculation is not supported for the 'knn' metric.")

        vec_a = self.features[index_a]
        vec_b = self.features[index_b]

        if metric in COMPOSITE_METRICS:
            return self._compute_composite_distance(vec_a, vec_b, metric)
        else:
            return self._compute_distance(vec_a, vec_b, metric)

    def search(self, query_label: str, metric: str, top_n: int = 5) -> List[str]:
        """Finds the top-N most similar objects to the query."""
        if query_label not in self.labels:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")

        query_idx = self.labels.index(query_label)

        if metric == 'knn':
            return self._search_knn(query_idx, top_n)

        dists = np.array([self.calculate_distance(query_idx, i, metric) for i in range(len(self.labels))], dtype=float)

        sorted_indices = np.argsort(dists)
        results = [self.labels[i] for i in sorted_indices if i != query_idx]

        return results[:top_n]

    def search_with_distances(self, query_label: str, metric: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Finds the top-N most similar objects to the query and returns labels with distances."""
        if query_label not in self.labels:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")

        query_idx = self.labels.index(query_label)

        if metric == 'knn':
            # For kNN, we need to get distances as well
            if self.knn_model is None:
                raise RuntimeError("k-NN model is not available.")
            query_vec_hd = self.features[query_idx].reshape(1, -1)
            distances, indices = self.knn_model.kneighbors(query_vec_hd, n_neighbors=top_n + 1)
            # Skip the first one (query itself)
            neighbor_indices = indices.flatten()[1:]
            neighbor_distances = distances.flatten()[1:]
            return [(self.labels[i], float(d)) for i, d in zip(neighbor_indices, neighbor_distances)]

        # Calculate distances for all objects
        dists = np.array([self.calculate_distance(query_idx, i, metric) for i in range(len(self.labels))], dtype=float)

        # Sort by distance
        sorted_indices = np.argsort(dists)

        # Build results excluding the query itself
        results = [(self.labels[i], float(dists[i])) for i in sorted_indices if i != query_idx]

        return results[:top_n]

    def _search_knn(self, query_index: int, top_n: int) -> List[str]:
        """Performs a k-NN search on the pre-computed high-dimensional k-NN model."""
        if self.knn_model is None:
            raise RuntimeError("k-NN model is not available.")

        query_vec_hd = self.features[query_index].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(query_vec_hd, n_neighbors=top_n + 1)
        neighbor_indices = indices.flatten()[1:]
        return [self.labels[i] for i in neighbor_indices]

    def _compute_composite_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
        metric_fns = COMPOSITE_METRICS[metric]
        scalar_fn = metric_fns['scalar_fn']
        hist_fn = metric_fns['hist_fn']
        sub_weights = _normalize_weights(self.weights, FEATURE_GROUP_ORDER, self.weighting_method)

        total_dist = 0.0
        for group in HIST_KEYS:
            sl = GROUP_SLICES[group]
            raw_dist = hist_fn(vec_a[sl], vec_b[sl])
            total_dist += sub_weights[group] * raw_dist

        for group in SCALAR_KEYS:
            sl = GROUP_SLICES[group]
            raw_dist = scalar_fn(vec_a[sl], vec_b[sl])
            total_dist += sub_weights[group] * raw_dist

        return float(total_dist)

    def _compute_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
        # For neutral weighting, use the simple, unweighted form.
        if self.weighting_method == 'neutral':
            if metric == 'euclidean':
                return _l2(vec_a, vec_b)
            if metric == 'manhattan':
                return _l1(vec_a, vec_b)

        # For 'feature' weighting, calculate and apply weights.
        groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
        sub_weights = _normalize_weights(self.weights, groups, self.weighting_method)

        # Expand group weights to all dimensions for weighted calculation
        total_dim = vec_a.shape[0]
        per_dim_weights = _expand_group_weights_to_dims(sub_weights, groups, total_dim)
        diff = vec_a - vec_b

        if metric == 'euclidean':
            return float(np.sqrt(np.sum(per_dim_weights * (diff ** 2))))
        if metric == 'manhattan':
            return float(np.sum(per_dim_weights * np.abs(diff)))

        # Fallback for other simple metrics (if any are added)
        dist_fn = DISTANCE_FUNCTIONS.get(metric)
        if dist_fn:
             # This part is less efficient but handles generic cases
            total_dist = 0.0
            for group in groups:
                sl = GROUP_SLICES[group]
                raw_dist = dist_fn(vec_a[sl], vec_b[sl])
                total_dist += sub_weights[group] * raw_dist
            return float(total_dist)

        raise ValueError(f"Metric '{metric}' has no defined weighted calculation.")
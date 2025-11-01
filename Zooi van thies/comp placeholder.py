import os
import numpy as np
import pandas as pd
from typing import Callable, List, Dict
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Feature & Distance Constants
# -----------------------------
HIST_KEYS = ['A3', 'D1', 'D2', 'D3', 'D4']
SCALAR_KEYS = ['Surface area', 'Sphericity', 'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']
FEATURE_GROUP_ORDER = HIST_KEYS + SCALAR_KEYS
HIST_BINS = 20
EPS = 1e-10

# -----------------------------
# Default Configuration
# -----------------------------
# Adjust feature weights here. They are normalized to sum to 1.
MANUAL_WEIGHTS = {
    'A3': 3.0, 'D1': 1.0, 'D2': 2.0, 'D3': 3.0, 'D4': 3.0,
    'Surface area': 1.0, 'Sphericity': 1.5, 'Rectangularity': 1.5,
    'Diameter': 1.0, 'Convexity': 0.5, 'Eccentricity': 1.0,
}

#MANUAL_WEIGHTS_ADJUSTED = {
#    'A3': 1.0, 'D1': 1.0, 'D2': 1.0, 'D3': 1.0, 'D4': 1.0,
#     'Surface area': 1.0, 'Sphericity': 1.0, 'Rectangularity': 1.0,
#     'Diameter': 1.0, 'Convexity': 1.0, 'Eccentricity': 1.0,
#}

# -----------------------------
# Helper Functions
# -----------------------------
def _compute_group_slices() -> Dict[str, slice]:
    """Computes the slice for each feature group in the concatenated vector."""
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
    """Loads features, labels, and object IDs from the CSV file."""
    df = pd.read_csv(csv_path, header=0, comment="#", engine="python")
    feat_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)
    y = df["Category"].astype(str).to_numpy()
    obj_ids = df["Object"].astype(str).to_numpy()
    labels = [os.path.join(cat, obj).replace('\\', '/') for cat, obj in zip(y, obj_ids)]

    return X, labels

def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalizes feature vectors."""
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
    """Computes L1 (Manhattan) distance."""
    return float(np.sum(np.abs(a - b)))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """Computes L2 (Euclidean) distance."""
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cosine distance."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1.0 - (dot_product / ((norm_a * norm_b) + EPS))


def _chi_squared_hist(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Chi-Squared distance for histograms."""
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + EPS)))


def _kl_hist(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence for histograms."""
    return float(np.sum(p * np.log(p / (q + EPS) + EPS)))


def _kl_sym_hist(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Kullback-Leibler divergence for histograms."""
    return 0.5 * (_kl_hist(a, b) + _kl_hist(b, a))


def _emd_1d_hist(a: np.ndarray, b: np.ndarray) -> float:
    """Earth Mover's Distance for 1D histograms."""
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    'euclidean': _l2,
    'manhattan': _l1,
    'cosine': _cosine_dist,
}
HIST_ONLY_METRICS = {'chi-squared', 'kullback-leibler', 'emd'}

COMPOSITE_METRICS: Dict[str, Dict[str, Callable]] = {
    'manhattan+chi-squared': {'scalar_fn': _l1, 'hist_fn': _chi_squared_hist},
    'manhattan+emd': {'scalar_fn': _l1, 'hist_fn': _emd_1d_hist},
    'manhattan+kullback-leibler': {'scalar_fn': _l1, 'hist_fn': _kl_sym_hist},
}

def _normalize_weights(weights: Dict[str, float], allowed: List[str]) -> Dict[str, float]:
    """Normalizes a weight dictionary to sum to 1."""
    w = {k: max(0.0, weights.get(k, 0.0)) for k in allowed}
    s = sum(w.values())
    if s <= 0:
        return {k: 1.0 / len(allowed) for k in allowed}
    return {k: v / s for k, v in w.items()}


# -----------------------------
# Core Search Logic
# -----------------------------
class ShapeSearcher:
    """
    Manages loading, processing, and searching for 3D shapes based on feature vectors.
    Always uses feature-based weighting (no 'distance' weighting).
    """

    def __init__(self, feature_csv_path: str, weights: Dict[str, float], weighting_method: str = 'feature'):
        """
        Initializes the searcher by loading and normalizing features from a CSV file.

        Note: the `weighting_method` parameter is accepted for compatibility but ignored.
        Feature weighting is always used.
        """
        if not os.path.exists(feature_csv_path):
            raise FileNotFoundError(f"Feature CSV file not found at '{feature_csv_path}'")

        self.weights = weights
        self.metrics = list(DISTANCE_FUNCTIONS.keys()) + list(COMPOSITE_METRICS.keys()) + ['knn']
        raw_features, self.labels = _load_dataset(feature_csv_path)
        self.features = _normalize_features(raw_features)

        # --- k-NN pre-computation ---
        self.knn_model = None
        self._setup_knn()

    def _setup_knn(self, n_neighbors=11):
        """Builds a k-NN model on the high-dimensional feature data."""
        print("Building k-NN model on feature data...")
        # Use 'minkowski' with p=2 for Euclidean distance, which is standard for k-NN
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='minkowski', p=2)
        self.knn_model.fit(self.features)
        print("k-NN model built.")

    def get_available_labels(self) -> List[str]:
        """Returns a list of all available model labels."""
        return self.labels

    def search(self, query_label: str, metric: str, top_n: int = 5) -> List[str]:
        """
        Finds the most similar shapes to a given query shape.

        Uses:
        - 'knn' -> k-NN search on the high-dimensional feature vectors.
        - composite metrics -> combined hist/scalar functions.
        - simple metrics -> single distance applied per feature group with feature weighting.
        """
        if query_label not in self.labels:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")

        query_idx = self.labels.index(query_label)

        if metric == 'knn':
            return self._search_knn(query_idx, top_n)

        query_vec = self.features[query_idx]
        if metric in COMPOSITE_METRICS:
            dist_computer = self._compute_composite_distance
        else:
            dist_computer = self._compute_distance

        dists = np.array([dist_computer(query_vec, vec, metric) for vec in self.features], dtype=float)

        # Get indices of sorted distances, excluding the query item itself
        sorted_indices = np.argsort(dists)
        results = [self.labels[i] for i in sorted_indices if i != query_idx]

        return results[:top_n]

    def _search_knn(self, query_index: int, top_n: int) -> List[str]:
        """Performs a k-NN search on the pre-computed feature model."""
        if self.knn_model is None:
            raise RuntimeError("k-NN model is not available.")

        query_vec = self.features[query_index].reshape(1, -1)
        # Query for top_n + 1 to account for the query item itself
        distances, indices = self.knn_model.kneighbors(query_vec, n_neighbors=top_n + 1)

        # Exclude the first result (which is the query item itself)
        neighbor_indices = indices.flatten()[1:]

        return [self.labels[i] for i in neighbor_indices]

    def _compute_composite_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
        """Computes a weighted distance using different functions for scalars and histograms."""
        metric_fns = COMPOSITE_METRICS[metric]
        scalar_fn = metric_fns['scalar_fn']
        hist_fn = metric_fns['hist_fn']
        sub_weights = _normalize_weights(self.weights, FEATURE_GROUP_ORDER)

        total_dist = 0.0

        # Calculate distance for histogram groups
        for group in HIST_KEYS:
            sl = GROUP_SLICES[group]
            raw_dist = hist_fn(vec_a[sl], vec_b[sl])
            total_dist += sub_weights[group] * raw_dist

        # Calculate distance for scalar groups
        for group in SCALAR_KEYS:
            sl = GROUP_SLICES[group]
            raw_dist = scalar_fn(vec_a[sl], vec_b[sl])
            total_dist += sub_weights[group] * raw_dist

        return float(total_dist)



    def _compute_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
        """Computes the final weighted distance between two feature vectors using feature weighting only."""
        dist_fn = DISTANCE_FUNCTIONS[metric]
        groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
        sub_weights = _normalize_weights(self.weights, groups)

        total_dist = 0.0
        for group in groups:
            sl = GROUP_SLICES[group]
            raw_dist = dist_fn(vec_a[sl], vec_b[sl])
            # Always apply feature weighting directly
            total_dist += sub_weights[group] * raw_dist

        return float(total_dist)

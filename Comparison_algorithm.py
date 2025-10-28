import os
import numpy as np
import pandas as pd
from typing import Callable, List, Dict

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
# 'feature': Weights feature groups before distance calculation.
# 'distance': Standardizes per-group distances before applying weights.
WEIGHTING_METHOD = 'feature'

# Adjust feature weights here. They are normalized to sum to 1.
MANUAL_WEIGHTS = {
    'A3': 2.0, 'D1': 1.0, 'D2': 2.0, 'D3': 2.0, 'D4': 2.0,
    'Surface area': 1.0, 'Sphericity': 1.0, 'Rectangularity': 1.0,
    'Diameter': 1.0, 'Convexity': 1.0, 'Eccentricity': 1.0,
}


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


def _normalize_single_feature_vector(self, features: np.ndarray) -> np.ndarray:
    """Normalizes a single feature vector (1D array)."""
    norm_features = features.copy()
    # Normalize Histograms by sum
    for key in HIST_KEYS:
        sl = GROUP_SLICES[key]
        hist_slice = norm_features[sl]
        hist_sum = np.sum(hist_slice)
        if hist_sum > 0:
            norm_features[sl] = hist_slice / hist_sum
    # Standardize Scalars
    for key in SCALAR_KEYS:
        sl = GROUP_SLICES[key]
        norm_features[sl] = (norm_features[sl] - self.scalar_means[key]) / self.scalar_stds[key] if self.scalar_stds[key] > 0 else 0.0
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
    'chi-squared': _chi_squared_hist,
    'manhattan': _l1,
    'cosine': _cosine_dist,
    'kullback-leibler': _kl_sym_hist,
    'emd': _emd_1d_hist,
}
HIST_ONLY_METRICS = {'chi-squared', 'kullback-leibler', 'emd'}


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
    """

    def __init__(self, feature_csv_path: str, weights: Dict[str, float], weighting_method: str = 'feature'):
        """
        Initializes the searcher by loading and normalizing features from a CSV file.

        Args:
            feature_csv_path (str): Path to the feature CSV file.
            weights (Dict[str, float]): Weights for each feature group.
            weighting_method (str): Method for weighting ('feature' or 'distance').
        """
        if not os.path.exists(feature_csv_path):
            raise FileNotFoundError(f"Feature CSV file not found at '{feature_csv_path}'")

        self.weights = weights
        self.weighting_method = weighting_method
        self.metrics = list(DISTANCE_FUNCTIONS.keys())
        self.dist_stats: Dict[str, Dict[str, tuple]] = {}

        raw_features, self.labels = _load_dataset(feature_csv_path)
        self.features = _normalize_features(raw_features)

        self.scalar_means = {}
        self.scalar_stds = {}
        for key in SCALAR_KEYS:
            sl = GROUP_SLICES[key]
            scalar_column = raw_features[:, sl].flatten()
            self.scalar_means[key] = np.mean(scalar_column)
            self.scalar_stds[key] = np.std(scalar_column)

        if self.weighting_method == 'distance':
            self._precompute_distance_stats()

    def get_available_labels(self) -> List[str]:
        """Returns a list of all available model labels."""
        return self.labels

    def search(self, query_label: str, metric: str, top_n: int = 5) -> List[str]:
        """
        Finds the most similar shapes to a given query shape.

        Args:
            query_label (str): The label of the query shape (e.g., 'Category/model.obj').
            metric (str): The distance metric to use (e.g., 'euclidean').
            top_n (int): The number of top results to return.

        Returns:
            List[str]: A list of labels for the most similar shapes.
        """
        if query_label not in self.labels:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")
        if metric not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Unknown metric: {metric}")

        query_idx = self.labels.index(query_label)
        query_vec = self.features[query_idx]

        dists = np.array([
            self._compute_distance(query_vec, vec, metric) for vec in self.features
        ], dtype=float)

        # Get indices of sorted distances, excluding the query item itself
        sorted_indices = np.argsort(dists)
        results = [self.labels[i] for i in sorted_indices if i != query_idx]

        return results[:top_n]

    def _compute_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
        """Computes the final weighted distance between two feature vectors."""
        dist_fn = DISTANCE_FUNCTIONS[metric]
        groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
        sub_weights = _normalize_weights(self.weights, groups)

        total_dist = 0.0
        for group in groups:
            sl = GROUP_SLICES[group]
            raw_dist = dist_fn(vec_a[sl], vec_b[sl])

            if self.weighting_method == 'distance':
                mean, std = self.dist_stats.get(metric, {}).get(group, (0.0, 1.0))
                norm_dist = (raw_dist - mean) / std if std > 0 else 0.0
                total_dist += sub_weights[group] * norm_dist
            else:  # 'feature' weighting
                total_dist += sub_weights[group] * raw_dist

        return float(total_dist)

    def _precompute_distance_stats(self, num_samples: int = 500):
        """Calculates mean/std of distances for the 'distance' weighting method."""
        num_shapes = self.features.shape[0]
        rng = np.random.default_rng(42)
        indices = rng.choice(num_shapes, size=min(num_samples, num_shapes), replace=False)
        sampled_features = self.features[indices]

        for metric, dist_fn in DISTANCE_FUNCTIONS.items():
            self.dist_stats[metric] = {}
            groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
            for g in groups:
                sl = GROUP_SLICES[g]
                group_vectors = sampled_features[:, sl]
                dists = [
                    dist_fn(group_vectors[i], group_vectors[j])
                    for i in range(len(group_vectors)) for j in range(i + 1, len(group_vectors))
                ]
                if dists:
                    self.dist_stats[metric][g] = (np.mean(dists), np.std(dists))
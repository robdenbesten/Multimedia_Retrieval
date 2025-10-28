import os
import json
import numpy as np
import pandas as pd
from typing import Callable, List, Dict

# --- (Keep existing constants and distance functions: lines 5 to 181) ---
# This includes HIST_KEYS, SCALAR_KEYS, GROUP_SLICES, distance functions, etc.
# The changes are primarily in the ShapeSearcher class.

# -----------------------------
# Feature & Distance Constants
# -----------------------------
HIST_KEYS = ['A3', 'D1', 'D2', 'D3', 'D4']
# Added extents to scalars
SCALAR_KEYS = ['Mesh volume', 'Surface area', 'Diameter', 'Compactness',
               'Rectangularity', 'Convexity', 'Eccentricity', 'Sphericity',
               'extents_0', 'extents_1', 'extents_2']
HIST_BINS = 20
EPS = 1e-10


# --- Helper to compute slices based on new SCALAR_KEYS ---
def _compute_group_slices() -> Dict[str, slice]:
    slices = {}
    idx = 0
    # Histograms first
    hist_feature_names = [f'{k}_bin_{i}' for k in HIST_KEYS for i in range(HIST_BINS)]
    # Scalars second
    scalar_feature_names = SCALAR_KEYS

    # This order must match the CSV file's column order after Object/Category
    all_feature_names = scalar_feature_names + hist_feature_names

    for key in SCALAR_KEYS:
        slices[key] = slice(idx, idx + 1)
        idx += 1
    for key in HIST_KEYS:
        slices[key] = slice(idx, idx + HIST_BINS)
        idx += HIST_BINS
    return slices


GROUP_SLICES = _compute_group_slices()

# Default Configuration
WEIGHTING_METHOD = 'feature'
MANUAL_WEIGHTS = {
    'A3': 2.0, 'D1': 1.0, 'D2': 2.0, 'D3': 2.0, 'D4': 2.0,
    'Mesh volume': 1.0, 'Surface area': 1.0, 'Sphericity': 1.0,
    'Rectangularity': 1.0, 'Diameter': 1.0, 'Convexity': 1.0,
    'Eccentricity': 1.0, 'extents_0': 0.5, 'extents_1': 0.5, 'extents_2': 0.5
}
FEATURE_GROUP_ORDER = SCALAR_KEYS + HIST_KEYS


def _load_dataset(csv_path: str):
    df = pd.read_csv(csv_path, header=0, comment="#", engine="python")
    feat_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = feat_df.to_numpy(dtype=np.float32)
    labels = [os.path.join(cat, obj).replace('\\', '/') for cat, obj in zip(df["Category"], df["Object"])]
    return X, labels


# --- (Distance functions _l1, _l2, etc. remain unchanged) ---
def _l1(a: np.ndarray, b: np.ndarray) -> float: return float(np.sum(np.abs(a - b)))


def _l2(a: np.ndarray, b: np.ndarray) -> float: return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1.0 - (dot_product / ((norm_a * norm_b) + EPS))


def _chi_squared_hist(a: np.ndarray, b: np.ndarray) -> float: return float(0.5 * np.sum((a - b) ** 2 / (a + b + EPS)))


def _kl_hist(p: np.ndarray, q: np.ndarray) -> float: return float(np.sum(p * np.log(p / (q + EPS) + EPS)))


def _kl_sym_hist(a: np.ndarray, b: np.ndarray) -> float: return 0.5 * (_kl_hist(a, b) + _kl_hist(b, a))


def _emd_1d_hist(a: np.ndarray, b: np.ndarray) -> float: return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    'euclidean': _l2, 'chi-squared': _chi_squared_hist, 'manhattan': _l1,
    'cosine': _cosine_dist, 'kullback-leibler': _kl_sym_hist, 'emd': _emd_1d_hist,
}
HIST_ONLY_METRICS = {'chi-squared', 'kullback-leibler', 'emd'}


def _normalize_weights(weights: Dict[str, float], allowed: List[str]) -> Dict[str, float]:
    w = {k: max(0.0, weights.get(k, 0.0)) for k in allowed}
    s = sum(w.values())
    if s <= 0: return {k: 1.0 / len(allowed) for k in allowed}
    return {k: v / s for k, v in w.items()}


class ShapeSearcher:
    def __init__(self, feature_csv_path: str, stats_path: str, weights: Dict[str, float],
                 weighting_method: str = 'feature'):
        if not os.path.exists(feature_csv_path):
            raise FileNotFoundError(f"Feature CSV not found: '{feature_csv_path}'")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats JSON not found: '{stats_path}'")

        self.weights = weights
        self.weighting_method = weighting_method
        self.metrics = list(DISTANCE_FUNCTIONS.keys())

        # Load pre-normalized features
        self.features, self.labels = _load_dataset(feature_csv_path)

        # Load normalization stats for single vectors
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

        if self.weighting_method == 'distance':
            self._precompute_distance_stats()

    def _normalize_single_vector(self, raw_vector: np.ndarray) -> np.ndarray:
        """Normalizes a single raw feature vector using pre-loaded stats."""
        norm_vector = raw_vector.copy()

        # Standardize Scalars
        for i, key in enumerate(SCALAR_KEYS):
            mean = self.stats['means'].get(key, 0.0)
            std = self.stats['stds'].get(key, 1.0)
            if std > 0:
                norm_vector[i] = (norm_vector[i] - mean) / std
            else:
                norm_vector[i] = 0.0

        # Normalize Histograms
        for key in HIST_KEYS:
            sl = GROUP_SLICES[key]
            hist_slice = norm_vector[sl]
            hist_sum = np.sum(hist_slice)
            if hist_sum > 0:
                norm_vector[sl] = hist_slice / hist_sum

        return norm_vector

    def search_by_vector(self, query_vector: np.ndarray, metric: str, top_n: int = 5) -> List[str]:
        """Normalizes a raw query vector and finds the most similar shapes."""
        if metric not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Unknown metric: {metric}")

        # Normalize the raw query vector
        normalized_query = self._normalize_single_vector(query_vector)

        dists = np.array([
            self._compute_distance(normalized_query, vec, metric) for vec in self.features
        ], dtype=float)

        sorted_indices = np.argsort(dists)
        return [self.labels[i] for i in sorted_indices[:top_n]]

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
                # This part requires pre-computation, which is complex.
                # Sticking to 'feature' weighting is simpler and often as effective.
                mean, std = self.dist_stats.get(metric, {}).get(group, (0.0, 1.0))
                norm_dist = (raw_dist - mean) / std if std > 0 else 0.0
                total_dist += sub_weights[group] * norm_dist
            else:  # 'feature' weighting
                total_dist += sub_weights[group] * raw_dist

        return float(total_dist)

    # _precompute_distance_stats can remain as is for 'distance' weighting method
    def _precompute_distance_stats(self, num_samples: int = 500):
        """Calculates mean/std of distances for the 'distance' weighting method."""
        num_shapes = self.features.shape[0]
        rng = np.random.default_rng(42)
        indices = rng.choice(num_shapes, size=min(num_samples, num_shapes), replace=False)
        sampled_features = self.features[indices]

        self.dist_stats: Dict[str, Dict[str, tuple]] = {}
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

#heatmap van elke categorie maken, curf maken en ook dus percentagen en combineren van de scalars en dan daarna de histo hybride dingen maken

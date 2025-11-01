import os
import numpy as np
import pandas as pd
import json
import math
import random
from typing import Callable, List, Dict, Tuple, Optional
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
MANUAL_WEIGHTS: Dict[str, float] = {
    'A3': 3.0, 'D1': 1.0, 'D2': 2.0, 'D3': 3.0, 'D4': 3.0,
    'Surface area': 1.0, 'Sphericity': 1.5, 'Rectangularity': 1.5,
    'Diameter': 1.0, 'Convexity': 0.5, 'Eccentricity': 1.0,
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


def _load_dataset(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """Loads features and creates unique labels from the CSV file."""
    df = pd.read_csv(csv_path, header=0, comment="#", engine="python")
    feat_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    features = feat_df.to_numpy(dtype=np.float32)
    labels = [os.path.join(str(cat), str(obj)).replace('\\', '/') for cat, obj in zip(df["Category"], df["Object"])]
    return features, labels


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalizes histogram groups by sum and standardizes scalar groups."""
    norm_features = features.copy()
    # Normalize Histograms
    for key in HIST_KEYS:
        sl = GROUP_SLICES[key]
        hist_group = norm_features[:, sl]
        row_sums = np.sum(hist_group, axis=1, keepdims=True)
        # Avoid division by zero for empty histograms
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        norm_features[:, sl] = hist_group / safe_sums

    # Standardize Scalars
    for key in SCALAR_KEYS:
        sl = GROUP_SLICES[key]
        scalar_column = norm_features[:, sl]
        mean, std = np.mean(scalar_column), np.std(scalar_column)
        norm_features[:, sl] = (scalar_column - mean) / std if std > EPS else 0.0

    return norm_features


def _normalize_weights(weights: Dict[str, float], allowed: List[str]) -> Dict[str, float]:
    """Normalizes a weight dictionary to sum to 1."""
    w = {k: max(0.0, weights.get(k, 0.0)) for k in allowed}
    s = sum(w.values())
    if s <= 0:
        # If no valid weights, fall back to uniform weighting
        return {k: 1.0 / len(allowed) for k in allowed}
    return {k: v / s for k, v in w.items()}


# -----------------------------
# Distance Functions
# -----------------------------
def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _l2_flat(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denominator = (norm_a * norm_b)
    return 1.0 - (dot_product / denominator if denominator > EPS else 0.0)


def _chi_squared_hist(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + EPS)))


def _kl_hist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / (q + EPS) + EPS)))


def _kl_sym_hist(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (_kl_hist(a, b) + _kl_hist(b, a))


def _emd_1d_hist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    'euclidean': _l2,
    'euclidean_flat': _l2_flat,
    'manhattan': _l1,
    'cosine': _cosine_dist,
}

COMPOSITE_METRICS: Dict[str, Dict[str, Callable]] = {
    'manhattan+chi-squared': {'scalar_fn': _l1, 'hist_fn': _chi_squared_hist},
    'manhattan+emd': {'scalar_fn': _l1, 'hist_fn': _emd_1d_hist},
    'manhattan+kullback-leibler': {'scalar_fn': _l1, 'hist_fn': _kl_sym_hist},
}

FLAT_METRICS = {'euclidean_flat', 'cosine'}


# -----------------------------
# Distance Weighting Map
# -----------------------------
class DistanceWeightMap:
    @staticmethod
    def _iter_sample_pairs(n: int, max_pairs: int, rng: random.Random):
        if n < 2 or max_pairs <= 0: return
        seen = set()
        # Cap budget at the total number of unique pairs
        budget = min(max_pairs, n * (n - 1) // 2)
        for _ in range(budget * 2):  # Add timeout mechanism
            if len(seen) >= budget: break
            i, j = rng.sample(range(n), 2)
            key = tuple(sorted((i, j)))
            if key not in seen:
                seen.add(key)
                yield key

    @staticmethod
    def build(features, metrics, group_slices, max_pairs=50000, random_state=42, min_std=1e-6) -> Dict:
        rng = random.Random(random_state)
        N = features.shape[0]
        if N < 2: return {m: {g: 1.0 for g in group_slices} for m in metrics}

        group_arrays = {g: features[:, sl] for g, sl in group_slices.items()}
        stats = {}  # Using Welford's online algorithm: (count, mean, M2)

        def update_stats(key, x):
            cnt, mean, M2 = stats.get(key, (0, 0.0, 0.0))
            cnt += 1
            delta = x - mean
            mean += delta / cnt
            delta2 = x - mean
            M2 += delta * delta2
            stats[key] = (cnt, mean, M2)

        for i, j in DistanceWeightMap._iter_sample_pairs(N, max_pairs, rng):
            for metric in metrics:
                for group, arr in group_arrays.items():
                    fn = None
                    if metric in COMPOSITE_METRICS:
                        fn = COMPOSITE_METRICS[metric]['hist_fn'] if group in HIST_KEYS else COMPOSITE_METRICS[metric][
                            'scalar_fn']
                    elif metric in DISTANCE_FUNCTIONS:
                        fn = DISTANCE_FUNCTIONS[metric]

                    if fn:
                        d = float(fn(arr[i], arr[j]))
                        update_stats((metric, group), d)

        scales = {m: {} for m in metrics}
        for (metric, group), (cnt, _, M2) in stats.items():
            if cnt > 1:
                std = math.sqrt(max(M2 / (cnt - 1), 0.0))
                scales[metric][group] = max(std, min_std)
            else:
                scales[metric][group] = 1.0
        return scales

    @staticmethod
    def save(scales: Dict, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: json.dump(scales, f)

    @staticmethod
    def load(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)


# -----------------------------
# Core Search Logic
# -----------------------------
class ShapeSearcher:
    def __init__(
            self,
            feature_csv_path: str,
            weights: Optional[Dict[str, float]],
            weighting_method: str = 'feature',
            distance_weightmap_path: Optional[str] = None,
            build_distance_map: bool = False,
            distance_max_pairs: int = 50000,
            distance_random_state: int = 42,
    ):
        if not os.path.exists(feature_csv_path):
            raise FileNotFoundError(f"Feature CSV file not found at '{feature_csv_path}'")

        self.weighting_method = weighting_method
        self.available_metrics = list(DISTANCE_FUNCTIONS.keys()) + list(COMPOSITE_METRICS.keys()) + ['knn']

        raw_features, self.labels = _load_dataset(feature_csv_path)
        self.features = _normalize_features(raw_features)
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        self.weights = {}
        if self.weighting_method == 'feature':
            self.weights = _normalize_weights(weights or {}, FEATURE_GROUP_ORDER)
        elif self.weighting_method == 'none':
            self.weights = {key: 1.0 for key in FEATURE_GROUP_ORDER}

        self.distance_scales = None
        if self.weighting_method == 'distance':
            self._initialize_distance_scales(
                feature_csv_path, distance_weightmap_path, build_distance_map,
                distance_max_pairs, distance_random_state
            )

        self.knn_model = None  # Lazy initialization

    def _initialize_distance_scales(self, *args):
        feature_csv_path, distance_weightmap_path, build_distance_map, max_pairs, random_state = args
        default_path = os.path.join(os.path.dirname(feature_csv_path) or ".", "distance_weightmap.json")
        wm_path = distance_weightmap_path or default_path

        if not build_distance_map and os.path.exists(wm_path):
            print(f"Loading distance weightmap from `{wm_path}`...")
            self.distance_scales = DistanceWeightMap.load(wm_path)
        else:
            print(f"Building distance weightmap at `{wm_path}`...")
            metrics_to_build = [m for m in self.available_metrics if m not in FLAT_METRICS + ['knn']]
            self.distance_scales = DistanceWeightMap.build(
                self.features, metrics_to_build, GROUP_SLICES, max_pairs, random_state
            )
            DistanceWeightMap.save(self.distance_scales, wm_path)

    def _setup_knn(self, n_neighbors=11):
        if self.knn_model is None:
            print("Building k-NN model on feature data...")
            self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='minkowski', p=2)
            self.knn_model.fit(self.features)
            print("k-NN model built.")

    def _compute_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
        if metric in FLAT_METRICS:
            return float(DISTANCE_FUNCTIONS[metric](vec_a, vec_b))

        total = 0.0
        for group in FEATURE_GROUP_ORDER:
            sl = GROUP_SLICES[group]
            dist = 0.0

            # Select the correct distance function for the group
            fn = None
            if metric in COMPOSITE_METRICS:
                fn = COMPOSITE_METRICS[metric]['hist_fn'] if group in HIST_KEYS else COMPOSITE_METRICS[metric][
                    'scalar_fn']
            elif metric in DISTANCE_FUNCTIONS:
                fn = DISTANCE_FUNCTIONS[metric]

            if fn:
                dist = fn(vec_a[sl], vec_b[sl])

            # Apply weighting strategy
            if self.weighting_method in ('feature', 'none'):
                total += self.weights[group] * dist
            elif self.weighting_method == 'distance':
                scale = self.distance_scales.get(metric, {}).get(group, 1.0)
                total += dist / max(scale, EPS)
        return float(total)

    def search(self, query_label: str, metric: str, top_n: int = 10) -> List[str]:
        if metric not in self.available_metrics:
            raise ValueError(f"Metric '{metric}' is not available. Choose from: {self.available_metrics}")

        if query_label not in self.label_to_idx:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")

        query_idx = self.label_to_idx[query_label]
        query_vec = self.features[query_idx]

        if metric == 'knn':
            self._setup_knn(n_neighbors=top_n + 1)
            distances, indices = self.knn_model.kneighbors([query_vec])
            # Exclude the query item itself from the results
            neighbor_indices = [idx for idx in indices[0] if idx != query_idx][:top_n]
            return [self.labels[i] for i in neighbor_indices]

        # Brute-force search for other metrics
        results = []
        for i, vec in enumerate(self.features):
            if i == query_idx: continue
            dist = self._compute_distance(query_vec, vec, metric)
            results.append((dist, self.labels[i]))

        results.sort(key=lambda x: x[0])
        return [label for dist, label in results[:top_n]]

    def get_available_labels(self) -> List[str]:
        return self.labels
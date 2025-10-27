import os
import json
import sys
import csv
import pandas as pd
from typing import Callable, List, Dict
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QApplication
)
from vedo import Plotter, Mesh, Text2D

# -----------------------------
# Feature groups and constants
# -----------------------------
HIST_KEYS = ['A3', 'D1', 'D2', 'D3', 'D4']
HIST_BINS = 20
SCALAR_KEYS = ['Surface area', 'Sphericity', 'Rectangularity', 'Diameter', 'Convexity', 'Eccentricity']
FEATURE_GROUP_ORDER = HIST_KEYS + SCALAR_KEYS
HIST_TOTAL_BINS = len(HIST_KEYS) * HIST_BINS
CROSS_BIN_SIGMA = 1.5  # neighborhood width in bins (tune as needed)
CROSS_BIN_P = 1.0  # p in the formula; try 1.0 or 2.0

EPS = 1e-10


def compute_group_slices() -> Dict[str, slice]:
    """Computes the slice for each feature group in the concatenated vector."""
    idx = 0
    slices: Dict[str, slice] = {}
    for hk in HIST_KEYS:
        slices[hk] = slice(idx, idx + HIST_BINS)
        idx += HIST_BINS
    for sk in SCALAR_KEYS:
        slices[sk] = slice(idx, idx + 1)
        idx += 1
    return slices


GROUP_SLICES = compute_group_slices()


# -----------------------------
# Feature Parsing and Normalization
# -----------------------------
def load_dataset(csv_path: str, models_dir: str | None = None, has_header: bool = False):
    # Read CSV; the file has no header in your sample
    df = pd.read_csv(csv_path, header=0 if has_header else None, comment="#", engine="python")

    if has_header:
        required = {"object", "category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    else:
        # Rename first two columns to `object`, `category`
        df = df.rename(columns={0: "object", 1: "category"})

    # Drop trailing completely empty columns (if any)
    if df.shape[1] > 2:
        empty_cols = [c for c in df.columns[2:] if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)

    # Convert feature columns to numeric
    feat_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    # Drop rows with no valid numeric features
    mask_all_nan = feat_df.isna().all(axis=1)
    if mask_all_nan.any():
        df = df.loc[~mask_all_nan].reset_index(drop=True)
        feat_df = feat_df.loc[~mask_all_nan].reset_index(drop=True)

    # Fill remaining NaNs with 0.0 (or use another strategy)
    feat_df = feat_df.fillna(0.0)

    # Outputs commonly needed downstream
    X = feat_df.to_numpy(dtype=np.float32)           # features
    y = df["category"].astype(str).to_numpy()        # labels
    obj_ids = df["object"].astype(str).to_numpy()    # object ids / filenames

    # Optional: if you still need full paths, build them from a base directory
    if models_dir:
        df["model_path"] = [os.path.normpath(os.path.join(models_dir, o)) for o in obj_ids]

    return df, X, y, obj_ids



def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalizes feature vectors. Histograms are divided by sum, scalars by standardization.
    """
    norm_features = features.copy()

    # 1. Normalize Histograms (by area/sum)
    for i in range(norm_features.shape[0]):  # For each shape
        for key in HIST_KEYS:
            sl = GROUP_SLICES[key]
            hist_slice = norm_features[i, sl]
            hist_sum = np.sum(hist_slice)
            if hist_sum > 0:
                norm_features[i, sl] = hist_slice / hist_sum

    # 2. Normalize Scalars (by standardization: (x - mean) / std_dev)
    for key in SCALAR_KEYS:
        sl = GROUP_SLICES[key]
        scalar_column = norm_features[:, sl]
        mean = np.mean(scalar_column)
        std = np.std(scalar_column)
        if std > 0:
            norm_features[:, sl] = (scalar_column - mean) / std
        else:
            norm_features[:, sl] = 0.0  # All values are the same, so no variance

    return norm_features


def get_obj_path(label: str, obj_root_dir: str) -> str:
    """Constructs the full .obj path from a label and root directory."""
    return os.path.join(obj_root_dir, label)


# -----------------------------
# Base Distance Functions
# -----------------------------
def l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return 1.0 if na == 0.0 or nb == 0.0 else float(1.0 - (np.dot(a, b) / (na * nb)))


def chi_squared_hist(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + EPS)))


def kl_sym_hist(a: np.ndarray, b: np.ndarray) -> float:
    kl_ab = np.sum(a * np.log((a + EPS) / (b + EPS)))
    kl_ba = np.sum(b * np.log((b + EPS) / (a + EPS)))
    return float(kl_ab + kl_ba)


def emd_1d_hist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


def make_cross_bin_kernel(n: int, sigma: float = 1.5, normalize: bool = True) -> np.ndarray:
    """
    Build an n×n kernel W with weights for cross-bin matching.
    By default uses a Gaussian on bin index distance.
    """
    idx = np.arange(n)
    # Gaussian weights by bin distance
    W = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / max(sigma, 1e-12)) ** 2)
    if normalize:
        s = W.sum()
        if s > 0:
            W /= s
    return W


# Global kernel for all histogram groups (all histograms have HIST_BINS bins)
CROSS_BIN_W = make_cross_bin_kernel(HIST_BINS, sigma=CROSS_BIN_SIGMA, normalize=True)


def cross_bin_hist(a: np.ndarray, b: np.ndarray, W: np.ndarray = CROSS_BIN_W, p: float = CROSS_BIN_P) -> float:
    """
    Cross-bin histogram distance:
        d(H1,H2) = ( sum_i sum_j w_ij * |H1[i] - H2[j]|^p )^(1/p)
    a, b: 1D histograms of same length
    W:    n×n nonnegative weights (ideally symmetric); typically normalized to sum=1
    p:    Minkowski order (>=1)
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"Histogram length mismatch: {a.size} vs {b.size}")
    if W.shape != (a.size, b.size):
        raise ValueError(f"Kernel shape {W.shape} does not match hist length {a.size}")
    # Pairwise |a_i - b_j|^p, weighted by W
    diff_p = np.abs(a[:, None] - b[None, :]) ** p
    s = float(np.sum(W * diff_p))
    return s ** (1.0 / p)


# -----------------------------
# Weight and Distance Aggregation
# -----------------------------
HIST_ONLY_METRICS = {'chi-squared', 'kullback-leibler', 'emd', 'cross-bin'}


def normalize_weights(weights: Dict[str, float], allowed: List[str]) -> Dict[str, float]:
    """Normalizes a weight dictionary to sum to 1."""
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in allowed}
    s = sum(w.values())
    if s <= 0:
        equal = 1.0 / len(allowed)
        return {k: equal for k in allowed}
    return {k: v / s for k, v in w.items()}


def compute_weighted_distance(a: np.ndarray, b: np.ndarray, metric: str, weights: Dict[str, float]) -> float:
    """Computes the final distance as a weighted sum of per-group distances."""
    groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
    sub_weights = normalize_weights(weights, groups)

    dist_fn_map: Dict[str, Callable] = {
        'euclidean': l2,
        'manhattan': l1,
        'cosine': cosine_dist,
        'chi-squared': chi_squared_hist,
        'kullback-leibler': kl_sym_hist,
        'emd': emd_1d_hist,
        'cross-bin': cross_bin_hist,
    }
    dist_fn = dist_fn_map.get(metric)
    if not dist_fn:
        raise ValueError(f"Unknown metric: {metric}")

    total_dist = 0.0
    for g in groups:
        sl = GROUP_SLICES[g]
        total_dist += sub_weights[g] * dist_fn(a[sl], b[sl])
    return float(total_dist)


def compute_distance_weighted(
        a: np.ndarray, b: np.ndarray, metric: str, weights: Dict[str, float], dist_stats: Dict[str, Dict[str, tuple]]
) -> float:
    """Computes distance by standardizing per-group distances before weighting."""
    groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
    sub_weights = normalize_weights(weights, groups)
    metric_stats = dist_stats.get(metric, {})

    dist_fn_map: Dict[str, Callable] = {
        'euclidean': l2, 'manhattan': l1, 'cosine': cosine_dist,
        'chi-squared': chi_squared_hist, 'kullback-leibler': kl_sym_hist,
        'emd': emd_1d_hist, 'cross-bin': cross_bin_hist,
    }
    dist_fn = dist_fn_map.get(metric)
    if not dist_fn:
        raise ValueError(f"Unknown metric: {metric}")

    total_dist = 0.0
    for g in groups:
        sl = GROUP_SLICES[g]
        raw_dist = dist_fn(a[sl], b[sl])

        mean, std = metric_stats.get(g, (0.0, 1.0))
        norm_dist = (raw_dist - mean) / std if std > 0 else 0.0

        total_dist += sub_weights[g] * norm_dist
    return float(total_dist)


# -----------------------------
# Shape Search Engine
# -----------------------------
class ShapeSearchEngine:
    def __init__(self, feature_csv_path: str, obj_root_dir: str, group_weights: Dict[str, float],
                 weighting_method: str = 'feature'):
        self.obj_root_dir = obj_root_dir
        self.metric_map = {
            'euclidean': 'euclidean', 'manhattan': 'manhattan', 'cosine': 'cosine',
            'emd': 'emd', 'chi-squared': 'chi-squared',
            'kullback-leibler': 'kullback-leibler', 'cross-bin': 'cross-bin',
        }
        self.group_weights = group_weights
        self.weighting_method = weighting_method
        self.dist_stats: Dict[str, Dict[str, tuple]] = {}

        try:
            # Use the load_dataset function to read and parse the CSV
            df, raw_feats, categories, obj_ids = load_dataset(feature_csv_path, models_dir=obj_root_dir)
        except FileNotFoundError:
            raise RuntimeError(f"Feature CSV file not found at '{feature_csv_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to read or parse CSV file: {e}")

        if raw_feats.shape[0] == 0:
            raise RuntimeError("No features loaded from CSV file. Check file content.")

        # The 'labels' should be in the format 'category/object_id' for the UI to work correctly
        self.labels = [os.path.join(cat, obj) for cat, obj in zip(categories, obj_ids)]
        self.raw_features = raw_feats
        self.features = normalize_features(self.raw_features)

        if self.weighting_method == 'distance':
            print("Pre-computing distance statistics for all feature groups...")
            self._precompute_distance_stats()
            print("✅ Distance statistics computed.")

    def _precompute_distance_stats(self):
        """Calculates mean and std of distances for each group and metric."""
        num_samples = 1000  # Use a subset to avoid O(N^2) cost
        num_shapes = self.features.shape[0]
        rng = np.random.default_rng(42)
        indices = rng.choice(num_shapes, size=min(num_samples, num_shapes), replace=False)
        sampled_features = self.features[indices]

        dist_fn_map: Dict[str, Callable] = {
            'euclidean': l2, 'manhattan': l1, 'cosine': cosine_dist,
            'chi-squared': chi_squared_hist, 'kullback-leibler': kl_sym_hist,
            'emd': emd_1d_hist, 'cross-bin': cross_bin_hist,
        }

        for metric, dist_fn in dist_fn_map.items():
            self.dist_stats[metric] = {}
            groups = HIST_KEYS if metric in HIST_ONLY_METRICS else FEATURE_GROUP_ORDER
            for g in groups:
                sl = GROUP_SLICES[g]
                group_vectors = sampled_features[:, sl]

                dists = []
                for i in range(len(group_vectors)):
                    for j in range(i + 1, len(group_vectors)):
                        dists.append(dist_fn(group_vectors[i], group_vectors[j]))

                if dists:
                    self.dist_stats[metric][g] = (np.mean(dists), np.std(dists))
                else:
                    self.dist_stats[metric][g] = (0.0, 1.0)

    def search(self, query_label: str, top_n: int = 5, metric: str = 'euclidean') -> List[str]:
        if query_label not in self.labels:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")

        query_idx = self.labels.index(query_label)
        metric_name = self.metric_map.get(metric)
        if not metric_name:
            raise ValueError(f"Unknown metric: {metric}")

        if self.weighting_method == 'distance':
            q_vec = self.features[query_idx]
            compute_fn = compute_distance_weighted
            features_to_compare = self.features
            extra_args = {'dist_stats': self.dist_stats}
        else:  # 'feature' weighting
            q_vec = self.features[query_idx]
            compute_fn = compute_weighted_distance
            features_to_compare = self.features
            extra_args = {}

        dists = np.array([
            compute_fn(q_vec, vec, metric_name, self.group_weights, **extra_args)
            for vec in features_to_compare
        ], dtype=float)

        order = np.argsort(dists)
        results: List[str] = []
        for idx in order:
            if idx == query_idx:
                continue
            results.append(self.labels[idx])
            if len(results) >= top_n:
                break
        return results


# -----------------------------
# Visualization UI
# -----------------------------
class ShapeSearchWindow(QWidget):
    def __init__(self, engine: ShapeSearchEngine, default_metric: str = 'euclidean'):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("Shape Search")
        layout = QVBoxLayout(self)

        # Row setup
        cat_row, file_row, met_row = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        layout.addLayout(cat_row)
        layout.addLayout(file_row)
        layout.addLayout(met_row)

        # Category
        cat_row.addWidget(QLabel("Category:"))
        self.cmb_category = QComboBox()
        cat_row.addWidget(self.cmb_category)

        # File
        file_row.addWidget(QLabel("File:"))
        self.cmb_file = QComboBox()
        file_row.addWidget(self.cmb_file)

        # Metric
        met_row.addWidget(QLabel("Metric:"))
        self.cmb_metric = QComboBox()
        self.cmb_metric.addItems(sorted(self.engine.metric_map.keys()))
        if default_metric in self.engine.metric_map:
            self.cmb_metric.setCurrentText(default_metric)
        met_row.addWidget(self.cmb_metric)

        # Actions
        self.btn_search = QPushButton("Search")
        layout.addWidget(self.btn_search)

        # Self-distance test button
        self.btn_self_distance = QPushButton("Test Self-Distance")
        layout.addWidget(self.btn_self_distance)
        self.btn_self_distance.clicked.connect(self._on_test_self_distance)

        # Data and hooks
        self.category_to_labels = self._build_category_map(self.engine.labels)
        self.cmb_category.addItems(sorted(self.category_to_labels.keys()))
        self.cmb_category.currentTextChanged.connect(self._on_category_changed)
        self.btn_search.clicked.connect(self._on_search_clicked)

        if self.cmb_category.count() > 0:
            self._on_category_changed(self.cmb_category.currentText())

    def _build_category_map(self, labels: List[str]) -> Dict[str, List[str]]:
        cat_map: Dict[str, List[str]] = {}
        for label in labels:
            parts = label.split(os.sep)
            cat = parts[0] if len(parts) > 1 else "(root)"
            cat_map.setdefault(cat, []).append(label)
        return cat_map

    def _on_category_changed(self, category: str):
        self._current_labels = sorted(self.category_to_labels.get(category, []))
        self.cmb_file.clear()
        self.cmb_file.addItems([os.path.basename(label) for label in self._current_labels])

    def _on_search_clicked(self):
        idx = self.cmb_file.currentIndex()
        if idx < 0 or not hasattr(self, "_current_labels") or not self._current_labels:
            QMessageBox.warning(self, "Warning", "Please select a valid category and file.")
            return
        query_label = self._current_labels[idx]
        metric = self.cmb_metric.currentText()
        try:
            results = self.engine.search(query_label, top_n=5, metric=metric)
            if not results:
                QMessageBox.information(self, "Info", "No similar models found.")
                return
            show_results_ui(query_label, results, self.engine.obj_root_dir, title_suffix=metric)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed:\n{e}")

    def _on_test_self_distance(self):
        # Compute and display distance of the selected item to itself
        idx = self.cmb_file.currentIndex()
        if idx < 0 or not hasattr(self, "_current_labels") or not self._current_labels:
            QMessageBox.warning(self, "Warning", "Please select a valid category and file.")
            return

        query_label = self._current_labels[idx]
        metric = self.cmb_metric.currentText()
        query_idx = self.engine.labels.index(query_label)

        if self.engine.weighting_method == 'distance':
            q_vec = self.engine.features[query_idx]
            dist = compute_distance_weighted(q_vec, q_vec, metric, self.engine.group_weights, self.engine.dist_stats)
        else:
            q_vec = self.engine.features[query_idx]
            dist = compute_weighted_distance(q_vec, q_vec, metric, self.engine.group_weights)

        QMessageBox.information(self, "Self Distance", f"Distance to itself: {dist:.6f}")

        # Show only the query in the viewer
        show_results_ui(query_label, [], self.engine.obj_root_dir, title_suffix=f"Self distance: {dist:.6f}")


def show_results_ui(query_label: str, similar_labels: List[str], obj_root_dir: str, title_suffix: str):
    query_obj_path = get_obj_path(query_label, obj_root_dir)
    similar_obj_paths = [get_obj_path(label, obj_root_dir) for label in similar_labels]
    plt = Plotter(shape=(1, 6), sharecam=False, title=f"Shape Search ({title_suffix})")

    plt.at(0).show(
        Mesh(query_obj_path).c("blue").normalize() if os.path.exists(query_obj_path) else Text2D("Query not found"),
        Text2D("Query", pos="bottom-center", s=0.8)
    )
    for i, obj_path in enumerate(similar_obj_paths):
        plt.at(i + 1).show(
            Mesh(obj_path).c("green").normalize() if os.path.exists(obj_path) else Text2D("Model not found"),
            Text2D(f"Result #{i + 1}", pos="bottom-center", s=0.8)
        )
    plt.interactive()


# -----------------------------
# Main Configuration and Execution
# -----------------------------
if __name__ == '__main__':
    # --- 1. CONFIGURE FILE PATHS ---
    # The script now reads from a CSV file.
    FEATURE_CSV = 'ShapeDatabase_INFOMR-master/all_features.csv'
    OBJ_ROOT_DIR = 'ShapeDatabase_INFOMR-master/normalized_5000'

    # --- 2. CHOOSE WEIGHTING METHOD ---
    # 'feature': Original method. Weights feature groups before distance calculation.
    # 'distance': New method. Standardizes per-group distances before applying weights.
    #
    # WEIGHTING_METHOD = 'distance'
    WEIGHTING_METHOD = 'feature'  # This is the original method

    # --- 3. ADJUST FEATURE WEIGHTS HERE ---
    # The weights will be automatically normalized to sum to 1.
    # You can give a feature zero weight by setting it to 0.0.
    MANUAL_WEIGHTS = {
        # Histograms
        'A3': 2.0,
        'D1': 1.0,
        'D2': 2.0,
        'D3': 2.0,
        'D4': 2.0,
        # Scalars
        'Surface area': 1.0,
        'Sphericity': 1.0,
        'Rectangularity': 1.0,
        'Diameter': 1.0,
        'Convexity': 1.0,
        'Eccentricity': 1.0,
    }

    # --- 4. SET DEFAULTS ---
    DEFAULT_METRIC = 'euclidean'

    # --- EXECUTION ---
    if not os.path.exists(FEATURE_CSV):
        # This check is now inside the ShapeSearchEngine, but we keep a basic one here for early exit.
        QMessageBox.critical(None, "Error", f"Feature file not found at '{FEATURE_CSV}'")
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        # Pass the CSV path to the engine
        engine = ShapeSearchEngine(FEATURE_CSV, OBJ_ROOT_DIR, MANUAL_WEIGHTS, weighting_method=WEIGHTING_METHOD)
        win = ShapeSearchWindow(engine, default_metric=DEFAULT_METRIC)
        win.resize(520, 160)
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Initialization Error", f"Failed to initialize the application:\n{e}")
        sys.exit(1)
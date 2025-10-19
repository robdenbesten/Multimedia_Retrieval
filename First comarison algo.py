# python
import os
import json
import sys
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
CROSS_BIN_SIGMA = 1.5   # neighborhood width in bins (tune as needed)
CROSS_BIN_P = 1.0       # p in the formula; try 1.0 or 2.0

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
def parse_feature_from_entry(entry: Dict) -> np.ndarray:
    """Parses a single entry from the JSON file into a raw feature vector."""
    # 1. Histograms
    all_hist_values = []
    for key in HIST_KEYS:
        vals = entry.get('histograms', {}).get(key, [0.0] * HIST_BINS)
        if not vals or len(vals) != HIST_BINS:
            vals = [0.0] * HIST_BINS
        all_hist_values.extend(vals)
    # 2. Scalar features
    scalar_values = []
    for key in SCALAR_KEYS:
        scalar_values.append(float(entry.get('metrics', {}).get(key, 0.0)))
    # 3. Combine
    feature_vector = np.array(all_hist_values + scalar_values, dtype=float)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vector

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

# -----------------------------
# Shape Search Engine
# -----------------------------
class ShapeSearchEngine:
    def __init__(self, feature_json_path: str, obj_root_dir: str, group_weights: Dict[str, float]):
        self.obj_root_dir = obj_root_dir
        self.labels: List[str] = []
        self.metric_map = {
            'euclidean': 'euclidean',
            'manhattan': 'manhattan',
            'cosine': 'cosine',
            'emd': 'emd',
            'chi-squared': 'chi-squared',
            'kullback-leibler': 'kullback-leibler',
            'cross-bin': 'cross-bin',
        }
        self.group_weights = group_weights

        with open(feature_json_path, 'r') as f:
            data = json.load(f)

        raw_feats = []
        for label, entry in data.items():
            self.labels.append(label)
            raw_feats.append(parse_feature_from_entry(entry))

        if not raw_feats:
            raise RuntimeError("No features loaded from JSON file.")

        # Normalize features at initialization
        self.features = normalize_features(np.stack(raw_feats, axis=0))

    def search(self, query_label: str, top_n: int = 5, metric: str = 'euclidean') -> List[str]:
        if query_label not in self.labels:
            raise ValueError(f"Query label '{query_label}' not found in dataset.")

        query_idx = self.labels.index(query_label)
        q_vec = self.features[query_idx]
        metric_name = self.metric_map.get(metric)
        if not metric_name:
            raise ValueError(f"Unknown metric: {metric}")

        dists = np.array([
            compute_weighted_distance(q_vec, vec, metric_name, self.group_weights)
            for vec in self.features
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
    FEATURE_JSON = 'ShapeDatabase_INFOMR-master/features.json'
    OBJ_ROOT_DIR = 'ShapeDatabase_INFOMR-master/normalized_5000'

    # --- 2. ADJUST FEATURE WEIGHTS HERE ---
    # The weights will be automatically normalized to sum to 1.
    # You can give a feature zero weight by setting it to 0.0.
    MANUAL_WEIGHTS = {
        # Histograms
        'A3': 1.5,
        'D1': 2.0,
        'D2': 1.5,
        'D3': 1.0,
        'D4': 1.0,
        # Scalars
        'Surface area': 1.2,
        'Sphericity': 1.0,
        'Rectangularity': 1.0,
        'Diameter': 1.0,
        'Convexity': 1.0,
        'Eccentricity': 1.0,
    }

    # --- 3. SET DEFAULTS ---
    DEFAULT_METRIC = 'euclidean'

    # --- EXECUTION ---
    if not os.path.exists(FEATURE_JSON):
        QMessageBox.critical(None, "Error", f"Feature file not found at '{FEATURE_JSON}'")
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        engine = ShapeSearchEngine(FEATURE_JSON, OBJ_ROOT_DIR, MANUAL_WEIGHTS)
        win = ShapeSearchWindow(engine, default_metric=DEFAULT_METRIC)
        win.resize(520, 160)
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Initialization Error", f"Failed to initialize the application:\n{e}")
        sys.exit(1)
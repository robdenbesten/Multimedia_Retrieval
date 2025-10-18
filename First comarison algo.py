import os
import re
import sys
import numpy as np
from typing import Callable, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QApplication
)
from vedo import Plotter, Mesh, Text2D

# -----------------------------
# Feature parsing (histograms only)
# -----------------------------
def parse_hist_file(file_path: str) -> np.ndarray:
    """
    Parse all 5 hist blocks into a single vector:
      A3_hist, D1_hist, D2_hist, D3_hist, D4_hist
    Robust to separators and whitespace. Pads missing blocks with zeros.
    """
    with open(file_path, 'r', newline='') as f:
        content = f.read()

    def extract_block(key: str) -> str:
        pattern = rf'{key}\s*:\s*([\s\S]*?)(?=\n[A-Z]\d_hist\s*:|\Z)'
        m = re.search(pattern, content, flags=re.MULTILINE)
        return m.group(1) if m else ''

    def parse_block(block: str) -> List[float]:
        block = block.replace('\r', '\n')
        tokens = re.split(r'[,\s;]+', block.strip())
        vals = []
        for t in tokens:
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                pass
        return vals

    all_hist_values = []
    for key in ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']:
        vals = parse_block(extract_block(key))
        if not vals:
            vals = [0.0] * 10
        all_hist_values.extend(vals)

    arr = np.array(all_hist_values, dtype=float)
    # Safety: ensure non-negative for EMD and numerical stability
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    return arr


def get_all_feature_files(root_dir: str) -> List[str]:
    files = []
    for subdir, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.txt'):
                files.append(os.path.join(subdir, fname))
    return files


def get_obj_path(feature_file_path: str) -> str:
    base_path = feature_file_path.replace('features_test', 'original database')
    if '_copy.txt' in base_path:
        return base_path.replace('_copy.txt', '.obj')
    return base_path.replace('.txt', '.obj')


# -----------------------------
# Distance functions
# -----------------------------
def distance_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance on equal-length vectors."""
    if a.shape[0] != b.shape[0]:
        L = min(a.shape[0], b.shape[0])
        a, b = a[:L], b[:L]
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def distance_manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance on equal-length vectors."""
    if a.shape[0] != b.shape[0]:
        L = min(a.shape[0], b.shape[0])
        a, b = a[:L], b[:L]
    return float(np.sum(np.abs(a - b)))


def distance_emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    1D Earth Mover's Distance (EMD) for equal-binned histograms.
    Implements EMD = sum(|CDF_a - CDF_b|) with unit bin cost.
    Assumes non-negative entries. Normalizes to L1 mass=1 to be scale-invariant.
    If length differs, uses the common prefix.
    """
    if a.shape[0] != b.shape[0]:
        L = min(a.shape[0], b.shape[0])
        a, b = a[:L], b[:L]

    a = np.clip(a, 0.0, np.inf)
    b = np.clip(b, 0.0, np.inf)
    sa = float(np.sum(a))
    sb = float(np.sum(b))
    if sa > 0:
        a = a / sa
    if sb > 0:
        b = b / sb
    cdf_a = np.cumsum(a)
    cdf_b = np.cumsum(b)
    return float(np.sum(np.abs(cdf_a - cdf_b)))


# -----------------------------
# Shape search (histogram-only)
# -----------------------------
class DistanceOnlyShapeSearch:
    def __init__(self, feature_dir: str):
        self.feature_dir = feature_dir
        self.files = get_all_feature_files(feature_dir)
        if not self.files:
            raise RuntimeError("No feature files found.")

        # Load histograms
        hists = [parse_hist_file(f) for f in self.files]
        # Make equal length by trimming to the shortest vector (robust across datasets)
        min_len = min(h.shape[0] for h in hists)
        self.histograms = np.stack([h[:min_len] for h in hists], axis=0)
        self.length = min_len

        # Optional: store per-file basename for filtering query out
        self._obj_basenames = [os.path.basename(get_obj_path(f)) for f in self.files]

        # Map string metric names to callables
        self.metric_map = {
            'euclidean': distance_euclidean,
            'manhattan': distance_manhattan,
            'emd': distance_emd_1d,
        }

    def search(self, input_file: str, top_n: int = 5, metric: str | Callable[[np.ndarray, np.ndarray], float] = 'euclidean') -> List[str]:
        q = parse_hist_file(input_file)
        q = q[:self.length] if q.shape[0] >= self.length else np.pad(q, (0, self.length - q.shape[0]), mode='constant')

        if isinstance(metric, str):
            if metric not in self.metric_map:
                raise ValueError(f"Unknown metric: {metric}")
            dist_fn = self.metric_map[metric]
        else:
            dist_fn = metric

        # Compute distances
        dists = np.array([dist_fn(q, h) for h in self.histograms], dtype=float)

        # Exclude the same basename as the query (if exists)
        query_base = os.path.basename(get_obj_path(input_file))
        order = np.argsort(dists)
        results = []
        seen = set()
        for idx in order:
            base = self._obj_basenames[idx]
            if base == query_base:
                continue
            if base in seen:
                continue
            results.append(self.files[idx])
            seen.add(base)
            if len(results) == top_n:
                break
        return results


# -----------------------------
# Visualization UI (metric selectable)
# -----------------------------
def show_results_ui(query_feature_file: str, similar_feature_files: List[str], title_suffix: str):
    query_obj_path = get_obj_path(query_feature_file)
    similar_obj_paths = [get_obj_path(f) for f in similar_feature_files]

    plt = Plotter(shape=(1, 6), sharecam=False, title=f"Shape Search ({title_suffix})")

    plt.at(0)
    if os.path.exists(query_obj_path):
        query_mesh = Mesh(query_obj_path).c("blue").normalize()
        plt.show(query_mesh, Text2D("Query", pos="bottom-center", s=0.8))
    else:
        plt.show(Text2D(f"Query not found:\n{os.path.basename(query_obj_path)}", pos="center", s=0.7))

    for i, obj_path in enumerate(similar_obj_paths):
        plt.at(i + 1)
        if os.path.exists(obj_path):
            res_mesh = Mesh(obj_path).c("green").normalize()
            plt.show(res_mesh, Text2D(f"Result #{i + 1}", pos="bottom-center", s=0.8))
        else:
            plt.show(Text2D(f"Model not found:\n{os.path.basename(obj_path)}", pos="center", s=0.7))

    plt.interactive()


class ShapeSearchWindow(QWidget):
    """Minimal UI: choose category, file, and distance metric."""
    def __init__(self, engine: DistanceOnlyShapeSearch, feature_dir: str, default_metric: str = 'euclidean'):
        super().__init__()
        self.engine = engine
        self.feature_dir = feature_dir
        self.setWindowTitle("Shape Search (Distance Only)")

        layout = QVBoxLayout(self)

        # Category
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Category:"))
        self.cmb_category = QComboBox()
        cat_row.addWidget(self.cmb_category)
        layout.addLayout(cat_row)

        # File
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self.cmb_file = QComboBox()
        file_row.addWidget(self.cmb_file)
        layout.addLayout(file_row)

        # Metric
        met_row = QHBoxLayout()
        met_row.addWidget(QLabel("Metric:"))
        self.cmb_metric = QComboBox()
        self.cmb_metric.addItems(['euclidean', 'manhattan', 'emd'])
        if default_metric in ['euclidean', 'manhattan', 'emd']:
            self.cmb_metric.setCurrentText(default_metric)
        met_row.addWidget(self.cmb_metric)
        layout.addLayout(met_row)

        # Action
        self.btn_search = QPushButton("Search")
        layout.addWidget(self.btn_search)

        # Data
        self.category_to_files = self._build_category_map(self.engine.files)
        self.cmb_category.addItems(sorted(self.category_to_files.keys()))

        # Hooks
        self.cmb_category.currentTextChanged.connect(self._on_category_changed)
        self.btn_search.clicked.connect(self._on_search_clicked)

        if self.cmb_category.count() > 0:
            self._on_category_changed(self.cmb_category.currentText())

    def _build_category_map(self, file_paths: List[str]):
        cat_map = {}
        for f in file_paths:
            rel = os.path.relpath(f, self.feature_dir)
            parts = rel.split(os.sep)
            cat = parts[0] if len(parts) > 1 else "(root)"
            cat_map.setdefault(cat, []).append(f)
        return cat_map

    def _on_category_changed(self, category: str):
        files = sorted(self.category_to_files.get(category, []))
        self._current_files = files
        self.cmb_file.clear()
        self.cmb_file.addItems([os.path.basename(f) for f in files])

    def _on_search_clicked(self):
        idx = self.cmb_file.currentIndex()
        if idx < 0 or not hasattr(self, "_current_files") or not self._current_files:
            QMessageBox.warning(self, "Warning", "Please select a valid category and file.")
            return
        input_file = self._current_files[idx]
        metric = self.cmb_metric.currentText()
        try:
            results = self.engine.search(input_file, top_n=5, metric=metric)
            if not results:
                QMessageBox.information(self, "Info", "No similar models found.")
                return
            show_results_ui(input_file, results, title_suffix=metric)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed:\n{e}")


# -----------------------------
# Main (choose the distance function here)
# -----------------------------
if __name__ == '__main__':
    # Select the distance function globally here for default UI selection:
    # Options: 'euclidean', 'manhattan', 'emd'
    DEFAULT_METRIC = 'euclidean'

    FEATURE_DIR = 'ShapeDatabase_INFOMR-master/features_test'

    app = QApplication.instance() or QApplication(sys.argv)
    engine = DistanceOnlyShapeSearch(FEATURE_DIR)
    win = ShapeSearchWindow(engine, FEATURE_DIR, default_metric=DEFAULT_METRIC)
    win.resize(520, 160)
    win.show()
    sys.exit(app.exec())
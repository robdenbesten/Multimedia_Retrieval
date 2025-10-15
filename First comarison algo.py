import os
import re
import sys
import zlib
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox, QApplication
)
from vedo import Plotter, Mesh, Text2D


# --- Helper Functions for Parsing (single, robust versions) ---

def parse_hist_file(file_path):
    """Parse 5 histograms into one vector. Robust to \\r\\n, spaces, and variable separators."""
    with open(file_path, 'r', newline='') as f:
        content = f.read()

    def extract_block(key):
        # Capture text after '<key>:' up to next '<Xn_hist>:' or end
        pattern = rf'{key}\s*:\s*([\s\S]*?)(?=\n[A-Z]\d_hist\s*:|\Z)'
        m = re.search(pattern, content, flags=re.MULTILINE)
        return m.group(1) if m else ''

    def parse_block(block):
        # Split on commas/whitespace/semicolons and parse floats
        block = block.replace('\r', '\n')
        tokens = re.split(r'[,\s;]+', block.strip())
        vals = []
        for t in tokens:
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                # Ignore non-numeric tokens
                continue
        return vals

    all_hist_values = []
    for key in ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']:
        vals = parse_block(extract_block(key))
        if not vals:
            # Fallback to 10 zeros if the block is missing/unreadable
            vals = [0.0] * 10
        all_hist_values.extend(vals)

    return np.array(all_hist_values, dtype=float)


def parse_metrics_file(file_path):
    """Parse scalar metrics and skip histogram blocks."""
    metrics = {}
    with open(file_path, 'r', newline='') as f:
        for line in f:
            if ':' not in line:
                continue
            head = line.split(':', 1)[0].strip()
            if head in {'A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist'}:
                continue
            key, val = line.split(':', 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except (ValueError, TypeError):
                # Skip non-numeric metric values
                continue
    return metrics


def get_all_feature_files(root_dir):
    """Collect all feature .txt files."""
    files = []
    for subdir, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.txt'):
                files.append(os.path.join(subdir, fname))
    return files


def get_obj_path(feature_file_path):
    """Map feature file to corresponding .obj path."""
    base_path = feature_file_path.replace('features_test', 'original database')
    if '_copy.txt' in base_path:
        return base_path.replace('_copy.txt', '.obj')
    return base_path.replace('.txt', '.obj')


# --- Main Search Engine ---

class EnhancedShapeSearchEngine:
    def __init__(self, feature_dir):
        self.files = get_all_feature_files(feature_dir)
        if not self.files:
            raise RuntimeError("No feature files found.")

        # Parse features
        hist_list = []
        metrics_list = []
        metric_keys = None
        for f in self.files:
            hist_list.append(parse_hist_file(f))
            metric_dict = parse_metrics_file(f)
            if metric_keys is None:
                metric_keys = sorted(metric_dict.keys())
            metrics_list.append([metric_dict.get(k, 0.0) for k in metric_keys])

        self.metric_keys = metric_keys
        self.histograms = np.array(hist_list, dtype=float)
        self.metrics = np.array(metrics_list, dtype=float)

        # Sanity: ensure histograms are not all-zero constant
        if self.histograms.ndim != 2 or self.histograms.shape[0] == 0:
            raise RuntimeError("Histogram parsing failed.")
        if np.allclose(np.var(self.histograms, axis=0), 0.0):
            print("Warning: histogram variance is zero across the dataset. Check parsing format.", file=sys.stderr)

        # Pre-normalize metrics and store min/range for query normalization
        self.metrics_min = np.min(self.metrics, axis=0)
        self.metrics_range = np.max(self.metrics, axis=0) - self.metrics_min
        self.metrics_range[self.metrics_range == 0] = 1.0  # Avoid div-by-zero
        self.normalized_metrics = (self.metrics - self.metrics_min) / self.metrics_range

        self._compute_feature_weights()
        self._compute_hist_boundaries()

    def _compute_hist_boundaries(self):
        """Split concatenated hist vector into 5 parts; tolerate non-multiples of 5 by trimming tail."""
        total_len = self.histograms.shape[1]
        base = total_len // 5
        if base == 0:
            raise RuntimeError("Histogram vector too short.")
        # Use first 5*base bins, drop any remainder to keep consistent slicing
        self._usable_hist_len = base * 5
        self.hist_boundaries = [(i * base, (i + 1) * base) for i in range(5)]

    def _align_hist_bins(self, q_hist):
        """Pad or trim query histogram to match dataset's usable length."""
        if q_hist.shape[0] < self._usable_hist_len:
            return np.pad(q_hist, (0, self._usable_hist_len - q_hist.shape[0]), mode='constant')
        if q_hist.shape[0] > self._usable_hist_len:
            return q_hist[:self._usable_hist_len]
        return q_hist

    def _preprocess_query(self, input_file):
        q_hist = parse_hist_file(input_file)
        q_hist = self._align_hist_bins(q_hist.astype(float))
        # Trim dataset histograms as well (if not already)
        if self.histograms.shape[1] != self._usable_hist_len:
            self.histograms = self.histograms[:, :self._usable_hist_len]
        metric_dict = parse_metrics_file(input_file)
        q_metrics = np.array([metric_dict.get(k, 0.0) for k in self.metric_keys], dtype=float)
        return q_hist, q_metrics

    def _compute_feature_weights(self):
        """Weights from variance (normalized) to avoid dominance."""
        hist_variance = np.var(self.histograms, axis=0)
        metrics_variance = np.var(self.normalized_metrics, axis=0)

        hist_sum = float(np.sum(hist_variance))
        if hist_sum > 0:
            self.hist_weights = hist_variance / (hist_sum + 1e-9)
        else:
            self.hist_weights = np.ones_like(hist_variance) / max(1, len(hist_variance))

        metrics_sum = float(np.sum(metrics_variance))
        if metrics_sum > 0:
            self.metrics_weights = metrics_variance / (metrics_sum + 1e-9)
        else:
            self.metrics_weights = np.ones_like(metrics_variance) / max(1, len(metrics_variance))

        # Relative weights across the 5 histogram types
        self.hist_type_weights = {'A3': 0.35, 'D1': 0.30, 'D2': 0.20, 'D3': 0.10, 'D4': 0.05}

    def _chi_square_distance(self, h1, h2):
        """Chi-square distance on normalized histograms, NaN-safe."""
        s1 = np.sum(h1)
        s2 = np.sum(h2)
        h1n = h1 / (s1 + 1e-9)
        h2n = h2 / (s2 + 1e-9)
        denom = h1n + h2n + 1e-9
        d = np.sum(((h1n - h2n) ** 2) / denom)
        return float(np.nan_to_num(d, nan=0.0, posinf=1e9, neginf=1e9))

    def _weighted_histogram_distance(self, h1, h2):
        """Weighted sum of per-type Chi-square distances."""
        total = 0.0
        names = ['A3', 'D1', 'D2', 'D3', 'D4']
        for i, (start, end) in enumerate(self.hist_boundaries):
            dist = self._chi_square_distance(h1[start:end], h2[start:end])
            total += self.hist_type_weights[names[i]] * dist
        return total

    def _adaptive_metrics_distance(self, m1_norm, m2_norm):
        """Weighted Euclidean on pre-normalized metrics, NaN-safe."""
        diff = (m1_norm - m2_norm) * self.metrics_weights
        d = float(np.sqrt(np.sum(diff * diff)))
        return float(np.nan_to_num(d, nan=0.0, posinf=1e9, neginf=1e9))

    def search(self, input_file, top_n=5, coarse_filter_ratio=0.3):
        """Three-stage search with NaN guards and tie-breaker."""
        q_hist, q_metrics = self._preprocess_query(input_file)
        q_metrics_norm = (q_metrics - self.metrics_min) / self.metrics_range

        # Stage 1: metrics prefilter
        metric_dists = np.array(
            [self._adaptive_metrics_distance(q_metrics_norm, m_norm) for m_norm in self.normalized_metrics],
            dtype=float
        )
        metric_dists = np.nan_to_num(metric_dists, nan=1e9, posinf=1e9, neginf=1e9)

        num_candidates = max(int(top_n * 5), int(len(self.files) * coarse_filter_ratio))
        num_candidates = max(num_candidates, top_n)
        candidate_indices = np.argsort(metric_dists)[:num_candidates]

        # Stage 2: histogram distances
        candidate_hists = self.histograms[candidate_indices]
        hist_dists = np.array([self._weighted_histogram_distance(q_hist, h) for h in candidate_hists], dtype=float)
        hist_dists = np.nan_to_num(hist_dists, nan=1e9, posinf=1e9, neginf=1e9)

        # Stage 3: combine (raw distances, no re-normalization)
        cand_metric_dists = metric_dists[candidate_indices]
        combined = 0.75 * hist_dists + 0.25 * cand_metric_dists

        # If all scores equal, add tiny query-seeded jitter to avoid "first N always"
        if np.allclose(combined, combined[0]):
            seed = zlib.adler32(str(input_file).encode('utf-8')) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            combined = combined + 1e-9 * rng.random(combined.shape[0])

        order = np.argsort(combined)
        final_sorted = [(candidate_indices[i], combined[i]) for i in order]

        # Post-filter: exclude query and duplicates by basename
        top_files = []
        seen_basenames = set()
        query_basename = os.path.basename(get_obj_path(input_file))
        for original_idx, _ in final_sorted:
            file_path = self.files[original_idx]
            basename = os.path.basename(get_obj_path(file_path))
            if basename == query_basename:
                continue
            if basename in seen_basenames:
                continue
            top_files.append(file_path)
            seen_basenames.add(basename)
            if len(top_files) == top_n:
                break

        return top_files


# --- Visualization ---

def show_results_ui(query_feature_file, similar_feature_files):
    """Display query and results side-by-side."""
    query_obj_path = get_obj_path(query_feature_file)
    similar_obj_paths = [get_obj_path(f) for f in similar_feature_files]

    plt = Plotter(shape=(1, 6), sharecam=False, title="Shape Search Results")

    plt.at(0)
    if os.path.exists(query_obj_path):
        query_mesh = Mesh(query_obj_path).c("blue").normalize()
        plt.show(query_mesh, Text2D("Query Model", pos="bottom-center", s=0.8))
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


# --- Simple UI ---

class ShapeSearchWindow(QWidget):
    """Minimal UI to pick a feature file and run search."""
    def __init__(self, engine, feature_dir):
        super().__init__()
        self.engine = engine
        self.feature_dir = feature_dir
        self.setWindowTitle("Shape Search")

        layout = QVBoxLayout(self)
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Category:"))
        self.cmb_category = QComboBox()
        cat_row.addWidget(self.cmb_category)
        layout.addLayout(cat_row)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self.cmb_file = QComboBox()
        file_row.addWidget(self.cmb_file)
        layout.addLayout(file_row)

        self.btn_search = QPushButton("Search")
        layout.addWidget(self.btn_search)

        self.category_to_files = self._build_category_map(self.engine.files)
        self.cmb_category.addItems(sorted(self.category_to_files.keys()))
        self.cmb_category.currentTextChanged.connect(self._on_category_changed)
        self.btn_search.clicked.connect(self._on_search_clicked)
        if self.cmb_category.count() > 0:
            self._on_category_changed(self.cmb_category.currentText())

    def _build_category_map(self, file_paths):
        cat_map = {}
        for f in file_paths:
            rel = os.path.relpath(f, self.feature_dir)
            parts = rel.split(os.sep)
            cat = parts[0] if len(parts) > 1 else "(root)"
            cat_map.setdefault(cat, []).append(f)
        return cat_map

    def _on_category_changed(self, category):
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
        try:
            results = self.engine.search(input_file, top_n=5)
            if not results:
                QMessageBox.information(self, "Info", "No similar models found.")
                return
            show_results_ui(input_file, results)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed:\n{e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    app = QApplication.instance() or QApplication(sys.argv)
    FEATURE_DIR = 'ShapeDatabase_INFOMR-master/features_test'
    engine = EnhancedShapeSearchEngine(FEATURE_DIR)
    win = ShapeSearchWindow(engine, FEATURE_DIR)
    win.resize(480, 140)
    win.show()
    sys.exit(app.exec())
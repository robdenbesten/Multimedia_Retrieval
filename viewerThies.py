import sys
import os
import shutil
import csv
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget, QCheckBox, QGridLayout)
from PyQt6.QtCore import Qt, QCoreApplication
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, load, Axes

from normalise import Mesh
from feature_extraction import extract_features_for_single_mesh, make_fixed_bin_edges, DEFAULT_BINS
from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS, WEIGHTING_METHOD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scalability import find_nearest_neighbours_knn, compute_tsne

DATABASE_LOCATION = r'Normalised-objects'
TEMP_OUTPUT_DIR = 'TEMP_OUTPUT'
FEATURE_CSV = 'Feature-matrix/all_features.csv'
STATS_JSON = 'Feature-matrix/normalization_stats.json'


class MeshViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh_object = None
        self.current_file = None
        self.normalized_file = None
        self.current_features = None # To store raw features of the loaded mesh
        self.database_location = DATABASE_LOCATION
        self.show_normalized = False

        self.edges = make_fixed_bin_edges(DEFAULT_BINS)
        self.searcher = None
        self.knn_ready = False
        self.knn_model = None
        self.knn_X2d = None
        self.knn_features_df = None
        self.result_plotters = []

        self.init_searcher()
        self.init_ui()

    def init_searcher(self):
        """Initializes the ShapeSearcher with features and stats."""
        try:
            self.searcher = ShapeSearcher(
                feature_csv_path=FEATURE_CSV,
                stats_path=STATS_JSON,
                weights=MANUAL_WEIGHTS,
                weighting_method=WEIGHTING_METHOD
            )
        except Exception as e:
            print(f"Failed to initialize searcher: {e}")

    # --- (init_ui and other UI methods remain largely the same) ---
    def init_ui(self):
        self.setWindowTitle('Mesh Viewer and Search')
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)

        # Category dropdown
        control_layout.addWidget(QLabel('Select Category:'))
        self.category_dropdown = QComboBox()
        self.category_dropdown.currentTextChanged.connect(self.load_category_objects)
        control_layout.addWidget(self.category_dropdown)

        # Object list
        control_layout.addWidget(QLabel('Objects:'))
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.load_mesh_from_list)
        control_layout.addWidget(self.object_list)

        # Status label
        self.status_label = QLabel('Select a category and object')
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        # Normalize checkbox
        self.normalize_checkbox = QCheckBox('Show Normalized')
        self.normalize_checkbox.stateChanged.connect(self.toggle_normalize_view)
        self.normalize_checkbox.setEnabled(False)
        control_layout.addWidget(self.normalize_checkbox)

        # --- Search Controls ---
        control_layout.addWidget(QLabel('Search Metric:'))
        self.metric_dropdown = QComboBox()
        if self.searcher:
            self.metric_dropdown.addItems(self.searcher.metrics)
        # ensure correct t-SNE entry (fix previous typo 'kNN tsen')
        if self.metric_dropdown.findText('kNN tsne') == -1:
            self.metric_dropdown.addItem('kNN tsne')
        control_layout.addWidget(self.metric_dropdown)

        self.search_button = QPushButton('Find Similar')
        self.search_button.clicked.connect(self.find_similar_shapes)
        self.search_button.setEnabled(False)
        control_layout.addWidget(self.search_button)

        # Right panel for viewers
        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)

        # Main VTK Widget
        self.vtk_widget = QVTKRenderWindowInteractor(viewer_panel)
        self.plotter = Plotter(qt_widget=self.vtk_widget, N=1, bg='white')
        viewer_layout.addWidget(self.vtk_widget, stretch=2) # Give more space to main viewer

        # Results viewers
        results_widget = QWidget()
        results_layout = QGridLayout(results_widget)
        for i in range(5):
            vtk_res_widget = QVTKRenderWindowInteractor(results_widget)
            plot = Plotter(qt_widget=vtk_res_widget, N=1, bg='lightgrey')
            self.result_plotters.append(plot)
            results_layout.addWidget(vtk_res_widget, 0, i)
        viewer_layout.addWidget(results_widget, stretch=1)

        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(viewer_panel)

        self.add_axes()
        self.load_categories()

        if not self.searcher:
            self.status_label.setText(f"Error: Could not find '{FEATURE_CSV}' or '{STATS_JSON}'.\nSearch is disabled.")
            self.metric_dropdown.setEnabled(False)
            self.search_button.setEnabled(False)

    def add_axes(self):
        axes = Axes(xtitle='x', ytitle='y', ztitle='z')
        self.plotter.add(axes)
        self.origin_axes = axes

    def load_categories(self):
        try:
            if os.path.exists(self.database_location):
                categories = [d for d in os.listdir(self.database_location) if os.path.isdir(os.path.join(self.database_location, d))]
                self.category_dropdown.addItems(sorted(categories))
        except Exception as e:
            self.status_label.setText(f'Error loading categories:\n{e}')

    def init_knn_index(self, n_neighbors=11, tsne_perplexity=30, cache_dir='cache'):
        """Build or load t-SNE embedding + k-NN index from `FEATURE_CSV` (lazy)."""
        if self.knn_ready:
            return
        try:
            if not os.path.exists(FEATURE_CSV):
                raise FileNotFoundError(f"`{FEATURE_CSV}` not found.")

            df = pd.read_csv(FEATURE_CSV)
            if 'Object' not in df.columns or 'Category' not in df.columns:
                raise ValueError("Feature CSV must contain `Object` and `Category` columns.")

            # Create a column with the basename of the stored object path (split Object into category + name)
            df['ObjectName'] = df['Object'].astype(str).apply(lambda p: os.path.basename(p))

            feat_cols = [c for c in df.columns if c not in ('Object', 'Category', 'ObjectName')]
            if not feat_cols:
                raise ValueError("No feature columns found in feature CSV.")

            X = df[feat_cols].values.astype(float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = StandardScaler().fit_transform(X)

            # compute_tsne from scalability provides caching when called with csv_file
            X_2d = compute_tsne(
                X_scaled,
                csv_file=FEATURE_CSV,
                cache_dir=cache_dir,
                n_components=2,
                perplexity=tsne_perplexity
            )

            nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            nn_model.fit(X_2d)

            # store for later use
            self.knn_features_df = df
            self.knn_X2d = X_2d
            self.knn_model = nn_model
            self.knn_ready = True
            print("k-NN / t-SNE index ready.")
        except Exception as e:
            self.knn_ready = False
            print(f"Failed to build k-NN index: {e}")

    def load_category_objects(self, category):
        self.object_list.clear()
        if not category: return
        try:
            category_path = os.path.join(self.database_location, category)
            objects = sorted([f for f in os.listdir(category_path) if f.endswith('.obj')])
            self.object_list.addItems(objects)
        except Exception as e:
            self.status_label.setText(f'Error loading objects:\n{e}')

    def load_mesh_from_list(self, item):
        category = self.category_dropdown.currentText()
        if not category: return
        self.load_mesh(os.path.join(self.database_location, category, item.text()))

    def load_mesh(self, file_path):
        if not file_path: return
        try:
            self.current_file = file_path
            self.normalized_file = None
            self.current_features = None
            self.normalize_checkbox.setEnabled(False)
            self.search_button.setEnabled(False)

            self.status_label.setText(f'Normalizing...\n{os.path.basename(file_path)}')
            QCoreApplication.processEvents()
            mesh = Mesh(file_path)
            mesh.full_normalize()
            self.normalized_file = mesh.save()

            self.status_label.setText(f'Extracting features...')
            QCoreApplication.processEvents()
            self.extract_features() # This will set self.current_features

            self.normalize_checkbox.setEnabled(True)
            if self.searcher and self.current_features is not None:
                self.search_button.setEnabled(True)

            self.display_mesh(self.normalized_file if self.show_normalized else self.current_file)
            self.update_status_label(mesh.vertex_count())
        except Exception as e:
            self.status_label.setText(f'Error loading:\n{e}')

    def extract_features(self):
        """Extracts raw features from the normalized mesh and stores them."""
        if not self.normalized_file: return
        try:
            category = self.category_dropdown.currentText()
            obj_name = os.path.basename(self.current_file)
            rel_path = f"{category}/{obj_name}"

            _, success, result = extract_features_for_single_mesh(
                obj_path=self.normalized_file, rel_path=rel_path,
                edges=self.edges, n_samples=250000, surface_points=5000
            )
            if success and isinstance(result, list):
                # Store the raw numeric features (skip object name and category)
                self.current_features = np.array(result[2:], dtype=float)
                print("Successfully extracted features for the current mesh.")
            else:
                self.current_features = None
                print(f"Feature extraction failed: {result}")
        except Exception as e:
            self.current_features = None
            print(f"Error in extract_features: {e}")

    def find_similar_shapes(self):
        """Performs search using stored raw features or the k-NN t-SNE option."""
        if not self.searcher and not os.path.exists(FEATURE_CSV):
            self.status_label.setText("Searcher not ready and no feature CSV available.")
            return
        if self.current_features is None:
            self.status_label.setText("No features extracted for current mesh.")
            return

        metric = self.metric_dropdown.currentText()
        self.status_label.setText(f"Searching ({metric})...")
        QCoreApplication.processEvents()

        try:
            if metric == 'kNN tsne':
                # build/load index if needed
                self.init_knn_index()

                if not self.knn_ready or self.knn_features_df is None:
                    self.status_label.setText("k-NN index not available.")
                    return

                # Use category + basename to find the CSV row (split Object into Category + ObjectName)
                category = self.category_dropdown.currentText()
                obj_name = os.path.basename(self.current_file)  # includes extension
                matches = self.knn_features_df.index[
                    (self.knn_features_df['ObjectName'] == obj_name) &
                    (self.knn_features_df['Category'] == category)
                    ].tolist()

                # fallback: try the full rel_path match if no match found
                if not matches:
                    rel_path = f"{category}/{obj_name}"
                    matches = self.knn_features_df.index[self.knn_features_df['Object'] == rel_path].tolist()

                if not matches:
                    self.status_label.setText(
                        f"Object `{obj_name}` not found in `{FEATURE_CSV}`; k-NN requires the object to be in the feature CSV.")
                    return

                query_index = matches[0]
                nearest_df, neighbor_indices = find_nearest_neighbours_knn(
                    self.knn_features_df, self.knn_model, query_index, k=5, X_2d=self.knn_X2d
                )

                # display top results (nearest_df['Object'] contains rel_path strings)
                results = nearest_df['Object'].tolist()
                for i, res_label in enumerate(results):
                    res_path = os.path.join(self.database_location, res_label)
                    self.display_result_mesh(res_path, i)

                for i in range(len(results), 5):
                    self.result_plotters[i].clear().render()

                self.status_label.setText(
                    f"Top {len(results)} results (k-NN t-SNE) for {os.path.basename(self.current_file)}")

            else:
                # fall back to existing searcher behavior
                results = self.searcher.search_by_vector(
                    query_vector=self.current_features,
                    metric=metric,
                    top_n=5
                )

                for i, res_label in enumerate(results):
                    res_path = os.path.join(self.database_location, res_label)
                    self.display_result_mesh(res_path, i)

                for i in range(len(results), 5):
                    self.result_plotters[i].clear().render()

                self.status_label.setText(f"Top 5 results for {os.path.basename(self.current_file)}")
        except Exception as e:
            self.status_label.setText(f"An unexpected error occurred during search: {e}")

    def display_result_mesh(self, file_path, index):
        if index >= len(self.result_plotters):
            return
        plot = self.result_plotters[index]
        plot.clear()
        try:
            resolved = os.path.normpath(file_path)
            if not os.path.exists(resolved):
                # attempt to find by basename under the database folder
                basename = os.path.basename(resolved)
                found = None
                for root, _, files in os.walk(self.database_location):
                    if basename in files:
                        found = os.path.join(root, basename)
                        break
                if found and os.path.exists(found):
                    print(f"Found result mesh by basename: `{found}`")
                    resolved = found
                else:
                    print(f"Result mesh not found: `{file_path}` (resolved: `{resolved}`)")
                    return

            loaded = load(resolved)
            if loaded is None:
                print(f"vedo.load returned None for `{resolved}`")
                return

            mesh = loaded[0] if isinstance(loaded, (list, tuple)) and len(loaded) > 0 else loaded

            if not hasattr(mesh, 'lighting'):
                print(f"Loaded object for `{resolved}` is not a vedo mesh (type: {type(mesh)})")
                return

            mesh = mesh.lighting('default').linecolor('black').linewidth(0.5)
            plot.show(mesh, resetcam=True)
        except Exception as e:
            print(f"Error displaying result mesh `{file_path}`: {e}")

    def display_mesh(self, file_path):
        try:
            resolved = os.path.normpath(file_path)
            if not os.path.exists(resolved):
                self.status_label.setText(f'Error displaying mesh: file not found `{file_path}`')
                return

            self.plotter.clear().add(self.origin_axes)
            loaded = load(resolved)
            if loaded is None:
                self.status_label.setText(f'Error displaying mesh: vedo.load returned None for `{file_path}`')
                return

            mesh = loaded[0] if isinstance(loaded, (list, tuple)) and len(loaded) > 0 else loaded
            if not hasattr(mesh, 'lighting'):
                self.status_label.setText(f'Error displaying mesh: unsupported object type `{type(mesh)}`')
                return

            mesh = mesh.lighting('default').linecolor('black').linewidth(1)
            self.plotter.show(mesh, resetcam=True)
        except Exception as e:
            self.status_label.setText(f'Error displaying mesh:\n{e}')

    def toggle_normalize_view(self):
        if not self.current_file: return
        self.show_normalized = self.normalize_checkbox.isChecked()
        self.display_mesh(self.normalized_file if self.show_normalized else self.current_file)
        self.update_status_label()

    def update_status_label(self, v_count='N/A'):
        if not self.current_file: return
        state = 'Normalized' if self.show_normalized else 'Original'
        self.status_label.setText(f'{state} [{v_count} vertices]:\n{os.path.basename(self.current_file)}')

    def closeEvent(self, event):
        if os.path.exists(TEMP_OUTPUT_DIR):
            shutil.rmtree(TEMP_OUTPUT_DIR)
        event.accept()

def main():
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

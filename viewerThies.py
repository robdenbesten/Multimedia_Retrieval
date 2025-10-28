# python
import sys
import os
import shutil
import csv
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget, QCheckBox, QGridLayout)
from PyQt6.QtCore import Qt, QCoreApplication
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, load, Line, Axes

# Import the Mesh class and feature extraction
from normalise import Mesh
from feature_extraction import extract_features_for_single_mesh, make_fixed_bin_edges, DEFAULT_BINS

# Import the comparison algorithm components
from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS

DATABASE_LOCATION = r'Normalised-objects'
TEMP_OUTPUT_DIR = 'TEMP_OUTPUT'
FEATURE_CSV = 'Feature-matrix/all_features.csv'  # Path to your main feature database


class MeshViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh_object = None
        self.current_file = None
        self.normalized_file = None
        self.database_location = DATABASE_LOCATION
        self.show_normalized = False

        self.edges = make_fixed_bin_edges(DEFAULT_BINS)
        self.searcher = None
        self.result_plotters = []

        self.init_searcher()
        self.init_ui()

    def init_searcher(self):
        """Initializes the ShapeSearcher."""
        try:
            # No weighting_method passed anymore; ShapeSearcher uses feature weighting only.
            self.searcher = ShapeSearcher(
                feature_csv_path=FEATURE_CSV,
                weights=MANUAL_WEIGHTS
            )
        except Exception as e:
            print(f"Failed to initialize searcher: {e}")
            # The UI will show an error message.

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

        # Prepare button
        self.prepare_button = QPushButton('Prepare Mesh (Normalize)')
        self.prepare_button.clicked.connect(self.prepare_mesh)
        self.prepare_button.setEnabled(False)
        control_layout.addWidget(self.prepare_button)

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
            self.status_label.setText(f"Error: Could not find `{FEATURE_CSV}`.\nSearch is disabled.")
            self.metric_dropdown.setEnabled(False)
            self.search_button.setEnabled(False)

    def add_axes(self):
        """Add coordinate axes to the main viewer."""
        axes = Axes(xtitle='x', ytitle='y', ztitle='z')
        self.plotter.add(axes)
        self.origin_axes = axes  # Store reference

    def load_categories(self):
        """Load all category folders from the database"""
        try:
            if os.path.exists(self.database_location):
                categories = [d for d in os.listdir(self.database_location)
                            if os.path.isdir(os.path.join(self.database_location, d))]
                categories.sort()
                self.category_dropdown.addItems(categories)
            else:
                self.status_label.setText(f'Database not found:\n{self.database_location}')
        except Exception as e:
            self.status_label.setText(f'Error loading categories:\n{e}')

    def load_category_objects(self, category):
        """Load all objects from the selected category"""
        self.object_list.clear()
        if not category:
            return
        try:
            category_path = os.path.join(self.database_location, category)
            objects = [f for f in os.listdir(category_path) if f.endswith('.obj')]
            objects.sort()
            self.object_list.addItems(objects)
            self.status_label.setText(f'{len(objects)} objects in {category}')
        except Exception as e:
            self.status_label.setText(f'Error loading objects:\n{e}')

    def load_mesh_from_list(self, item):
        """Load the selected mesh from the list"""
        category = self.category_dropdown.currentText()
        if not category:
            return
        file_path = os.path.join(self.database_location, category, item.text())
        self.load_mesh(file_path)

    def load_mesh(self, file_path):
        """Load a mesh file and display it without calculations."""
        if not file_path or not os.path.exists(file_path):
            self.status_label.setText(f'File not found:\n{file_path}')
            return
        try:
            self.current_file = file_path
            self.mesh_object = Mesh(file_path)
            self.normalized_file = None
            self.show_normalized = False

            # Reset UI state
            self.normalize_checkbox.setChecked(False)
            self.normalize_checkbox.setEnabled(False)
            self.prepare_button.setEnabled(True)
            if self.searcher:
                self.search_button.setEnabled(True)

            # Display the original mesh
            self.display_mesh(self.current_file)
            self.update_status_label()

        except Exception as e:
            self.status_label.setText(f'Error loading:\n{e}')
            self.prepare_button.setEnabled(False)
            self.search_button.setEnabled(False)

    def prepare_mesh(self):
        """Normalize the currently loaded mesh."""
        if not self.mesh_object:
            self.status_label.setText("No mesh loaded to prepare.")
            return
        try:
            self.status_label.setText(f'Normalizing...\n{os.path.basename(self.current_file)}')
            QApplication.processEvents()

            self.mesh_object.full_normalize()
            self.normalized_file = self.mesh_object.save()

            self.normalize_checkbox.setEnabled(True)
            self.status_label.setText(f'Normalization complete for\n{os.path.basename(self.current_file)}')

        except Exception as e:
            self.status_label.setText(f'Error during normalization:\n{e}')
            self.normalize_checkbox.setEnabled(False)

    def toggle_normalize_view(self):
        """Toggle between showing original and normalized mesh"""
        if not self.current_file:
            return
        self.show_normalized = self.normalize_checkbox.isChecked()
        try:
            # Only show normalized view if the file has been prepared
            if self.show_normalized and self.normalized_file:
                self.display_mesh(self.normalized_file)
            else:
                self.display_mesh(self.current_file)
            self.update_status_label()
        except Exception as e:
            self.status_label.setText(f'Error toggling view:\n{e}')

    def update_status_label(self):
        """Updates the status label based on the current view."""
        if not self.current_file: return
        state = 'Normalized' if self.show_normalized and self.normalized_file else 'Original'
        v_count = self.mesh_object.vertex_count() if self.mesh_object else 'N/A'
        self.status_label.setText(f'{state} [{v_count} vertices]:\n{os.path.basename(self.current_file)}')

    def find_similar_shapes(self):
        """Performs search and displays the top 5 results."""
        if not self.searcher or not self.current_file:
            self.status_label.setText("Searcher not ready or no mesh loaded.")
            return

        category = self.category_dropdown.currentText()
        object_name = os.path.basename(self.current_file)
        query_label = os.path.join(category, object_name).replace('\\', '/')
        metric = self.metric_dropdown.currentText()

        self.status_label.setText(f"Searching for similar shapes to {object_name}...")
        QApplication.processEvents()

        try:
            results = self.searcher.search(query_label=query_label, metric=metric, top_n=5)

            for i, res_label in enumerate(results):
                res_path = os.path.join(self.database_location, res_label)
                self.display_result_mesh(res_path, i)

            # Clear remaining result plotters if less than 5 results found
            for i in range(len(results), 5):
                self.result_plotters[i].clear().render()

            self.status_label.setText(f"Top 5 results for {object_name}")

        except ValueError as e:
            self.status_label.setText(f"Search Error: {e}")
        except Exception as e:
            self.status_label.setText(f"An unexpected error occurred: {e}")

    def display_result_mesh(self, file_path, index):
        """Displays a result mesh in one of the five small viewers."""
        if index >= len(self.result_plotters):
            return
        try:
            plot = self.result_plotters[index]
            plot.clear()
            mesh = load(file_path).lighting('default').linecolor('black').linewidth(0.5)
            plot.show(mesh, resetcam=True)
        except Exception as e:
            print(f"Error displaying result mesh {file_path}: {e}")

    def display_mesh(self, file_path):
        """Display mesh in the main viewer."""
        try:
            self.plotter.clear()
            self.plotter.add(self.origin_axes)
            mesh = load(file_path).lighting('default').linecolor('black').linewidth(1)
            self.plotter.show(mesh, resetcam=True)
        except Exception as e:
            self.status_label.setText(f'Error displaying mesh:\n{e}')

    def closeEvent(self, event):
        """Clean up temp folder when closing the viewer"""
        try:
            if os.path.exists(TEMP_OUTPUT_DIR):
                shutil.rmtree(TEMP_OUTPUT_DIR)
                print(f'Cleaned up {TEMP_OUTPUT_DIR} folder')
        except Exception as e:
            print(f'Error cleaning up {TEMP_OUTPUT_DIR}: {e}')
        finally:
            event.accept()


def main():
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
# python
import sys
import os
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QLabel, QComboBox, QPushButton, QListWidget)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, load

# Import from your existing project files
from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS
from viewerRob import DATABASE_LOCATION, FEATURE_CSV

# Define the location of the normalized objects used for displaying results
NORMALIZED_DB_LOCATION = r'Normalised-objects'


class ResultsWindow(QMainWindow):
    """A window to display query objects and their search results in a grid."""

    def __init__(self, query_data, metric, parent=None):
        super().__init__(parent)
        self.query_data = query_data
        self.metric = metric or "(none)"
        self.plotters = []
        self.labels = []
        self.init_ui()
        self.load_all_meshes()

    def init_ui(self):
        self.setWindowTitle('Shape Search Results')
        self.setGeometry(50, 50, 1800, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Vertical layout: top label for metric, then grid of viewers
        vlayout = QVBoxLayout(central_widget)
        metric_label = QLabel(f"Distance metric: {self.metric}")
        metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metric_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        vlayout.addWidget(metric_label)

        grid_layout = QGridLayout()
        vlayout.addLayout(grid_layout)

        # Create a 4x6 grid of viewers with labels
        for row in range(4):
            row_plotters = []
            row_labels = []
            for col in range(6):
                container = QWidget()
                layout = QVBoxLayout(container)
                layout.setContentsMargins(2, 2, 2, 2)
                layout.setSpacing(2)

                vtk_widget = QVTKRenderWindowInteractor(self)
                plotter = Plotter(qt_widget=vtk_widget, N=1, bg='lightgrey', axes=0)
                label = QLabel("-")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                layout.addWidget(vtk_widget)
                layout.addWidget(label)

                grid_layout.addWidget(container, row, col)
                row_plotters.append(plotter)
                row_labels.append(label)
            self.plotters.append(row_plotters)
            self.labels.append(row_labels)

    def load_mesh(self, file_path, plotter, label_widget, category_text):
        """Loads a single mesh into a given plotter and updates its label."""
        label_widget.setText(category_text)
        try:
            if file_path and os.path.exists(file_path):
                mesh = load(file_path).lighting('default').linecolor('black').linewidth(0.5)
                plotter.show(mesh, resetcam=True)
            else:
                plotter.clear().render()  # Clear if no path
                label_widget.setText("-")  # Clear label if no mesh
        except Exception as e:
            print(f"Error loading mesh {file_path}: {e}")
            plotter.clear().render()
            label_widget.setText("Error")

    def load_all_meshes(self):
        """Populates the grid with query and result meshes."""
        for row_idx in range(4):
            if row_idx < len(self.query_data):
                data = self.query_data[row_idx]
                # Load query object (column 0)
                query_path = data['query_path']
                query_category = data.get('query_category', '-')
                self.plotters[row_idx][0].background('lightblue')  # Highlight query
                self.load_mesh(query_path, self.plotters[row_idx][0], self.labels[row_idx][0], query_category)

                # Load result objects (columns 1-5)
                for col_idx, result_label in enumerate(data['results']):
                    if col_idx < 5:  # Ensure we only load up to 5 results
                        result_path = os.path.join(NORMALIZED_DB_LOCATION, result_label)
                        result_category = result_label.split('/')[0] if '/' in result_label else '-'
                        self.load_mesh(result_path, self.plotters[row_idx][col_idx + 1],
                                       self.labels[row_idx][col_idx + 1], result_category)
            else:
                # Clear unused rows
                for col_idx in range(6):
                    self.plotters[row_idx][col_idx].clear().render()
                    self.labels[row_idx][col_idx].setText("-")


class SelectionWindow(QMainWindow):
    """The main window for selecting objects and search parameters."""

    def __init__(self):
        super().__init__()
        self.searcher = None
        self.results_window = None
        self.object_selectors = []

        self.init_ui()
        self.load_categories()
        self.init_searcher()

    def init_ui(self):
        self.setWindowTitle('Shape Search Setup')
        self.setGeometry(100, 100, 500, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Object Selection Area
        main_layout.addWidget(QLabel("<h3>Select up to 4 Query Objects</h3>"))
        for i in range(4):
            layout = QHBoxLayout()
            cat_combo = QComboBox()
            cat_combo.setPlaceholderText(f"Category {i + 1}")
            obj_list = QListWidget()
            obj_list.setMaximumHeight(100)

            # Correctly capture the obj_list for the lambda
            cat_combo.currentTextChanged.connect(
                lambda text, lst=obj_list: self.populate_objects(text, lst)
            )

            layout.addWidget(cat_combo)
            layout.addWidget(obj_list)
            main_layout.addLayout(layout)
            self.object_selectors.append({'category': cat_combo, 'object': obj_list})

        # Search Parameters
        main_layout.addWidget(QLabel("<h3>Search Parameters</h3>"))

        self.weight_method_combo = QComboBox()
        self.weight_method_combo.addItems(['feature', 'neutral'])
        main_layout.addWidget(QLabel("Weighting Method:"))
        main_layout.addWidget(self.weight_method_combo)

        self.metric_combo = QComboBox()
        main_layout.addWidget(QLabel("Distance Metric:"))
        main_layout.addWidget(self.metric_combo)

        # Search Button
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.perform_search)
        main_layout.addWidget(self.search_button)

        self.status_label = QLabel("Initialize searcher...")
        main_layout.addWidget(self.status_label)

    def init_searcher(self):
        """Initializes the ShapeSearcher and updates UI accordingly."""
        try:
            # We initialize with default 'feature' and change it on-demand
            self.searcher = ShapeSearcher(
                feature_csv_path=FEATURE_CSV,
                weights=MANUAL_WEIGHTS,
                weighting_method='feature'
            )
            self.metric_combo.addItems(self.searcher.metrics)
            self.status_label.setText("Ready. Select objects and a metric.")
            self.search_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.search_button.setEnabled(False)

    def load_categories(self):
        """Loads categories from the database into all category dropdowns."""
        try:
            categories = [d for d in os.listdir(DATABASE_LOCATION) if os.path.isdir(os.path.join(DATABASE_LOCATION, d))]
            categories.sort()
            for selector in self.object_selectors:
                selector['category'].addItems(categories)
                selector['category'].setCurrentIndex(-1)  # Set to placeholder
        except Exception as e:
            self.status_label.setText(f"Could not load categories: {e}")

    def populate_objects(self, category, object_list_widget):
        """Populates the object list for a given category."""
        object_list_widget.clear()
        if not category:
            return
        try:
            category_path = os.path.join(DATABASE_LOCATION, category)
            objects = [f for f in os.listdir(category_path) if f.endswith('.obj')]
            objects.sort()
            object_list_widget.addItems(objects)
        except Exception as e:
            print(f"Error loading objects for {category}: {e}")

    def perform_search(self):
        """Gathers selections, runs the search, and opens the results window."""
        self.searcher.weighting_method = self.weight_method_combo.currentText()
        metric = self.metric_combo.currentText()

        query_data = []
        for selector in self.object_selectors:
            category = selector['category'].currentText()
            selected_items = selector['object'].selectedItems()
            if category and selected_items:
                obj_name = selected_items[0].text()

                # Construct the label used in the feature matrix
                base_name = os.path.splitext(obj_name)[0]
                normalized_name = f"{base_name}_rm.obj"
                query_label = os.path.join(category, normalized_name).replace('\\', '/')

                # Get the path to the original object for display
                original_path = os.path.join(DATABASE_LOCATION, category, obj_name)

                try:
                    results = self.searcher.search(query_label=query_label, metric=metric, top_n=5)
                    query_data.append({
                        'query_path': original_path,
                        'query_category': category,
                        'results': results
                    })
                except ValueError as e:
                    self.status_label.setText(f"Search failed for {obj_name}: {e}")
                    return  # Stop search if one object fails

        if not query_data:
            self.status_label.setText("No valid objects selected for search.")
            return

        # Launch the results window and pass the chosen metric
        self.results_window = ResultsWindow(query_data, metric)
        self.results_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = SelectionWindow()
    main_win.show()
    sys.exit(app.exec())
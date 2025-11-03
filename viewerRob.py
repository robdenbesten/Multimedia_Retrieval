import sys
import os
import shutil
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget, QCheckBox, QGridLayout,
                             QScrollArea, QTextEdit)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, load, Line, Axes

# Import the Mesh class and feature extraction
from normalise import Mesh
from feature_extraction import extract_features_for_single_mesh, make_fixed_bin_edges, DEFAULT_BINS

# Import the comparison algorithm components
from Comparison_algorithm import ShapeSearcher, MANUAL_WEIGHTS

DATABASE_LOCATION = r'ShapeDatabase_INFOMR-master\Original Database'
TEMP_OUTPUT_DIR = 'TEMP_OUTPUT'
FEATURE_CSV = 'Feature-matrix/all_features_modified.csv'  # Path to your main feature database


class MeshViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh_object = None
        self.current_file = None
        self.normalized_file = None
        self.database_location = DATABASE_LOCATION
        self.show_normalized = False  # Track toggle state
        
        # Initialize feature extraction bins
        self.edges = make_fixed_bin_edges(DEFAULT_BINS)
        self.searcher = None
        self.result_plotters = []
        self.result_labels = []  # Store result labels for dissimilarity display
        self.current_features = None  # Store current mesh features

        self.init_searcher()
        self.init_ui()

    def init_searcher(self):
        """Initializes the ShapeSearcher."""
        try:
            self.searcher = ShapeSearcher(
                feature_csv_path=FEATURE_CSV,
                weights=MANUAL_WEIGHTS,
                weighting_method = 'feature'  # or 'neutral'
            )
        except Exception as e:
            print(f"Failed to initialize searcher: {e}")
            # The UI will show an error message.
        
    def init_ui(self):
        self.setWindowTitle('Mesh Viewer and Search')
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)
        
        # Category dropdown
        category_label = QLabel('Select Category:')
        control_layout.addWidget(category_label)
        
        self.category_dropdown = QComboBox()
        self.category_dropdown.currentTextChanged.connect(self.load_category_objects)
        control_layout.addWidget(self.category_dropdown)
        
        # Object list
        objects_label = QLabel('Objects:')
        control_layout.addWidget(objects_label)
        
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

        # Show axes checkbox
        self.show_axes_checkbox = QCheckBox('Show Axes')
        self.show_axes_checkbox.setChecked(True)  # Axes visible by default
        self.show_axes_checkbox.stateChanged.connect(self.toggle_axes_visibility)
        control_layout.addWidget(self.show_axes_checkbox)

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

        # Middle panel for main viewer and mesh info
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)

        # Mesh info label (vertices/faces) above main viewer
        self.mesh_info_label = QLabel('Mesh Info: Load a mesh')
        self.mesh_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mesh_info_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        middle_layout.addWidget(self.mesh_info_label)

        # Main VTK Widget
        self.vtk_widget = QVTKRenderWindowInteractor(middle_panel)
        self.plotter = Plotter(qt_widget=self.vtk_widget, N=1, bg='white')
        middle_layout.addWidget(self.vtk_widget, stretch=3)

        # Results viewers with dissimilarity labels
        results_widget = QWidget()
        results_layout = QGridLayout(results_widget)
        results_layout.setSpacing(5)

        self.result_labels = []
        for i in range(5):
            # Create container for each result
            result_container = QWidget()
            result_container_layout = QVBoxLayout(result_container)
            result_container_layout.setContentsMargins(0, 0, 0, 0)
            result_container_layout.setSpacing(2)

            # Dissimilarity label
            dissim_label = QLabel(f'Result {i+1}')
            dissim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dissim_label.setStyleSheet("font-size: 10px; font-weight: bold;")
            result_container_layout.addWidget(dissim_label)
            self.result_labels.append(dissim_label)

            # VTK widget for result mesh
            vtk_res_widget = QVTKRenderWindowInteractor(result_container)
            plot = Plotter(qt_widget=vtk_res_widget, N=1, bg='lightgrey')
            self.result_plotters.append(plot)
            result_container_layout.addWidget(vtk_res_widget)

            results_layout.addWidget(result_container, 0, i)

        middle_layout.addWidget(results_widget, stretch=1)

        # Right panel for feature visualization
        feature_panel = QWidget()
        feature_layout = QVBoxLayout(feature_panel)
        feature_panel.setMaximumWidth(400)

        feature_label = QLabel('Feature Visualization')
        feature_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feature_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        feature_layout.addWidget(feature_label)

        # Scroll area for features
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Scalar features text display
        self.scalar_features_text = QTextEdit()
        self.scalar_features_text.setReadOnly(True)
        self.scalar_features_text.setMaximumHeight(200)
        self.scalar_features_text.setPlainText("Scalar Features:\nLoad a mesh to see features")
        scroll_layout.addWidget(QLabel("Scalar Features:"))
        scroll_layout.addWidget(self.scalar_features_text)

        # Matplotlib canvas for histograms
        self.feature_figure = Figure(figsize=(4, 8))
        self.feature_canvas = FigureCanvas(self.feature_figure)
        scroll_layout.addWidget(QLabel("Histogram Features:"))
        scroll_layout.addWidget(self.feature_canvas)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        feature_layout.addWidget(scroll)

        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(middle_panel, stretch=2)
        main_layout.addWidget(feature_panel)

        # Add coordinate axes
        self.add_axes()
        
        # Load categories
        self.load_categories()

        if not self.searcher:
            self.status_label.setText(f"Error: Could not find '{FEATURE_CSV}'.\nSearch is disabled.")
            self.metric_dropdown.setEnabled(False)
            self.search_button.setEnabled(False)
    
    def add_axes(self):
        """Add coordinate axes to the viewer with labels and real values"""
        # Create axes with vedo's Axes class for proper labeling
        from vedo import Axes

        # Create axes object with real-world scale
        # The axes will auto-adjust to the mesh bounds when a mesh is loaded
        # Vedo Axes uses xrange, yrange, zrange tuples and xtitle, ytitle, ztitle for labels
        self.axes_object = Axes(
            xrange=(-0.5, 0.5),
            yrange=(-0.5, 0.5),
            zrange=(-0.5, 0.5),
            xtitle='X',
            ytitle='Y',
            ztitle='Z',
            text_scale=1.2,
            c='black',
        )

        # Store reference
        self.origin_axes = self.axes_object

        # Add to plotter
        self.plotter.add(self.axes_object)

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
            if os.path.exists(category_path):
                objects = [f for f in os.listdir(category_path) 
                          if f.endswith('.obj')]
                objects.sort()
                self.object_list.addItems(objects)
                self.status_label.setText(f'{len(objects)} objects in {category}')
            else:
                self.status_label.setText(f'Category folder not found')
        except Exception as e:
            self.status_label.setText(f'Error loading objects:\n{e}')
    
    def load_mesh_from_list(self, item):
        """Load the selected mesh from the list"""
        category = self.category_dropdown.currentText()
        if not category:
            return
            
        object_name = item.text()
        file_path = os.path.join(self.database_location, category, object_name)
        self.load_mesh(file_path)
        
    def load_mesh(self, file_path):
        """Load a mesh file, normalize it, and display progressively during each step"""
        if file_path:
            try:
                # Create a new Mesh object
                self.mesh_object = Mesh(file_path)
                self.current_file = file_path
                self.normalized_file = None
                
                # Disable checkbox and search during processing
                self.normalize_checkbox.setEnabled(False)
                self.search_button.setEnabled(False)
                
                # Step 1: Display original file immediately if that's what we want to show
                self.status_label.setText(f'Loaded original file:\n{os.path.basename(file_path)}')
                if not self.show_normalized:
                    self.display_mesh(self.current_file)
                    self.update_status_label()
                QApplication.processEvents()
                
                # Step 2: Perform normalization
                self.status_label.setText(f'Normalizing and remeshing:\n{os.path.basename(file_path)}')
                QApplication.processEvents()
                self.mesh_object.full_normalize()
                self.normalized_file = self.mesh_object.save()
                
                # Step 3: Display normalized file immediately if that's what we want to show
                if self.show_normalized:
                    self.display_mesh(self.normalized_file)
                    self.update_status_label()
                QApplication.processEvents()
                
                # Step 4: Extract features (this doesn't affect display)
                #self.status_label.setText(f'Extracting features:\n{os.path.basename(file_path)}')
                #QApplication.processEvents()
                #self.extract_features()
                
                # Step 5: Extract and display features
                self.extract_and_display_features()

                # Step 6: Enable controls and finalize
                self.normalize_checkbox.setEnabled(True)
                if self.searcher:
                    self.search_button.setEnabled(True)
                
                # Final status update
                self.update_status_label()
                
            except Exception as e:
                self.status_label.setText(f'Error loading:\n{e}')
                self.normalize_checkbox.setEnabled(False)
                self.search_button.setEnabled(False)
            
    def toggle_normalize_view(self):
        """Toggle between showing original and normalized mesh"""
        if not self.current_file or not self.normalized_file:
            return
        
        self.show_normalized = self.normalize_checkbox.isChecked()
        
        try:
            self.display_mesh(self.normalized_file if self.show_normalized else self.current_file)
            self.update_status_label()
        except Exception as e:
            self.status_label.setText(f'Error toggling view:\n{e}')

    def update_status_label(self):
        """Updates the status label based on the current view."""
        if not self.current_file: return
        
        if self.show_normalized:
            state = 'Normalized'
            # For normalized view, use the mesh_object vertex and face count
            try:
                v_count = self.mesh_object.vertex_count() if self.mesh_object else 'N/A'
                f_count = self.mesh_object.face_count() if self.mesh_object else 'N/A'
            except Exception as e:
                print(f"Error getting normalized mesh counts: {e}")
                v_count = 'N/A'
                f_count = 'N/A'
        else:
            state = 'Original'
            # For original view, load the original file to get its vertex and face count
            try:
                import trimesh
                original_mesh = trimesh.load(self.current_file)
                v_count = len(original_mesh.vertices)
                f_count = len(original_mesh.faces)
            except Exception as e:
                print(f"Error getting original mesh counts: {e}")
                v_count = 'N/A'
                f_count = 'N/A'

        self.status_label.setText(f'{state} ({v_count} vertices):\n{os.path.basename(self.current_file)}')
        self.mesh_info_label.setText(f'Vertices: {v_count} | Faces: {f_count}')

    def extract_and_display_features(self):
        """Extract features from the current mesh and display them."""
        if not self.normalized_file or not os.path.exists(self.normalized_file):
            return

        try:
            # Get category and object name for feature lookup
            category = self.category_dropdown.currentText()
            original_object_name = os.path.basename(self.current_file)
            base_name = os.path.splitext(original_object_name)[0]
            normalized_object_name = f"{base_name}_rm.obj"
            query_label = os.path.join(category, normalized_object_name).replace('\\', '/')

            # Get features from the searcher's database
            if self.searcher and query_label in self.searcher.features_df.index:
                features = self.searcher.features_df.loc[query_label]
                self.current_features = features

                # Display scalar features
                scalar_names = ['Surface area', 'Sphericity', 'Rectangularity',
                               'Diameter', 'Convexity', 'Eccentricity']
                scalar_text = "Scalar Features:\n" + "="*30 + "\n"
                for name in scalar_names:
                    if name in features.index:
                        scalar_text += f"{name:20s}: {features[name]:.4f}\n"
                self.scalar_features_text.setPlainText(scalar_text)

                # Display histogram features
                self.display_histograms(features)
            else:
                self.scalar_features_text.setPlainText(f"Features not found for:\n{query_label}")

        except Exception as e:
            print(f"Error extracting features: {e}")
            self.scalar_features_text.setPlainText(f"Error extracting features:\n{e}")

    def toggle_axes_visibility(self):
        """Toggle the visibility of coordinate axes"""
        if self.origin_axes:
            if self.show_axes_checkbox.isChecked():
                self.plotter.add(self.origin_axes)
            else:
                self.plotter.remove(self.origin_axes)
            self.plotter.render()

    def display_histograms(self, features):
        """Display histogram features as bar charts."""
        try:
            self.feature_figure.clear()

            hist_descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
            n_bins = 20

            for idx, desc in enumerate(hist_descriptors):
                ax = self.feature_figure.add_subplot(5, 1, idx + 1)

                # Extract histogram bins for this descriptor
                bin_cols = [f'{desc}_bin_{i}' for i in range(n_bins)]
                hist_values = [features[col] if col in features.index else 0 for col in bin_cols]

                # Plot histogram
                ax.bar(range(n_bins), hist_values, color='steelblue', edgecolor='black', linewidth=0.5)
                ax.set_title(desc, fontsize=10, fontweight='bold')
                ax.set_xlim(-0.5, n_bins - 0.5)
                ax.set_ylim(0, max(hist_values) * 1.1 if max(hist_values) > 0 else 1)
                ax.set_ylabel('Frequency', fontsize=8)
                if idx == len(hist_descriptors) - 1:
                    ax.set_xlabel('Bin', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(axis='y', alpha=0.3)

            self.feature_figure.tight_layout()
            self.feature_canvas.draw()

        except Exception as e:
            print(f"Error displaying histograms: {e}")

    def find_similar_shapes(self):
        """Performs search and displays the top 5 results with dissimilarity values."""
        if not self.searcher or not self.current_file:
            self.status_label.setText("Searcher not ready or no mesh loaded.")
            return

        category = self.category_dropdown.currentText()
        original_object_name = os.path.basename(self.current_file)
        
        # Convert original filename to normalized filename format
        # Original: "m1337.obj" -> Normalized: "m1337_rm.obj"
        base_name = os.path.splitext(original_object_name)[0]
        normalized_object_name = f"{base_name}_rm.obj"
        query_label = os.path.join(category, normalized_object_name).replace('\\', '/')
        
        metric = self.metric_dropdown.currentText()

        self.status_label.setText(f"Searching for similar shapes to {original_object_name}...")
        QApplication.processEvents()

        try:
            # Get search results with distances
            results_with_distances = self.searcher.search_with_distances(
                query_label=query_label, metric=metric, top_n=5
            )

            # Use normalized database for displaying results since feature matrix is based on normalized objects
            normalized_db_location = r'Normalised-objects'
            
            for i, (res_label, distance) in enumerate(results_with_distances):
                res_path = os.path.join(normalized_db_location, res_label)
                self.display_result_mesh(res_path, i, distance)

            # Clear remaining result plotters if less than 5 results found
            for i in range(len(results_with_distances), 5):
                self.result_plotters[i].clear().render()
                self.result_labels[i].setText(f'Result {i+1}')

            self.status_label.setText(f"Top 5 results for {original_object_name}")

        except ValueError as e:
            self.status_label.setText(f"Search Error: {e}\nTried to find: {query_label}")
            print(f"Available labels sample: {self.searcher.get_available_labels()[:10]}")
            # Let's try to find a similar label in the same category
            available_labels = self.searcher.get_available_labels()
            category_labels = [label for label in available_labels if label.startswith(category + "/")]
            print(f"Available labels in {category}: {category_labels[:5]}")
        except Exception as e:
            self.status_label.setText(f"An unexpected error occurred: {e}")

    def display_result_mesh(self, file_path, index, dissimilarity=None):
        """Displays a result mesh in one of the five small viewers with dissimilarity value."""
        if index >= len(self.result_plotters):
            return
        try:
            plot = self.result_plotters[index]
            plot.clear()
            mesh = load(file_path).lighting('default').linecolor('black').linewidth(0.5)
            plot.show(mesh, resetcam=True)

            # Update label with dissimilarity value
            if dissimilarity is not None:
                obj_name = os.path.basename(file_path)
                self.result_labels[index].setText(f'{obj_name}\nDissimilarity: {dissimilarity:.4f}')
            else:
                self.result_labels[index].setText(f'Result {index+1}')
        except Exception as e:
            print(f"Error displaying result mesh {file_path}: {e}")

    """def extract_features(self):
        #Extract features from the normalized mesh and save to CSV
        if not self.normalized_file or not os.path.exists(self.normalized_file):
            print("No normalized file to extract features from")
            return
        
        try:
            # Get category and object name
            category = self.category_dropdown.currentText()
            obj_name = os.path.basename(self.current_file)
            
            # Create relative path for deterministic RNG (format: "category/filename.obj")
            rel_path = f"{category}/{obj_name}"
            
            # Output CSV path in remeshed_output folder
            feature_filename = os.path.splitext(os.path.basename(self.normalized_file))[0] + '_features.csv'
            out_path = os.path.join(TEMP_OUTPUT_DIR, feature_filename)
            
            # Settings matching all_features.csv generation
            n_samples = 250000
            surface_points = 5000
            
            # Call the feature extraction function
            obj_path_result, success, result = extract_features_for_single_mesh(
                obj_path=self.normalized_file,
                rel_path=rel_path,
                out_path=out_path,
                edges=self.edges,
                n_samples=n_samples,
                surface_points=surface_points
            )
            
            if success and isinstance(result, list):
                # Create CSV header matching all_features.csv
                metric_keys = ["Mesh volume", "Surface area", "Diameter", "Compactness",
                              "Rectangularity", "Convexity", "Eccentricity", "Sphericity"]
                header = ["Object", "Category"] + metric_keys + ["extents_0", "extents_1", "extents_2"]
                
                # Add histogram bin headers (5 descriptors × 20 bins each)
                hist_order = ['D1', 'D2', 'A3', 'D3', 'D4']
                for k in hist_order:
                    n_bins = 20
                    header += [f'{k}_bin_{i}' for i in range(n_bins)]
                
                # Write CSV file
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerow(result)
                
                print(f"Features saved to: {out_path}")
            else:
                print(f"Feature extraction failed: {result}")
                
        except Exception as e:
            print(f"Error in extract_features: {e}")
            import traceback
            traceback.print_exc() """

    def display_mesh(self, file_path):
        """Display mesh in the main viewer."""
        try:
            self.plotter.clear()
            
            # Re-add origin axes if they should be shown
            if self.show_axes_checkbox.isChecked() and self.origin_axes:
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
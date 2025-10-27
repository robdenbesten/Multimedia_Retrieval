import sys
import os
import shutil
import csv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget, QCheckBox)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, load, Line

# Import the Mesh class and feature extraction
from normalise import Mesh
from feature_extraction import extract_features_for_single_mesh, make_fixed_bin_edges, DEFAULT_BINS

DATABASE_LOCATION = r'ShapeDatabase_INFOMR-master\Original Database'
REMESHED_OUTPUT_DIR = 'remeshed_output'


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
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Mesh Normalizer Viewer')
        self.setGeometry(100, 100, 1200, 600)
        
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
        
        # VTK Widget for 3D rendering with vedo
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        self.plotter = Plotter(qt_widget=self.vtk_widget)
        
        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.vtk_widget)
        
        # Add coordinate axes
        self.add_axes()
        
        # Load categories
        self.load_categories()
    
    def add_axes(self):
        """Add coordinate axes to the viewer (X=red, Y=green, Z=blue)"""
        x_axis = Line([0, 0, 0], [0.5, 0, 0]).c('red').lw(1)
        y_axis = Line([0, 0, 0], [0, 0.5, 0]).c('green').lw(1)
        z_axis = Line([0, 0, 0], [0, 0, 0.5]).c('blue').lw(1)
        
        # Store axes references
        self.origin_axes = [x_axis, y_axis, z_axis]
        
        # Add to plotter
        for axis in self.origin_axes:
            self.plotter.add(axis)
        
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
        """Load a mesh file, normalize it, and display based on toggle state"""
        if file_path:
            try:
                # Create a new Mesh object
                self.mesh_object = Mesh(file_path)
                self.current_file = file_path
                self.normalized_file = None
                
                # Disable checkbox during normalization
                self.normalize_checkbox.setEnabled(False)
                self.status_label.setText(f'Loading and normalizing:\n{os.path.basename(file_path)}')
                QApplication.processEvents()
                
                # Always normalize in the background
                self.mesh_object.full_normalize()
                self.normalized_file = self.mesh_object.save()
                
                # Extract features from normalized mesh
                self.status_label.setText(f'Extracting features:\n{os.path.basename(file_path)}')
                QApplication.processEvents()
                self.extract_features()
                
                # Enable checkbox
                self.normalize_checkbox.setEnabled(True)
                
                # Display based on toggle state
                if self.show_normalized:
                    self.display_mesh(self.normalized_file)
                    self.status_label.setText(f'Normalized [{self.mesh_object.vertex_count()} vertices]:\n{os.path.basename(file_path)}')
                else:
                    self.display_mesh(self.current_file)
                    self.status_label.setText(f'Original:\n{os.path.basename(file_path)}')
                
            except Exception as e:
                self.status_label.setText(f'Error loading:\n{e}')
                self.normalize_checkbox.setEnabled(False)
            
    def toggle_normalize_view(self):
        """Toggle between showing original and normalized mesh"""
        if not self.current_file or not self.normalized_file:
            return
        
        self.show_normalized = self.normalize_checkbox.isChecked()
        
        try:
            if self.show_normalized:
                # Show normalized version
                self.display_mesh(self.normalized_file)
                self.status_label.setText(f'Normalized [{self.mesh_object.vertex_count()} vertices]:\n{os.path.basename(self.current_file)}')
            else:
                # Show original version
                self.display_mesh(self.current_file)
                self.status_label.setText(f'Original:\n{os.path.basename(self.current_file)}')
        except Exception as e:
            self.status_label.setText(f'Error toggling view:\n{e}')

    def extract_features(self):
        """Extract features from the normalized mesh and save to CSV"""
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
            out_path = os.path.join(REMESHED_OUTPUT_DIR, feature_filename)
            
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
            traceback.print_exc()

    def display_mesh(self, file_path):
        """Display mesh using vedo for better lighting and style"""
        try:
            # Clear plotter but keep axes
            self.plotter.clear()
            
            # Re-add origin axes
            for axis in self.origin_axes:
                self.plotter.add(axis)
            
            # Load mesh with vedo
            mesh = load(file_path)
            mesh.lighting('default').linecolor('black').linewidth(1)
            self.plotter.show(mesh, resetcam=True)
            
        except Exception as e:
            self.status_label.setText(f'Error displaying mesh:\n{e}')
    
    def closeEvent(self, event):
        """Clean up remeshed_output folder when closing the viewer"""
        try:
            if os.path.exists(REMESHED_OUTPUT_DIR):
                # Remove all files in the directory
                for filename in os.listdir(REMESHED_OUTPUT_DIR):
                    file_path = os.path.join(REMESHED_OUTPUT_DIR, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
                print(f'Cleaned up {REMESHED_OUTPUT_DIR} folder')
        except Exception as e:
            print(f'Error cleaning up {REMESHED_OUTPUT_DIR}: {e}')
        finally:
            # Accept the close event
            event.accept()


def main():
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, load, Line

# Import the Mesh class
from normalise import Mesh

DATABASE_LOCATION = r'ShapeDatabase_INFOMR-master\Original Database'


class MeshViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh_object = None
        self.current_file = None
        self.normalized_file = None
        self.database_location = DATABASE_LOCATION
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
        
        # Normalize button
        self.normalize_btn = QPushButton('Normalize')
        self.normalize_btn.clicked.connect(self.normalize_mesh)
        self.normalize_btn.setEnabled(False)
        control_layout.addWidget(self.normalize_btn)
        
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
        """Load a mesh file and display it"""
        if file_path:
            try:
                # Create a new Mesh object
                self.mesh_object = Mesh(file_path)
                self.current_file = file_path
                self.normalized_file = None
                
                # Show the mesh
                self.display_mesh(self.current_file)
                self.status_label.setText(f'Showing:\n{os.path.basename(file_path)}')
                self.normalize_btn.setEnabled(True)
                
            except Exception as e:
                self.status_label.setText(f'Error loading:\n{e}')
            
    def normalize_mesh(self):
        if self.current_file and self.mesh_object:
            try:
                self.status_label.setText('Normalizing...')
                QApplication.processEvents()
                
                # Normalize the mesh using the Mesh class
                self.mesh_object.full_normalize()

                # Save the normalized mesh
                self.normalized_file = self.mesh_object.save()
                
                if not self.normalized_file:
                    self.status_label.setText('Error saving mesh')
                    return
                
                # Force the viewer to update
                QApplication.processEvents()
                
                # Show the normalized mesh
                self.display_mesh(self.normalized_file)
                self.status_label.setText(f'Normalized:\n{os.path.basename(self.current_file)} [{self.mesh_object.vertex_count()} vertices]')
                
            except Exception as e:
                print(f"Normalization error: {e}")
                import traceback
                traceback.print_exc()
                self.status_label.setText(f'Error:\n{str(e)[:100]}')
            
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


def main():
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

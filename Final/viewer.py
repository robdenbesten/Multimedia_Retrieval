import sys
import os
import pymeshlab as ml
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

# Import the normalization function
from normalise import full_normalise

DATABASE_LOCATION = r'ShapeDatabase_INFOMR-master\Original Database'


class MeshViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh_set = None
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
        
        # VTK Widget for 3D rendering
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        
        # Setup renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.9, 0.9, 0.9)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.vtk_widget)
        
        # Initialize interactor
        self.interactor.Initialize()
        self.interactor.Start()
        
        # Add coordinate axes
        self.add_axes()
        
        # Load categories
        self.load_categories()
    
    def add_axes(self):
        """Add coordinate axes to the viewer (X=red, Y=green, Z=blue)"""
        # X-axis (red)
        x_line = vtk.vtkLineSource()
        x_line.SetPoint1(0, 0, 0)
        x_line.SetPoint2(1, 0, 0)
        x_mapper = vtk.vtkPolyDataMapper()
        x_mapper.SetInputConnection(x_line.GetOutputPort())
        x_actor = vtk.vtkActor()
        x_actor.SetMapper(x_mapper)
        x_actor.GetProperty().SetColor(1, 0, 0)  # Red
        x_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(x_actor)
        
        # Y-axis (green)
        y_line = vtk.vtkLineSource()
        y_line.SetPoint1(0, 0, 0)
        y_line.SetPoint2(0, 1, 0)
        y_mapper = vtk.vtkPolyDataMapper()
        y_mapper.SetInputConnection(y_line.GetOutputPort())
        y_actor = vtk.vtkActor()
        y_actor.SetMapper(y_mapper)
        y_actor.GetProperty().SetColor(0, 1, 0)  # Green
        y_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(y_actor)
        
        # Z-axis (blue)
        z_line = vtk.vtkLineSource()
        z_line.SetPoint1(0, 0, 0)
        z_line.SetPoint2(0, 0, 1)
        z_mapper = vtk.vtkPolyDataMapper()
        z_mapper.SetInputConnection(z_line.GetOutputPort())
        z_actor = vtk.vtkActor()
        z_actor.SetMapper(z_mapper)
        z_actor.GetProperty().SetColor(0, 0, 1)  # Blue
        z_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(z_actor)
        
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
                self.mesh_set = ml.MeshSet()
                self.mesh_set.load_new_mesh(file_path)
                self.current_file = file_path
                self.normalized_file = None
                
                # Show the mesh
                self.display_mesh(self.current_file)
                self.status_label.setText(f'Showing:\n{os.path.basename(file_path)}')
                self.normalize_btn.setEnabled(True)
                
            except Exception as e:
                self.status_label.setText(f'Error loading:\n{e}')
            
    def normalize_mesh(self):
        if self.current_file:
            try:
                self.status_label.setText('Normalizing...')
                QApplication.processEvents()
                
                print(f"Starting normalization of: {self.current_file}")
                
                # Import functions from normalise
                from normalise import full_normalise, save
                
                # First remesh the file
                normalized_ms = full_normalise(self.current_file)
                
                if normalized_ms is None:
                    self.status_label.setText('Error: Empty mesh')
                    return

                print(f"After normalization - vertices: {normalized_ms.current_mesh().vertex_number()}")

                # Save using the save function from normalise.py
                self.normalized_file = save(normalized_ms)
                
                if not self.normalized_file:
                    self.status_label.setText('Error saving mesh')
                    return
                
                # Force the viewer to update
                QApplication.processEvents()
                
                # Show the normalized mesh
                self.display_mesh(self.normalized_file)
                self.status_label.setText(f'Normalized:\n{os.path.basename(self.current_file)}')
                
            except Exception as e:
                print(f"Normalization error: {e}")
                import traceback
                traceback.print_exc()
                self.status_label.setText(f'Error:\n{str(e)[:100]}')
            
    def display_mesh(self, file_path):
        """Display mesh in VTK viewer"""
        try:
            # Clear previous actors (except axes)
            actors_to_remove = []
            for actor in self.renderer.GetActors():
                # Keep the axis lines (they have line width 3)
                if actor.GetProperty().GetLineWidth() != 3:
                    actors_to_remove.append(actor)
            
            for actor in actors_to_remove:
                self.renderer.RemoveActor(actor)
            
            # Load OBJ file
            reader = vtk.vtkOBJReader()
            reader.SetFileName(file_path)
            reader.Update()
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            # Create actor with edges visible
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.8, 0.8, 0.9)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
            actor.GetProperty().SetLineWidth(1)
            
            # Add actor to renderer
            self.renderer.AddActor(actor)
            
            # Reset camera to fit the mesh
            self.renderer.ResetCamera()
            
            # Render
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            self.status_label.setText(f'Error displaying mesh:\n{e}')


def main():
    app = QApplication(sys.argv)
    viewer = MeshViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

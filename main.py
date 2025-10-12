"""
Compact 3D Shape Browser and Processing GUI v2
"""
import sys
import os
import math
import threading
import shutil
from typing import Tuple, Optional
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox, QPushButton
from vedo import Plotter, load, Box, Line
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import trimesh
import pymeshlab as ml

# Constants
TARGET_VERTICES = 5000
SHAPEDATA_PARENT = os.path.abspath('ShapeDatabase_INFOMR-master/Original Database')
TEMP_REMESH_DIR = os.path.abspath('temp_remesh')


def parse_obj_info(filepath: str) -> Tuple[int, int, str, str]:
    """Parse OBJ file and extract basic information."""
    vertices, faces = [], []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) == 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append(parts[1:])
    
    # Determine face types
    face_types = set()
    for face in faces:
        count = len(face)
        if count == 3: face_types.add("triangles")
        elif count == 4: face_types.add("quads")
        else: face_types.add("other")
    
    face_type = " and ".join(sorted(face_types)) if face_types else "unknown"
    
    # Calculate bounding box
    bbox = "N/A"
    if vertices:
        xs, ys, zs = zip(*vertices)
        bbox = f"X:[{min(xs):.2f},{max(xs):.2f}] Y:[{min(ys):.2f},{max(ys):.2f}] Z:[{min(zs):.2f},{max(zs):.2f}]"
    
    return len(vertices), len(faces), face_type, bbox


def remesh_to_target_vertices(input_path: str, output_path: str) -> bool:
    """Remesh a mesh to target vertex count with robust error handling."""
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return False
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh_set = ml.MeshSet()
        
        # Load mesh with error handling
        try:
            mesh_set.load_new_mesh(input_path)
        except Exception as e:
            print(f"Failed to load mesh {input_path}: {e}")
            return False
        
        # Check if mesh loaded successfully
        if mesh_set.current_mesh().vertex_number() == 0:
            print(f"Empty mesh loaded from {input_path}")
            return False
        
        # Clean mesh with individual error handling
        cleaning_filters = [
            "meshing_remove_duplicate_faces", "meshing_remove_duplicate_vertices",
            "meshing_remove_unreferenced_vertices", "meshing_remove_null_faces",
            "meshing_repair_non_manifold_edges", "meshing_repair_non_manifold_vertices"
        ]
        
        for filter_name in cleaning_filters:
            try:
                mesh_set.apply_filter(filter_name)
            except Exception as e:
                print(f"Warning: {filter_name} failed: {e}")
                continue  # Continue with other filters
        
        # Remeshing loop with better error handling
        counter = 0
        max_iterations = 20
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while (mesh_set.current_mesh().vertex_number() != TARGET_VERTICES and 
               counter < max_iterations and consecutive_failures < max_consecutive_failures):
            counter += 1
            current_vertices = mesh_set.current_mesh().vertex_number()
            
            try:
                if current_vertices < TARGET_VERTICES:
                    mesh_set.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
                    consecutive_failures = 0  # Reset on success
                elif current_vertices > TARGET_VERTICES:
                    estimated_faces = int(mesh_set.current_mesh().face_number() * 
                                        (TARGET_VERTICES / current_vertices))
                    if estimated_faces > 0:  # Ensure valid face count
                        mesh_set.apply_filter("meshing_decimation_quadric_edge_collapse",
                                            targetfacenum=estimated_faces, qualitythr=0.5,
                                            preservenormal=True, preserveboundary=True,
                                            preservetopology=True, optimalplacement=True, autoclean=True)
                        consecutive_failures = 0  # Reset on success
                    else:
                        consecutive_failures += 1
                else:
                    break  # Target reached
                    
            except Exception as e:
                consecutive_failures += 1
                print(f"Remeshing iteration {counter} failed: {e}")
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Too many consecutive failures, stopping remeshing")
                    break
        
        # Save result with error handling
        try:
            mesh_set.save_current_mesh(output_path)
            final_vertices = mesh_set.current_mesh().vertex_number()
            print(f"Remeshing completed: {final_vertices} vertices (target: {TARGET_VERTICES})")
            return True
        except Exception as e:
            print(f"Failed to save remeshed mesh: {e}")
            return False
            
    except Exception as e:
        print(f"Critical error during remeshing: {e}")
        return False
    finally:
        # Clean up mesh_set to free memory
        try:
            if 'mesh_set' in locals():
                del mesh_set
        except:
            pass


def normalize_mesh(input_path: str, output_path: str) -> bool:
    """Normalize mesh (center and scale to unit size) with robust error handling."""
    if not os.path.exists(input_path):
        print(f"Input file not found for normalization: {input_path}")
        return False
    
    try:
        # Load mesh with error handling
        try:
            mesh = trimesh.load_mesh(input_path)
        except Exception as e:
            print(f"Failed to load mesh for normalization {input_path}: {e}")
            return False
        
        # Validate mesh
        if mesh is None:
            print(f"Mesh is None for {input_path}")
            return False
            
        if not hasattr(mesh, "vertices") or mesh.vertices is None:
            print(f"No vertices found in mesh {input_path}")
            return False
            
        if mesh.vertices.size == 0:
            print(f"Empty vertices in mesh {input_path}")
            return False
        
        # Check for valid geometry
        if len(mesh.vertices) < 3:
            print(f"Too few vertices in mesh {input_path}: {len(mesh.vertices)}")
            return False
        
        # Center at origin
        try:
            centroid = mesh.centroid
            if not all(not math.isnan(x) and not math.isinf(x) for x in centroid):
                print(f"Invalid centroid for mesh {input_path}: {centroid}")
                return False
            mesh.apply_translation(-centroid)
        except Exception as e:
            print(f"Failed to center mesh {input_path}: {e}")
            return False
        
        # Scale to unit size
        try:
            bounds = mesh.bounds
            if bounds is None or len(bounds) != 2:
                print(f"Invalid bounds for mesh {input_path}")
                return False
                
            size = bounds[1] - bounds[0]
            max_dimension = size.max()
            
            if max_dimension <= 0 or math.isnan(max_dimension) or math.isinf(max_dimension):
                print(f"Invalid mesh dimensions for {input_path}: {max_dimension}")
                return False
                
            mesh.apply_scale(1.0 / max_dimension)
        except Exception as e:
            print(f"Failed to scale mesh {input_path}: {e}")
            return False
        
        # Save result
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mesh.export(output_path)
            print(f"Normalization completed for {os.path.basename(input_path)}")
            return True
        except Exception as e:
            print(f"Failed to export normalized mesh {output_path}: {e}")
            return False
            
    except Exception as e:
        print(f"Critical error during normalization: {e}")
        return False
    finally:
        # Clean up mesh to free memory
        try:
            if 'mesh' in locals():
                del mesh
        except:
            pass


def cleanup_temp_folder() -> None:
    """Delete contents of temporary folder synchronously with timeout."""
    if not os.path.exists(TEMP_REMESH_DIR):
        return
    
    try:
        # Try to delete synchronously first (faster for small folders)
        for filename in os.listdir(TEMP_REMESH_DIR):
            file_path = os.path.join(TEMP_REMESH_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        # If synchronous fails, try async as fallback
        def _delete_contents():
            try:
                for filename in os.listdir(TEMP_REMESH_DIR):
                    file_path = os.path.join(TEMP_REMESH_DIR, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path, ignore_errors=True)
                    except Exception:
                        pass
            except Exception:
                pass
        
        thread = threading.Thread(target=_delete_contents, daemon=True)
        thread.start()
        # Don't wait for thread to complete - let it finish in background


class Shape:
    """3D mesh shape with loading, remeshing, and normalization capabilities."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vertices = None
        self.faces = None
        self.mesh = None
        self.temp_copy_path = None
        os.makedirs(TEMP_REMESH_DIR, exist_ok=True)

    def load(self) -> None:
        """Load mesh using vedo and parse info."""
        self.mesh = load(self.file_path)
        self.vertices, self.faces, _, _ = parse_obj_info(self.file_path)

    def resample(self) -> bool:
        """Remesh to target vertex count and save to temp folder."""
        if not os.path.exists(self.file_path):
            print(f"File not found for resampling: {self.file_path}")
            return False

        try:
            name, ext = os.path.splitext(os.path.basename(self.file_path))
            self.temp_copy_path = os.path.join(TEMP_REMESH_DIR, f"{name}_remesh{ext}")
            
            if remesh_to_target_vertices(self.file_path, self.temp_copy_path):
                try:
                    self.mesh = load(self.temp_copy_path)
                    self.vertices, self.faces, _, _ = parse_obj_info(self.temp_copy_path)
                    return True
                except Exception as e:
                    print(f"Failed to load remeshed file: {e}")
                    return False
            else:
                print(f"Remeshing failed for {self.file_path}")
                return False
        except Exception as e:
            print(f"Error during resampling: {e}")
            return False

    def normalize(self) -> bool:
        """Normalize mesh (center and scale) and save to temp folder."""
        if not os.path.exists(self.file_path):
            print(f"File not found for normalization: {self.file_path}")
            return False

        try:
            name, ext = os.path.splitext(os.path.basename(self.file_path))
            normalized_path = os.path.join(TEMP_REMESH_DIR, f"{name}_normalized{ext}")
            
            if normalize_mesh(self.file_path, normalized_path):
                try:
                    self.temp_copy_path = normalized_path
                    self.mesh = load(self.temp_copy_path)
                    self.vertices, self.faces, _, _ = parse_obj_info(self.temp_copy_path)
                    return True
                except Exception as e:
                    print(f"Failed to load normalized file: {e}")
                    return False
            else:
                print(f"Normalization failed for {self.file_path}")
                return False
        except Exception as e:
            print(f"Error during normalization: {e}")
            return False


class CBSRApp(QWidget):
    """3D Shape Browser and Processing GUI."""
    
    def __init__(self, parent_folder: str):
        super().__init__()
        self.parent_folder = parent_folder
        self.loaded_shapes = []
        self.current_mesh_actor = None
        self.bbox_actor = None
        self.origin_axes = None

        self.setWindowTitle("CBSR Debug GUI")
        self.resize(1200, 600)
        
        # Setup UI
        layout = QHBoxLayout(self)
        layout.addLayout(self._create_file_panel())
        layout.addLayout(self._create_viewer_panel())

        # Initialize with first category
        if self.categories:
            self.on_category_changed(self.categories[0])

    def _create_origin_axes(self) -> None:
        """Create origin axes (X=red, Y=green, Z=blue)."""
        # X-axis: red line from (0,0,0) to (1,0,0)
        x_axis = Line([0, 0, 0], [1, 0, 0]).c('red').lw(3)
        
        # Y-axis: green line from (0,0,0) to (0,1,0)
        y_axis = Line([0, 0, 0], [0, 1, 0]).c('green').lw(3)
        
        # Z-axis: blue line from (0,0,0) to (0,0,1)
        z_axis = Line([0, 0, 0], [0, 0, 1]).c('blue').lw(3)
        
        # Combine all axes
        self.origin_axes = [x_axis, y_axis, z_axis]
        
        # Add axes to plotter
        for axis in self.origin_axes:
            self.plotter.add(axis)

    def _create_file_panel(self) -> QVBoxLayout:
        """Create file browser panel."""
        panel = QVBoxLayout()
        
        # Categories dropdown (replaces main folder selection)
        panel.addWidget(QLabel("Categories"))
        self.category_combo = QComboBox()
        self.categories = [d for d in os.listdir(self.parent_folder) 
                          if os.path.isdir(os.path.join(self.parent_folder, d))]
        self.category_combo.addItems(self.categories)
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        panel.addWidget(self.category_combo)

        # Files list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        panel.addWidget(QLabel("Files"))
        panel.addWidget(self.file_list)
        
        return panel

    def _create_viewer_panel(self) -> QVBoxLayout:
        """Create 3D viewer panel."""
        panel = QVBoxLayout()
        
        panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        panel.addWidget(self.viewer_widget)

        # Create and add origin axes
        self._create_origin_axes()

        self.info_label = QLabel("Select a file to see info.")
        panel.addWidget(self.info_label)

        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        panel.addWidget(self.bbox_toggle)

        self.clean_button = QPushButton("Clean (Remesh + Normalize)")
        self.clean_button.clicked.connect(self.on_clean_clicked)
        panel.addWidget(self.clean_button)
        
        return panel

    def on_category_changed(self, category_name: str) -> None:
        """Handle category selection change."""
        self.file_list.clear()
        self.current_category = category_name
        category_path = os.path.join(self.parent_folder, category_name)
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)

    def on_file_selected(self, item) -> None:
        """Handle file selection and display mesh."""
        full_path = os.path.join(self.parent_folder, self.current_category, item.text())

        shape = Shape(full_path)
        shape.load()
        self.loaded_shapes.append(shape)

        self.plotter.clear()
        # Re-add origin axes after clearing
        for axis in self.origin_axes:
            self.plotter.add(axis)
        
        shape.mesh.wireframe(True)
        self.plotter.show(shape.mesh, resetcam=True)
        self.current_mesh_actor = shape.mesh

        self.info_label.setText(f"File: {item.text()}\nVertices: {shape.vertices}\nFaces: {shape.faces}")
        self.bbox_toggle.setChecked(False)

    def on_bbox_toggle(self, state) -> None:
        """Toggle bounding box display."""
        if not self.current_mesh_actor:
            return
            
        if state:
            self.bbox_actor = Box(self.current_mesh_actor.bounds()).wireframe().c('red')
            self.plotter.add(self.bbox_actor)
        else:
            if self.bbox_actor:
                self.plotter.remove(self.bbox_actor)
                self.bbox_actor = None
        self.plotter.render()

    def on_clean_clicked(self) -> None:
        """Clean mesh (remesh + normalize) with robust error handling."""
        if not self.loaded_shapes:
            self.info_label.setText("No shape loaded to clean!")
            return

        shape = self.loaded_shapes[-1]
        self.info_label.setText("Cleaning (remesh + normalize), please wait...")
        QApplication.processEvents()

        try:
            # Step 1: Remesh
            self.info_label.setText("Step 1/2: Remeshing...")
            QApplication.processEvents()
            
            if not shape.resample():
                self.info_label.setText("Cleaning failed: Remeshing step failed.\nCheck console for details.")
                return

            # Step 2: Normalize
            self.info_label.setText("Step 2/2: Normalizing...")
            QApplication.processEvents()
            
            # Use remeshed version if available, otherwise use original
            input_path = shape.temp_copy_path if shape.temp_copy_path and os.path.exists(shape.temp_copy_path) else shape.file_path
            
            # Create a temporary shape for normalization
            temp_shape = Shape(input_path)
            if not temp_shape.normalize():
                self.info_label.setText("Cleaning failed: Normalization step failed.\nCheck console for details.")
                return

            # Update the original shape with normalized result
            shape.temp_copy_path = temp_shape.temp_copy_path
            shape.mesh = temp_shape.mesh
            shape.vertices = temp_shape.vertices
            shape.faces = temp_shape.faces

            # Display result
            self.plotter.clear()
            # Re-add origin axes after clearing
            for axis in self.origin_axes:
                self.plotter.add(axis)
            
            shape.mesh.wireframe(True)
            self.plotter.show(shape.mesh, resetcam=True)
            self.current_mesh_actor = shape.mesh
            self.info_label.setText(f"✅ Cleaned successfully!\n"
                                   f"File: {os.path.basename(shape.file_path)}\n"
                                   f"Vertices: {shape.vertices}\nFaces: {shape.faces}")
            
        except Exception as e:
            print(f"Unexpected error during cleaning: {e}")
            self.info_label.setText(f"Cleaning failed: Unexpected error.\nCheck console for details.\nError: {str(e)}")
            return

    def closeEvent(self, event) -> None:
        """Handle application close with proper cleanup."""
        try:
            # Clear the plotter and release 3D resources
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.clear()
                self.plotter.close()
            
            # Clear loaded shapes to free memory
            self.loaded_shapes.clear()
            self.current_mesh_actor = None
            self.bbox_actor = None
            self.origin_axes = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Cleanup temp folder
            cleanup_temp_folder()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            event.accept()


if __name__ == '__main__':
    import signal
    
    def signal_handler(signum, frame):
        """Handle system signals for clean shutdown."""
        print("Received signal, shutting down...")
        cleanup_temp_folder()
        sys.exit(0)
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)  # Ensure app quits when window is closed
    
    window = CBSRApp(SHAPEDATA_PARENT)
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Interrupted by user")
        cleanup_temp_folder()
        sys.exit(0)
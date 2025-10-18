"""
Compact 3D Shape Browser and Processing GUI v3
"""
import sys
import os
import math
import threading
import shutil
import random
from typing import Tuple, Optional, List
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox, QPushButton
from PyQt6.QtGui import QPalette, QColor
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
            if not parts: continue
            if parts[0] == 'v' and len(parts) == 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append(parts[1:])
    
    # Determine face types and bounding box in one pass
    face_types = {"triangles" if len(f) == 3 else "quads" if len(f) == 4 else "other" for f in faces}
    face_type = " and ".join(sorted(face_types)) if face_types else "unknown"
    
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
    
    mesh_set = None
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh_set = ml.MeshSet()
        mesh_set.load_new_mesh(input_path)
        
        if mesh_set.current_mesh().vertex_number() == 0:
            print(f"Empty mesh loaded from {input_path}")
            return False
        
        # Clean mesh
        for filter_name in ["meshing_remove_duplicate_faces", "meshing_remove_duplicate_vertices",
                           "meshing_remove_unreferenced_vertices", "meshing_remove_null_faces",
                           "meshing_repair_non_manifold_edges", "meshing_repair_non_manifold_vertices"]:
            try:
                mesh_set.apply_filter(filter_name)
            except Exception as e:
                print(f"Warning: {filter_name} failed: {e}")
        
        # Remesh to target
        counter = consecutive_failures = 0
        while (mesh_set.current_mesh().vertex_number() != TARGET_VERTICES and 
               counter < 20 and consecutive_failures < 3):
            counter += 1
            current_vertices = mesh_set.current_mesh().vertex_number()
            
            try:
                if current_vertices < TARGET_VERTICES:
                    mesh_set.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
                    consecutive_failures = 0
                elif current_vertices > TARGET_VERTICES:
                    estimated_faces = int(mesh_set.current_mesh().face_number() * (TARGET_VERTICES / current_vertices))
                    if estimated_faces > 0:
                        mesh_set.apply_filter("meshing_decimation_quadric_edge_collapse",
                                            targetfacenum=estimated_faces, qualitythr=0.5,
                                            preservenormal=True, preserveboundary=True,
                                            preservetopology=True, optimalplacement=True, autoclean=True)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                else:
                    break
            except Exception as e:
                consecutive_failures += 1
                print(f"Remeshing iteration {counter} failed: {e}")
                if consecutive_failures >= 3:
                    print("Too many consecutive failures, stopping remeshing")
                    break
        
        mesh_set.save_current_mesh(output_path)
        print(f"Remeshing completed: {mesh_set.current_mesh().vertex_number()} vertices (target: {TARGET_VERTICES})")
        return True
        
    except Exception as e:
        print(f"Error during remeshing: {e}")
        return False
    finally:
        if mesh_set: del mesh_set


def normalize_mesh(input_path: str, output_path: str) -> bool:
    """Normalize mesh (center, scale, align, and flip) with robust error handling."""
    if not os.path.exists(input_path):
        print(f"Input file not found for normalization: {input_path}")
        return False
    
    mesh = None
    try:
        import numpy as np
        mesh = trimesh.load_mesh(input_path)
        
        # Validate mesh
        if not mesh or not hasattr(mesh, "vertices") or mesh.vertices is None or mesh.vertices.size == 0 or len(mesh.vertices) < 3:
            print(f"Invalid mesh for {input_path}")
            return False
        
        # Step 1: Center at origin
        centroid = mesh.centroid
        if not all(not math.isnan(x) and not math.isinf(x) for x in centroid):
            print(f"Invalid centroid for mesh {input_path}: {centroid}")
            return False
        mesh.apply_translation(-centroid)
        
        # Step 2: Scale to unit size
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
        
        # Step 3: Alignment using PCA
        covariance_matrix = np.cov(mesh.vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Normalize eigenvectors and ensure right-handed coordinate system
        e1, e2, e3 = [sorted_eigenvectors[:, i] / np.linalg.norm(sorted_eigenvectors[:, i]) for i in range(3)]
        if np.dot(e3, np.cross(e1, e2)) < 0: e3 = -e3
        
        # Transform vertices
        rotation_matrix = np.column_stack([e1, e2, e3])
        aligned_vertices = mesh.vertices @ rotation_matrix
        mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=mesh.faces, process=False)
        
        # Step 4: Flipping based on triangle center analysis
        triangle_centers = mesh.vertices[mesh.faces].mean(axis=1)
        flip_factors = np.array([np.sign(np.sum(np.sign(triangle_centers[:, axis]) * (triangle_centers[:, axis] ** 2))) for axis in range(3)])
        
        # Apply flipping
        flipped_vertices = mesh.vertices * flip_factors
        faces = np.fliplr(mesh.faces) if np.prod(flip_factors) == -1 else mesh.faces
        mesh = trimesh.Trimesh(vertices=flipped_vertices, faces=faces, process=False)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        print(f"Full normalization completed for {os.path.basename(input_path)}")
        return True
        
    except Exception as e:
        print(f"Error during normalization: {e}")
        return False
    finally:
        if mesh: del mesh


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
            self.temp_copy_path = os.path.join(TEMP_REMESH_DIR, f"{name}_processed{ext}")
            
            if remesh_to_target_vertices(self.file_path, self.temp_copy_path):
                self._load_processed_mesh()
                return True
            else:
                print(f"Remeshing failed for {self.file_path}")
                return False
        except Exception as e:
            print(f"Error during resampling: {e}")
            return False

    def normalize(self) -> bool:
        """Normalize mesh (center, scale, align, flip) in place on temp file."""
        input_path = self.temp_copy_path if self.temp_copy_path and os.path.exists(self.temp_copy_path) else self.file_path
        
        if not os.path.exists(input_path):
            print(f"File not found for normalization: {input_path}")
            return False

        try:
            if normalize_mesh(input_path, self.temp_copy_path):
                self._load_processed_mesh()
                return True
            else:
                print(f"Normalization failed for {input_path}")
                return False
        except Exception as e:
            print(f"Error during normalization: {e}")
            return False

    def _load_processed_mesh(self) -> None:
        """Load processed mesh from temp file."""
        try:
            self.mesh = load(self.temp_copy_path)
            self.vertices, self.faces, _, _ = parse_obj_info(self.temp_copy_path)
        except Exception as e:
            print(f"Failed to load processed file: {e}")
            raise


class CBSRApp(QWidget):
    """3D Shape Browser and Processing GUI."""
    
    def __init__(self, parent_folder: str):
        super().__init__()
        self.parent_folder = parent_folder
        self.loaded_shapes = []
        self.current_mesh_actor = None
        self.bbox_actor = None
        self.bbox_labels = []
        self.origin_axes = None
        self.show_bbox_preference = False
        self.show_reference_preference = True
        self.dark_mode_enabled = False

        self.setWindowTitle("CBSR Debug GUI")
        self.resize(1200, 900)
        
        # Setup UI: top row (file panel + main viewer), bottom row (gallery spans full width)
        root_layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        top_row.addLayout(self._create_file_panel())
        top_row.addLayout(self._create_viewer_panel())
        root_layout.addLayout(top_row)
        root_layout.addLayout(self._create_gallery_panel())

        # Initialize with first category
        if self.categories:
            self.on_category_changed(self.categories[0])

    def _create_origin_axes(self) -> None:
        """Create origin axes (X=red, Y=green, Z=blue)."""
        # X-axis: red line
        x_axis = Line([0, 0, 0], [0.5, 0, 0]).c('red').lw(1)
        
        # Y-axis: green line
        y_axis = Line([0, 0, 0], [0, 0.5, 0]).c('green').lw(1)
        
        # Z-axis: blue line
        z_axis = Line([0, 0, 0], [0, 0, 0.5]).c('blue').lw(1)
        
        # Combine all reference objects (only axes, no cube)
        self.origin_axes = [x_axis, y_axis, z_axis]
        
        # Add reference objects to plotter
        for obj in self.origin_axes:
            self.plotter.add(obj)

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

        self.reference_toggle = QCheckBox("Show Reference Axes")
        self.reference_toggle.stateChanged.connect(self.on_reference_toggle)
        self.reference_toggle.setChecked(self.show_reference_preference)
        panel.addWidget(self.reference_toggle)

        self.darkmode_toggle = QCheckBox("Dark Mode")
        self.darkmode_toggle.stateChanged.connect(self.on_darkmode_toggle)
        panel.addWidget(self.darkmode_toggle)

        self.auto_normalize_toggle = QCheckBox("Auto-Normalize")
        self.auto_normalize_toggle.stateChanged.connect(self.on_auto_normalize_toggle)
        panel.addWidget(self.auto_normalize_toggle)
        return panel

    def _create_gallery_panel(self) -> QVBoxLayout:
        """Create bottom gallery panel that spans the full window width."""
        panel = QVBoxLayout()
        panel.addWidget(QLabel("Gallery (5 random from current category)"))
        self.gallery_layout = QHBoxLayout()
        self.gallery_widgets: List[QVTKRenderWindowInteractor] = []
        self.gallery_plotters: List[Plotter] = []
        for _ in range(5):
            w = QVTKRenderWindowInteractor(self)
            p = Plotter(qt_widget=w)
            self.gallery_widgets.append(w)
            self.gallery_plotters.append(p)
            self.gallery_layout.addWidget(w)
        panel.addLayout(self.gallery_layout)
        self.refresh_gallery_button = QPushButton("Refresh Gallery")
        self.refresh_gallery_button.clicked.connect(self.load_random_gallery)
        panel.addWidget(self.refresh_gallery_button)
        return panel

    def on_category_changed(self, category_name: str) -> None:
        """Handle category selection change."""
        self.file_list.clear()
        self.current_category = category_name
        category_path = os.path.join(self.parent_folder, category_name)
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)
        # Update gallery when category changes
        self.load_random_gallery()

    def on_file_selected(self, item) -> None:
        """Handle file selection and display mesh."""
        full_path = os.path.join(self.parent_folder, self.current_category, item.text())

        shape = Shape(full_path)
        shape.load()
        self.loaded_shapes.append(shape)

        # If auto-normalize is enabled, normalize the shape before displaying
        if self.auto_normalize_toggle.isChecked():
            self.info_label.setText("Auto-normalizing, please wait...")
            QApplication.processEvents()
            
            # First resample to target vertices
            if not shape.resample():
                self.info_label.setText("Auto-normalize failed: Remeshing step failed.")
                return
            
            # Then normalize
            if not shape.normalize():
                self.info_label.setText("Auto-normalize failed: Normalization step failed.")
                return

        self.plotter.clear()
        # Re-add origin axes after clearing (only if reference toggle is on)
        if self.show_reference_preference:
            for axis in self.origin_axes:
                self.plotter.add(axis)
        
        # Display mesh with lighting enabled (like pressing 'L' key)
        shape.mesh.lighting('default').linecolor('black').linewidth(1)
        self.plotter.show(shape.mesh, resetcam=True)
        self.current_mesh_actor = shape.mesh

        status_text = "(Auto-Normalized) " if self.auto_normalize_toggle.isChecked() else ""
        self.info_label.setText(f"{status_text}File: {item.text()}\nVertices: {shape.vertices}\nFaces: {shape.faces}")
        self.bbox_toggle.setChecked(self.show_bbox_preference)
        self.reference_toggle.setChecked(self.show_reference_preference)
        
        # If preference is to show bounding box, trigger it
        if self.show_bbox_preference:
            self.on_bbox_toggle(True)
        
        # If preference is to hide reference objects, remove them
        if not self.show_reference_preference:
            self.on_reference_toggle(False)

    def on_bbox_toggle(self, state) -> None:
        """Toggle bounding box display with dimension labels."""
        if not self.current_mesh_actor:
            return
        
        # Remember the user's preference
        self.show_bbox_preference = bool(state)
            
        if state:
            try:
                # Create bounding box
                bounds = self.current_mesh_actor.bounds()
                self.bbox_actor = Box(bounds).wireframe().c('grey')
                self.plotter.add(self.bbox_actor)
                
                # Calculate dimensions
                x_size = bounds[1] - bounds[0]  # xmax - xmin
                y_size = bounds[3] - bounds[2]  # ymax - ymin
                z_size = bounds[5] - bounds[4]  # zmax - zmin
                
                # Update info label with dimensions
                current_info = self.info_label.text()
                dimension_info = f"\nBBox: X={x_size:.2f}, Y={y_size:.2f}, Z={z_size:.2f}"
                self.info_label.setText(current_info + dimension_info)
                
                # Store that we have labels (for cleanup)
                self.bbox_labels = ['info_updated']
                
            except Exception as e:
                print(f"Error creating bounding box: {e}")
                self.info_label.setText(f"Bounding box error: {str(e)}")
        else:
            # Remove bounding box
            if self.bbox_actor:
                try:
                    self.plotter.remove(self.bbox_actor)
                except Exception as e:
                    print(f"Error removing bounding box: {e}")
                self.bbox_actor = None
            
            # Remove dimension info from label
            if self.bbox_labels:
                current_info = self.info_label.text()
                # Remove the bbox info line if it exists
                lines = current_info.split('\n')
                filtered_lines = [line for line in lines if not line.startswith('BBox:')]
                self.info_label.setText('\n'.join(filtered_lines))
            
            self.bbox_labels = []
            
        try:
            self.plotter.render()
        except Exception as e:
            print(f"Error rendering: {e}")

    def on_reference_toggle(self, state) -> None:
        """Toggle reference objects (unit cube and axes) display."""
        # Remember the user's preference
        self.show_reference_preference = bool(state)
        
        if state:
            # Add reference objects back
            for obj in self.origin_axes:
                self.plotter.add(obj)
        else:
            # Remove reference objects
            for obj in self.origin_axes:
                self.plotter.remove(obj)
        self.plotter.render()

    def on_darkmode_toggle(self, state) -> None:
        """Toggle dark mode for the entire app and 3D viewers."""
        self.dark_mode_enabled = bool(state)
        self._apply_qt_palette(self.dark_mode_enabled)
        # Update backgrounds of all plotters
        try:
            bg = 'black' if self.dark_mode_enabled else 'white'
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.background(bg)
                self.plotter.render()
            if hasattr(self, 'gallery_plotters'):
                for p in self.gallery_plotters:
                    try:
                        p.background(bg)
                        p.render()
                    except Exception:
                        pass
        except Exception:
            pass

    def _apply_qt_palette(self, dark: bool) -> None:
        """Apply a dark or default palette to the Qt application."""
        app = QApplication.instance()
        if app is None:
            return
        if not dark:
            app.setPalette(QPalette())
            return
        palette = QPalette()
        # Window
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        # Base/Alternate
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        # Tooltips
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(53, 53, 53))
        # Text/Button
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        # Bright/Dark/Shadow
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(240, 240, 240))
        app.setPalette(palette)

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

            # Step 2: Normalize (in place on the same temp file)
            self.info_label.setText("Step 2/2: Normalizing...")
            QApplication.processEvents()
            
            if not shape.normalize():
                self.info_label.setText("Cleaning failed: Normalization step failed.\nCheck console for details.")
                return

            # Display result
            self.plotter.clear()
            # Re-add origin axes after clearing (only if reference toggle is on)
            if self.show_reference_preference:
                for axis in self.origin_axes:
                    self.plotter.add(axis)
            
            # Display cleaned mesh with lighting enabled (like pressing 'L' key)
            shape.mesh.lighting('default').linecolor('black').linewidth(1)
            self.plotter.show(shape.mesh, resetcam=True)
            self.current_mesh_actor = shape.mesh
            
            # Re-apply bounding box if it was previously shown
            if self.show_bbox_preference:
                self.bbox_actor = Box(self.current_mesh_actor.bounds()).wireframe().c('red')
                self.plotter.add(self.bbox_actor)
            
            self.info_label.setText(f"Cleaned successfully!\n"
                                   f"File: {os.path.basename(shape.file_path)}\n"
                                   f"Vertices: {shape.vertices}\nFaces: {shape.faces}")
            
        except Exception as e:
            print(f"Unexpected error during cleaning: {e}")
            self.info_label.setText(f"Cleaning failed: Unexpected error.\nCheck console for details.\nError: {str(e)}")
            return

    def on_auto_normalize_toggle(self, state) -> None:
        """Handle auto-normalize checkbox state change and immediately update viewer."""
        # If no shape is currently loaded, just update the status message
        if not self.loaded_shapes:
            if state:
                self.info_label.setText("Auto-normalize enabled. Select a file to see normalized version.")
            else:
                self.info_label.setText("Auto-normalize disabled. Objects will be shown as original.")
            return
        
        # Get the current shape
        current_shape = self.loaded_shapes[-1]
        
        if state:
            # Checkbox checked - show normalized version
            self.info_label.setText("Normalizing current object, please wait...")
            QApplication.processEvents()
            
            # Create a fresh copy of the shape for normalization
            temp_shape = Shape(current_shape.file_path)
            temp_shape.load()
            
            # Normalize the temp shape
            if not temp_shape.resample():
                self.info_label.setText("Auto-normalize failed: Remeshing step failed.")
                return
            
            if not temp_shape.normalize():
                self.info_label.setText("Auto-normalize failed: Normalization step failed.")
                return
            
            # Update the loaded shape with normalized version
            self.loaded_shapes[-1] = temp_shape
            current_shape = temp_shape
            status_prefix = "(Auto-Normalized) "
        else:
            # Checkbox unchecked - reload original version
            self.info_label.setText("Loading original object...")
            QApplication.processEvents()
            
            # Reload the original shape
            original_shape = Shape(current_shape.file_path)
            original_shape.load()
            
            # Update the loaded shape with original version
            self.loaded_shapes[-1] = original_shape
            current_shape = original_shape
            status_prefix = ""
        
        # Clear and redisplay the mesh
        self.plotter.clear()
        
        # Re-add origin axes after clearing (only if reference toggle is on)
        if self.show_reference_preference:
            for axis in self.origin_axes:
                self.plotter.add(axis)
        
        # Display mesh with lighting enabled
        current_shape.mesh.lighting('default').linecolor('black').linewidth(1)
        self.plotter.show(current_shape.mesh, resetcam=True)
        self.current_mesh_actor = current_shape.mesh
        
        # Update info label
        filename = os.path.basename(current_shape.file_path)
        self.info_label.setText(f"{status_prefix}File: {filename}\nVertices: {current_shape.vertices}\nFaces: {current_shape.faces}")
        
        # Re-apply bounding box if it was previously shown
        if self.show_bbox_preference:
            self.on_bbox_toggle(True)

    def _list_obj_files_in_current_category(self) -> List[str]:
        """List absolute paths to .obj files in the current category."""
        if not getattr(self, 'current_category', None):
            return []
        category_path = os.path.join(self.parent_folder, self.current_category)
        try:
            files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.lower().endswith('.obj')]
            return files
        except Exception:
            return []

    def load_random_gallery(self) -> None:
        """Load 5 random objects into the gallery viewers from current category."""
        obj_files = self._list_obj_files_in_current_category()
        if not obj_files:
            # Clear gallery if nothing available
            for p in getattr(self, 'gallery_plotters', []):
                try:
                    p.clear()
                    p.render()
                except Exception:
                    pass
            return

        choices = random.sample(obj_files, k=min(5, len(obj_files)))
        # Ensure we have 5 entries by repeating if fewer than 5 files exist
        while len(choices) < 5:
            choices.append(random.choice(obj_files))

        for plotter, path in zip(self.gallery_plotters, choices):
            try:
                plotter.clear()
                mesh = load(path)
                mesh.lighting('default').linecolor('black').linewidth(1)
                plotter.show(mesh, resetcam=True)
            except Exception:
                try:
                    plotter.clear()
                    plotter.render()
                except Exception:
                    pass
        # Enforce square viewers after content is shown
        self._enforce_gallery_square_sizes()

    def _enforce_gallery_square_sizes(self) -> None:
        """Adjust gallery viewer heights to keep them square (height = width)."""
        if not hasattr(self, 'gallery_widgets'):
            return
        for w in self.gallery_widgets:
            try:
                w.setFixedHeight(max(0, w.width()))
            except Exception:
                pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep gallery viewers square on window resize
        self._enforce_gallery_square_sizes()

    def closeEvent(self, event) -> None:
        """Handle application close with proper cleanup."""
        try:
            # Clear the plotter and release 3D resources
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.clear()
                self.plotter.close()
            # Close gallery plotters
            if hasattr(self, 'gallery_plotters') and self.gallery_plotters:
                for p in self.gallery_plotters:
                    try:
                        p.clear()
                        p.close()
                    except Exception:
                        pass
            
            # Clear loaded shapes to free memory
            self.loaded_shapes.clear()
            self.current_mesh_actor = None
            self.bbox_actor = None
            self.bbox_labels = []
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
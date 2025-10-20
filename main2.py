"""
Compact 3D Shape Browser and Processing GUI v3
"""
import sys
import os
import math
import threading
import shutil

import json
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox, QPushButton
from PyQt6.QtGui import QPalette, QColor, QPainter, QPen
from PyQt6.QtCore import Qt
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


class HistogramWidget(QWidget):
    """Widget to display mini histograms stacked vertically."""
    
    def __init__(self):
        super().__init__()
        self.histograms = []
        self.setFixedSize(60, 100)
    
    def set_histograms(self, histograms: List[List[float]]):
        self.histograms = histograms
        self.update()
    
    def paintEvent(self, event):
        if not self.histograms:
            return
        
        painter = QPainter(self)
        hist_names = ['D1', 'D2', 'D3', 'D4', 'A3']
        hist_height = self.height() // max(1, len(self.histograms))
        
        for i, hist_data in enumerate(self.histograms):
            if i < len(hist_names) and hist_data:
                y_offset = i * hist_height
                max_val = max(hist_data) if hist_data else 1.0
                if max_val == 0: max_val = 1.0
                
                painter.setPen(QPen(Qt.GlobalColor.blue, 1))
                for j, value in enumerate(hist_data[:40]):  # Max 40 bars
                    bar_height = int((value / max_val) * (hist_height - 10))
                    if bar_height > 0:
                        x = j
                        y = y_offset + hist_height - bar_height - 8
                        painter.drawLine(x, y, x, y + bar_height)
                
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawText(42, y_offset + hist_height // 2, hist_names[i])


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
        
        # Load features dictionary
        try:
            with open('features_dictionary.json', 'r') as f:
                self.features_dict = json.load(f)
            print(f"Loaded features for {len(self.features_dict)} categories")
        except Exception as e:
            print(f"Error loading features: {e}")
            self.features_dict = {}

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
        
        # Create horizontal layout for metrics and histograms
        features_layout = QHBoxLayout()
        
        # Add metrics display label for main viewer
        self.metrics_label = QLabel("Metrics will appear here")
        self.metrics_label.setStyleSheet("font-family: monospace; font-size: 9px; color: #666;")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        features_layout.addWidget(self.metrics_label)
        
        # Add histogram display widget for main viewer
        self.histogram_widget = HistogramWidget()
        features_layout.addWidget(self.histogram_widget)
        
        panel.addLayout(features_layout)

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

        self.auto_normalize_toggle = QCheckBox("Normalize")
        self.auto_normalize_toggle.stateChanged.connect(self.on_auto_normalize_toggle)
        panel.addWidget(self.auto_normalize_toggle)
        return panel

    def _create_gallery_panel(self) -> QVBoxLayout:
        """Create bottom gallery panel that spans the full window width."""
        panel = QVBoxLayout()
        panel.addWidget(QLabel("Gallery (5 most similar objects)"))
        self.gallery_layout = QHBoxLayout()
        self.gallery_widgets: List[QVTKRenderWindowInteractor] = []
        self.gallery_plotters: List[Plotter] = []
        self.gallery_metrics_labels: List[QLabel] = []
        self.gallery_histogram_widgets: List[HistogramWidget] = []
        self.gallery_distance_labels: List[QLabel] = []
        
        for i in range(5):
            # Create vertical layout for each gallery item (viewer + features)
            item_layout = QVBoxLayout()
            
            # 3D viewer widget
            w = QVTKRenderWindowInteractor(self)
            p = Plotter(qt_widget=w)
            self.gallery_widgets.append(w)
            self.gallery_plotters.append(p)
            item_layout.addWidget(w)
            
            # Create horizontal layout for metrics and histograms
            features_layout = QHBoxLayout()
            
            # Metrics label for this gallery item
            metrics_label = QLabel("Metrics will appear here")
            metrics_label.setStyleSheet("font-family: monospace; font-size: 7px; color: #555; max-height: 80px;")
            metrics_label.setWordWrap(True)
            metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.gallery_metrics_labels.append(metrics_label)
            features_layout.addWidget(metrics_label)
            
            # Histogram widget for this gallery item
            histogram_widget = HistogramWidget()
            self.gallery_histogram_widgets.append(histogram_widget)
            features_layout.addWidget(histogram_widget)
            
            item_layout.addLayout(features_layout)
            
            # Distance score label for this gallery item
            distance_label = QLabel("Similarity: --")
            distance_label.setStyleSheet("font-family: monospace; font-size: 8px; color: #007acc; font-weight: bold;")
            distance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gallery_distance_labels.append(distance_label)
            item_layout.addWidget(distance_label)
            
            # Add the complete item layout to gallery
            self.gallery_layout.addLayout(item_layout)
            
        panel.addLayout(self.gallery_layout)
        self.refresh_gallery_button = QPushButton("Find Similar Objects")
        self.refresh_gallery_button.clicked.connect(self._refresh_gallery_smart)
        panel.addWidget(self.refresh_gallery_button)
        return panel

    def on_category_changed(self, category_name: str) -> None:
        """Handle category selection change."""
        self.file_list.clear()
        self.current_category = category_name
        category_path = os.path.join(self.parent_folder, category_name)
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)
        # Clear gallery when category changes
        self._clear_gallery()

    def on_file_selected(self, item) -> None:
        """Handle file selection and display mesh."""
        full_path = os.path.join(self.parent_folder, self.current_category, item.text())

        shape = Shape(full_path)
        shape.load()
        self.loaded_shapes.append(shape)

        # If auto-normalize is enabled, normalize the shape before displaying
        if self.auto_normalize_toggle.isChecked():
            self.info_label.setText("Normalizing, please wait...")
            QApplication.processEvents()
            
            # First resample to target vertices
            if not shape.resample():
                self.info_label.setText("Normalize failed: Remeshing step failed.")
                return
            
            # Then normalize
            if not shape.normalize():
                self.info_label.setText("Normalize failed: Normalization step failed.")
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

        status_text = "(Normalized) " if self.auto_normalize_toggle.isChecked() else ""
        self.info_label.setText(f"{status_text}File: {item.text()}\nVertices: {shape.vertices}")
        
        # Update features display for main viewer
        self._update_main_viewer_features(item.text())
        
        # Load similar objects to gallery
        self._load_similar_objects_to_gallery(item.text())
        
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
            
        # Always remove existing dimension info first (to prevent stacking)
        current_info = self.info_label.text()
        lines = current_info.split('\n')
        filtered_lines = [line for line in lines if not (line.startswith('BBox:') or line.startswith('[X='))]
        base_info = '\n'.join(filtered_lines)
        
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
                
                # Add dimension info to clean base info
                dimension_info = f"\n[X={x_size:.2f}, Y={y_size:.2f}, Z={z_size:.2f}]"
                self.info_label.setText(base_info + dimension_info)
                
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
            
            # Set info label to clean base info (dimensions already removed above)
            self.info_label.setText(base_info)
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
                                   f"Vertices: {shape.vertices}")
            
        except Exception as e:
            print(f"Unexpected error during cleaning: {e}")
            self.info_label.setText(f"Cleaning failed: Unexpected error.\nCheck console for details.\nError: {str(e)}")
            return

    def on_auto_normalize_toggle(self, state) -> None:
        """Handle auto-normalize checkbox state change and immediately update viewer."""
        # If no shape is currently loaded, just update the status message
        if not self.loaded_shapes:
            if state:
                self.info_label.setText("Normalize enabled. Select a file to see normalized version.")
            else:
                self.info_label.setText("Normalize disabled. Objects will be shown as original.")
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
                self.info_label.setText("Normalize failed: Remeshing step failed.")
                return
            
            if not temp_shape.normalize():
                self.info_label.setText("Normalize failed: Normalization step failed.")
                return
            
            # Update the loaded shape with normalized version
            self.loaded_shapes[-1] = temp_shape
            current_shape = temp_shape
            status_prefix = "(Normalized) "
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
        self.info_label.setText(f"{status_prefix}File: {filename}\nVertices: {current_shape.vertices}")
        
        # Re-apply bounding box if it was previously shown
        if self.show_bbox_preference:
            self.on_bbox_toggle(True)





    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep gallery viewers square on window resize
        for w in getattr(self, 'gallery_widgets', []):
            try:
                w.setFixedHeight(max(0, w.width()))
            except Exception:
                pass

    def _get_features_for_file(self, filename):
        """Extract features for a given filename from the features dictionary."""
        base_filename = os.path.splitext(filename)[0]
        for category, files in self.features_dict.items():
            if base_filename in files:
                file_data = files[base_filename]
                metrics = dict(list(file_data.get('Metrics', {}).items())[:8])
                histograms = [file_data[k]['histogram_data'] for k in ['D1_hist', 'D2_hist', 'D3_hist', 'D4_hist', 'A3_hist'] if k in file_data and file_data[k].get('histogram_data')]
                return metrics, histograms
        return {}, []
    
    def _format_metrics_text(self, metrics):
        """Format metrics dictionary into display text."""
        return "\n".join(f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items()) or "No metrics available"
    
    def _update_main_viewer_features(self, filename):
        """Update the main viewer with features for the selected file."""
        metrics, histograms = self._get_features_for_file(filename)
        self.metrics_label.setText(self._format_metrics_text(metrics))
        self.histogram_widget.set_histograms(histograms)

    def _update_gallery_item_features(self, item_index, filename):
        """Update features for a specific gallery item."""
        if item_index < len(self.gallery_metrics_labels):
            metrics, histograms = self._get_features_for_file(filename)
            self.gallery_metrics_labels[item_index].setText(self._format_metrics_text(metrics))
            self.gallery_histogram_widgets[item_index].set_histograms(histograms)
    
    def _extract_feature_vector(self, filename):
        """Extract normalized feature vector for comparison."""
        metrics, histograms = self._get_features_for_file(filename)
        if not metrics or not histograms:
            return None
        
        # Combine histogram data (5 histograms with 20 bins each = 100 values)
        hist_vector = []
        for hist in histograms[:5]:  # Use first 5 histograms
            hist_array = np.array(hist[:20])  # Use first 20 bins
            hist_sum = np.sum(hist_array)
            if hist_sum > 0:
                hist_array = hist_array / hist_sum  # Normalize histogram
            hist_vector.extend(hist_array)
        
        # Add scalar metrics (8 metrics) - normalize these too
        scalar_vector = []
        metric_values = list(metrics.values())[:8]
        for value in metric_values:
            if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                scalar_vector.append(float(value))
            else:
                scalar_vector.append(0.0)
        
        # Normalize scalar metrics using z-score normalization
        if len(scalar_vector) > 0:
            scalar_array = np.array(scalar_vector)
            # Use robust normalization to handle outliers
            scalar_std = np.std(scalar_array) if np.std(scalar_array) > 1e-8 else 1.0
            scalar_mean = np.mean(scalar_array)
            scalar_array = (scalar_array - scalar_mean) / scalar_std
            scalar_vector = scalar_array.tolist()
        
        # Combine and normalize the entire feature vector
        full_vector = np.array(hist_vector + scalar_vector)
        
        # L2 normalization to make all vectors unit length
        vector_norm = np.linalg.norm(full_vector)
        if vector_norm > 1e-8:
            full_vector = full_vector / vector_norm
        
        return full_vector
    
    def _compute_distance(self, vec1, vec2):
        """Compute Euclidean distance between two feature vectors."""
        if vec1 is None or vec2 is None:
            return float('inf')
        return np.linalg.norm(vec1 - vec2)
    
    def _distance_to_similarity_score(self, distance):
        """Convert distance to a 0-100% similarity score."""
        # For unit-normalized vectors, Euclidean distance is in range [0, 2]
        # Distance 0 = identical, Distance 2 = completely opposite
        max_distance = 2.0  # Maximum possible distance for unit vectors
        
        # Convert to similarity percentage (exponential decay for better discrimination)
        # This gives better separation between similar and dissimilar objects
        similarity = 100 * np.exp(-2.0 * distance)  # Exponential decay
        
        return max(0, min(100, similarity))
    
    def _find_similar_objects(self, query_filename, top_n=5):
        """Find the most similar objects to the query."""
        query_vector = self._extract_feature_vector(query_filename)
        if query_vector is None:
            return []
        
        distances = []
        query_base = os.path.splitext(query_filename)[0]
        
        # Compare with all objects in the features dictionary
        for category, files in self.features_dict.items():
            for file_key, file_data in files.items():
                if file_key == query_base:
                    continue  # Skip the query object itself
                
                # Reconstruct filename and extract vector
                candidate_filename = file_key + '.obj'
                candidate_vector = self._extract_feature_vector(candidate_filename)
                
                if candidate_vector is not None:
                    distance = self._compute_distance(query_vector, candidate_vector)
                    distances.append((distance, category, candidate_filename))
        
        # Sort by distance and return top N
        distances.sort(key=lambda x: x[0])
        
        # Debug: Print distance range for analysis
        if distances:
            min_dist = distances[0][0]
            max_dist = distances[-1][0] if len(distances) > 1 else min_dist
            print(f"Distance range for {query_filename}: {min_dist:.3f} to {max_dist:.3f}")
        
        return distances[:top_n]
    
    def _load_similar_objects_to_gallery(self, query_filename):
        """Load the 5 most similar objects to the gallery in order: closest on left, 2nd closest, etc."""
        similar_objects = self._find_similar_objects(query_filename, 5)
        
        if not similar_objects:
            # No similar objects found, clear gallery
            self._clear_gallery()
            return
        
        # Load similar objects into gallery in order
        for i, (distance, category, filename) in enumerate(similar_objects):
            if i >= len(self.gallery_plotters):
                break
            
            try:
                # Construct full path
                obj_path = os.path.join(self.parent_folder, category, filename)
                
                if os.path.exists(obj_path):
                    self.gallery_plotters[i].clear()
                    mesh = load(obj_path)
                    mesh.lighting('default').linecolor('black').linewidth(1)
                    self.gallery_plotters[i].show(mesh, resetcam=True)
                    self._update_gallery_item_features(i, filename)
                    
                    # Update distance score display
                    if i < len(self.gallery_distance_labels):
                        similarity_score = self._distance_to_similarity_score(distance)
                        self.gallery_distance_labels[i].setText(f"Similarity: {similarity_score:.1f}% (d={distance:.3f})")
                else:
                    # Object file not found, clear the gallery slot
                    self.gallery_plotters[i].clear()
                    self.gallery_plotters[i].render()
                    if i < len(self.gallery_metrics_labels):
                        self.gallery_metrics_labels[i].setText("File not found")
                        self.gallery_histogram_widgets[i].set_histograms([])
                        if i < len(self.gallery_distance_labels):
                            self.gallery_distance_labels[i].setText("Similarity: --")
            except Exception as e:
                print(f"Error loading similar object {i}: {e}")
                self.gallery_plotters[i].clear()
                self.gallery_plotters[i].render()
                if i < len(self.gallery_metrics_labels):
                    self.gallery_metrics_labels[i].setText("Load error")
                    self.gallery_histogram_widgets[i].set_histograms([])
                    if i < len(self.gallery_distance_labels):
                        self.gallery_distance_labels[i].setText("Similarity: --")
        
        # Keep gallery viewers square
        for w in self.gallery_widgets:
            try:
                w.setFixedHeight(max(0, w.width()))
            except Exception:
                pass
    
    def _refresh_gallery_smart(self):
        """Smart gallery refresh: show similar objects if object selected, otherwise clear gallery."""
        if hasattr(self, 'loaded_shapes') and self.loaded_shapes:
            # Get the current object filename
            current_shape = self.loaded_shapes[-1]
            filename = os.path.basename(current_shape.file_path)
            self._load_similar_objects_to_gallery(filename)
        else:
            # No object selected, clear gallery
            self._clear_gallery()
    
    def _clear_gallery(self):
        """Clear all gallery viewers."""
        for p in self.gallery_plotters:
            p.clear()
            p.render()
        for i in range(len(self.gallery_metrics_labels)):
            self.gallery_metrics_labels[i].setText("Select an object to see similar items")
            self.gallery_histogram_widgets[i].set_histograms([])
            if i < len(self.gallery_distance_labels):
                self.gallery_distance_labels[i].setText("Similarity: --")

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
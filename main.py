import shutil
import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox, QPushButton
from vedo import Plotter, load, Box
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pymeshlab as ml
import time
import gc

TARGET_VERTICES = 5000
SHAPEDATA_PARENT = os.path.abspath('ShapeDatabase_INFOMR-master/')
TEMP_REMESH_DIR = os.path.abspath('temp_remesh')


class Shape:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vertices = None
        self.faces = None
        self.mesh = None
        self.temp_copy_path = None  # path to remeshed temp copy

        # ensure temp folder exists
        os.makedirs(TEMP_REMESH_DIR, exist_ok=True)

    def load(self):
        """Load mesh using vedo and parse info"""
        self.mesh = load(self.file_path)
        self.vertices, self.faces, _, _ = Shape.parse_obj_info(self.file_path)

    def resample(self):
        """Remesh the object to ~TARGET_VERTICES vertices and save in temp folder"""
        if not os.path.exists(self.file_path):
            print("File not found:", self.file_path)
            return

        # Create temp remesh path in TEMP_REMESH_DIR
        name, ext = os.path.splitext(os.path.basename(self.file_path))
        self.temp_copy_path = os.path.join(TEMP_REMESH_DIR, f"{name}_remesh{ext}")

        ms = ml.MeshSet()
        ms.load_new_mesh(self.file_path)

        # Clean the mesh
        ms.apply_filter("meshing_remove_duplicate_faces")
        ms.apply_filter("meshing_remove_duplicate_vertices")
        ms.apply_filter("meshing_remove_unreferenced_vertices")
        ms.apply_filter("meshing_remove_null_faces")
        ms.apply_filter("meshing_repair_non_manifold_edges")
        ms.apply_filter("meshing_repair_non_manifold_vertices")

        # Remeshing logic
        def decrease_vertices():
            try:
                estimated_faces = int(ms.current_mesh().face_number() * (TARGET_VERTICES / ms.current_mesh().vertex_number()))
                ms.apply_filter(
                    "meshing_decimation_quadric_edge_collapse",
                    targetfacenum=estimated_faces,
                    qualitythr=0.5,
                    preservenormal=True,
                    preserveboundary=True,
                    preservetopology=True,
                    optimalplacement=True,
                    autoclean=True
                )
                return True
            except ml.PyMeshLabException:
                return False

        def increase_vertices():
            try:
                ms.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
                return True
            except ml.PyMeshLabException:
                return False

        counter = 0
        while (ms.current_mesh().vertex_number() < TARGET_VERTICES or
               ms.current_mesh().vertex_number() > TARGET_VERTICES) and counter < 20:
            counter += 1
            if ms.current_mesh().vertex_number() < TARGET_VERTICES:
                if not increase_vertices(): break
            elif ms.current_mesh().vertex_number() > TARGET_VERTICES:
                if not decrease_vertices(): break

        # Save remeshed copy to temp folder
        ms.save_current_mesh(self.temp_copy_path)

        # Reload remeshed mesh into vedo
        self.mesh = load(self.temp_copy_path)
        self.vertices, self.faces, _, _ = Shape.parse_obj_info(self.temp_copy_path)

    @staticmethod
    def parse_obj_info(filepath):
        # same as before
        vertices, faces, face_type, bbox = [], [], "N/A", "N/A"
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) == 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    faces.append(parts[1:])
        face_types = set()
        for face in faces:
            if len(face) == 3: face_types.add("triangles")
            elif len(face) == 4: face_types.add("quads")
            else: face_types.add("other")
        face_type = " and ".join(sorted(face_types)) if face_types else "unknown"
        if vertices:
            xs, ys, zs = zip(*vertices)
            bbox = f"X:[{min(xs):.2f},{max(xs):.2f}] Y:[{min(ys):.2f},{max(ys):.2f}] Z:[{min(zs):.2f},{max(zs):.2f}]"
        return len(vertices), len(faces), face_type, bbox

    @staticmethod
    def cleanup_temp_folder():
        """Delete the temporary remesh folder when closing."""
        if os.path.exists(TEMP_REMESH_DIR):
            shutil.rmtree(TEMP_REMESH_DIR)
            print(f"[Shape] Deleted temporary remesh folder: {TEMP_REMESH_DIR}")



class FeatureExtractor:
    def __init__(self, method="default"):
        pass

    def extract(self, shape):
        pass


class FeatureDatabase:
    def __init__(self, db_path):
        pass

    def add_feature(self, shape_id, feature_vector):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def get_all_features(self):
        pass


class QueryEngine:
    def __init__(self, feature_extractor, database):
        pass

    def compute_similarity(self, feature1, feature2):
        pass

    def query(self, query_shape, top_k=10):
        pass


class CBSRApp(QWidget):
    def __init__(self, parent_folder):
        super().__init__()
        self.parent_folder = parent_folder
        self.loaded_shapes = []
        self.current_mesh_actor = None

        self.setWindowTitle("CBSR Debug GUI")
        self.resize(1200, 600)
        layout = QHBoxLayout(self)
        self.setLayout(layout)

        # -----------------
        # Left panel: file explorer
        # -----------------
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel)

        # Main folders
        left_panel.addWidget(QLabel("Main Folder"))
        self.main_folder_combo = QComboBox()
        self.main_folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
        self.main_folder_combo.addItems(self.main_folders)
        self.main_folder_combo.currentTextChanged.connect(self.on_main_folder_changed)
        left_panel.addWidget(self.main_folder_combo)

        # Categories list
        self.category_list = QListWidget()
        self.category_list.itemClicked.connect(self.on_category_selected)
        left_panel.addWidget(QLabel("Categories"))
        left_panel.addWidget(self.category_list)

        # Files list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_panel.addWidget(QLabel("Files"))
        left_panel.addWidget(self.file_list)

        # -----------------
        # Right panel: 3D viewer + controls
        # -----------------
        viewer_panel = QVBoxLayout()
        layout.addLayout(viewer_panel)
        viewer_panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        viewer_panel.addWidget(self.viewer_widget)

        self.info_label = QLabel("Select a file to see info.")
        viewer_panel.addWidget(self.info_label)

        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        viewer_panel.addWidget(self.bbox_toggle)

        # Remesh button
        self.remesh_button = QPushButton("Remesh")
        self.remesh_button.clicked.connect(self.on_remesh_clicked)
        viewer_panel.addWidget(self.remesh_button)

        # Initialize first folder
        if self.main_folders:
            self.on_main_folder_changed(self.main_folders[0])

    # -----------------
    # GUI Methods
    # -----------------
    def on_main_folder_changed(self, folder_name):
        self.category_list.clear()
        self.file_list.clear()
        self.current_main_folder = os.path.join(self.parent_folder, folder_name)
        categories = [d for d in os.listdir(self.current_main_folder) if os.path.isdir(os.path.join(self.current_main_folder, d))]
        self.category_list.addItems(categories)

    def on_category_selected(self, item):
        self.file_list.clear()
        category_path = os.path.join(self.current_main_folder, item.text())
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)

    def on_file_selected(self, item):
        selected_category = self.category_list.currentItem().text()
        full_path = os.path.join(self.current_main_folder, selected_category, item.text())

        shape = Shape(full_path)
        shape.load()
        self.loaded_shapes.append(shape)

        self.plotter.clear()
        shape.mesh.wireframe(True)
        self.plotter.show(shape.mesh, resetcam=True)
        self.current_mesh_actor = shape.mesh

        info = f"File: {item.text()}\nVertices: {shape.vertices}\nFaces: {shape.faces}"
        self.info_label.setText(info)
        self.bbox_toggle.setChecked(False)

    def on_bbox_toggle(self, state):
        if self.current_mesh_actor:
            if state:
                self.bbox_actor = Box(self.current_mesh_actor.bounds()).wireframe().c('red')
                self.plotter.add(self.bbox_actor)
            else:
                if hasattr(self, 'bbox_actor') and self.bbox_actor:
                    self.plotter.remove(self.bbox_actor)
                    self.bbox_actor = None
            self.plotter.render()

    def on_remesh_clicked(self):
        if not self.loaded_shapes:
            self.info_label.setText("No shape loaded to remesh!")
            return

        shape = self.loaded_shapes[-1]  # last loaded
        self.info_label.setText("Remeshing, please wait...")
        QApplication.processEvents()  # update label immediately

        shape.resample()  # runs PyMeshLab remesh

        # Update viewer
        self.plotter.clear()
        shape.mesh.wireframe(True)
        self.plotter.show(shape.mesh, resetcam=True)
        self.current_mesh_actor = shape.mesh

        self.info_label.setText(f"Remeshed: {os.path.basename(shape.file_path)}\nVertices: {shape.vertices}\nFaces: {shape.faces}")

    def closeEvent(self, event):
        Shape.cleanup_temp_folder()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CBSRApp(SHAPEDATA_PARENT)
    window.show()
    sys.exit(app.exec())

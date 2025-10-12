import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox
from vedo import Plotter, load, Box
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from mesh_utils import parse_obj_info

SHAPEDATA_PARENT = os.path.abspath('ShapeDatabase_INFOMR-master')

class FileExplorer(QWidget):
    """Simple 3D file explorer and viewer."""
    
    def __init__(self, parent_folder: str):
        super().__init__()
        self.parent_folder = parent_folder
        self.current_main_folder = None
        self.current_mesh = None
        self.bbox_actor = None

        # Setup UI
        main_layout = QHBoxLayout(self)
        
        # Left panel: file browser
        left_panel = QVBoxLayout()
        self._setup_file_browser(left_panel)
        main_layout.addLayout(left_panel)

        # Right panel: 3D viewer
        viewer_panel = QVBoxLayout()
        self._setup_3d_viewer(viewer_panel)
        main_layout.addLayout(viewer_panel)

        # Initialize with first folder
        if self.main_folders:
            self.on_main_folder_changed(self.main_folders[0])

    def _setup_file_browser(self, panel: QVBoxLayout) -> None:
        """Setup file browser components."""
        panel.addWidget(QLabel("Main Folder"))
        self.main_folder_combo = QComboBox()
        self.main_folders = [d for d in os.listdir(self.parent_folder) 
                           if os.path.isdir(os.path.join(self.parent_folder, d))]
        self.main_folder_combo.addItems(self.main_folders)
        self.main_folder_combo.currentTextChanged.connect(self.on_main_folder_changed)
        panel.addWidget(self.main_folder_combo)

        self.category_list, category_panel = self._create_list_panel("Categories", self.on_category_selected)
        panel.addLayout(category_panel)

        self.file_list, file_panel = self._create_list_panel("Files", self.on_file_selected)
        panel.addLayout(file_panel)

    def _setup_3d_viewer(self, panel: QVBoxLayout) -> None:
        """Setup 3D viewer components."""
        panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        panel.addWidget(self.viewer_widget)
        
        self.info_label = QLabel("Select a file to see info.")
        panel.addWidget(self.info_label)
        
        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        panel.addWidget(self.bbox_toggle)

    def create_list_panel(self, label_text, click_handler):
        list_widget = QListWidget()
        list_widget.itemClicked.connect(click_handler)
        panel = QVBoxLayout()
        panel.addWidget(QLabel(label_text))
        panel.addWidget(list_widget)
        return list_widget, panel

    def on_main_folder_changed(self, folder_name):
        self.current_main_folder = os.path.join(self.parent_folder, folder_name)
        self.category_list.clear()
        categories = [d for d in os.listdir(self.current_main_folder) if os.path.isdir(os.path.join(self.current_main_folder, d))]
        self.category_list.addItems(categories)
        self.file_list.clear()

    def on_category_selected(self, item):
        self.file_list.clear()
        category_path = os.path.join(self.current_main_folder, item.text())
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)

    def on_file_selected(self, item):
        selected_category = self.category_list.currentItem().text()
        full_path = os.path.join(self.current_main_folder, selected_category, item.text())
        if os.path.exists(full_path):
            self.plotter.clear()
            self.current_mesh = load(full_path)
            self.plotter.show(self.current_mesh, resetcam=True)
            v_count, f_count, f_type, bbox = parse_obj_info(full_path)
            info = f"Vertices: {v_count}\nFaces: {f_count}\nFace type: {f_type}\nBounding box: {bbox}"
            self.info_label.setText(info)
            self.bbox_toggle.setChecked(False)
            if self.bbox_actor:
                self.plotter.remove(self.bbox_actor)
                self.bbox_actor = None

    def on_bbox_toggle(self, state):
        if self.current_mesh:
            if state:
                self.bbox_actor = Box(self.current_mesh.bounds()).wireframe().c('red')
                self.plotter.add(self.bbox_actor)
            else:
                if self.bbox_actor:
                    self.plotter.remove(self.bbox_actor)
                    self.bbox_actor = None
            self.plotter.render()

def parse_obj_info(filepath):
    vertices = []
    faces = []
    face_type = "N/A"
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
        count = len(face)
        if count == 3:
            face_types.add("triangles")
        elif count == 4:
            face_types.add("quads")
        else:
            face_types.add("other")
    face_type = " and ".join(sorted(face_types)) if face_types else "unknown"
    bbox = "N/A"
    if vertices:
        xs, ys, zs = zip(*vertices)
        bbox = f"X:[{min(xs):.2f}, {max(xs):.2f}] Y:[{min(ys):.2f}, {max(ys):.2f}] Z:[{min(zs):.2f}, {max(zs):.2f}]"
    return len(vertices), len(faces), face_type, bbox

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("3D Viewer")
    window.resize(1200, 600)
    layout = QHBoxLayout(window)
    layout.addWidget(FileExplorer(SHAPEDATA_PARENT))
    window.show()
    sys.exit(app.exec())
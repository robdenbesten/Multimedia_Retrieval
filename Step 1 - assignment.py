import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox
from vedo import Plotter, load, Box, Axes
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Define the parent directory containing main folders
SHAPEDATA_PARENT = os.path.abspath('ShapeDatabase_INFOMR-master')

class FileExplorer(QWidget):
    def __init__(self, parent_folder):
        super().__init__()
        self.parent_folder = parent_folder
        self.current_main_folder = None
        self.current_mesh = None
        self.bbox_actor = None

        main_layout = QHBoxLayout(self)

        # Main folder selection box
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Main Folder"))
        self.main_folder_combo = QComboBox()
        self.main_folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
        self.main_folder_combo.addItems(self.main_folders)
        self.main_folder_combo.currentTextChanged.connect(self.on_main_folder_changed)
        left_panel.addWidget(self.main_folder_combo)

        # Categories panel
        self.category_list, category_panel = self.create_list_panel("Categories", self.on_category_selected)
        left_panel.addLayout(category_panel)

        # Files panel
        self.file_list, file_panel = self.create_list_panel("Files", self.on_file_selected)
        left_panel.addLayout(file_panel)

        main_layout.addLayout(left_panel)

        # 3D viewer panel
        viewer_panel = QVBoxLayout()
        viewer_panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        viewer_panel.addWidget(self.viewer_widget)
        self.info_label = QLabel("Select a file to see info.")
        viewer_panel.addWidget(self.info_label)
        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        viewer_panel.addWidget(self.bbox_toggle)
        main_layout.addLayout(viewer_panel)

        #initialize axes toggle
        self.axes_actor = None
        self.axes_toggle = QCheckBox("Show Axes")
        self.axes_toggle.stateChanged.connect(self.on_axes_toggle)
        viewer_panel.addWidget(self.axes_toggle)

        # Initialize with the first main folder
        if self.main_folders:
            self.on_main_folder_changed(self.main_folders[0])

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

    def on_axes_toggle(self, state):
        if self.current_mesh:
            if state:
                self.add_axes()
            else:
                if self.axes_actor:
                    self.plotter.remove(self.axes_actor)
                    self.axes_actor = None
            self.plotter.render()

    def add_axes(self):
        if self.axes_actor:
            self.plotter.remove(self.axes_actor)
            self.axes_actor = None
        if self.current_mesh is not None:
            self.axes_actor = Axes(self.current_mesh, xtitle="X", ytitle="Y", ztitle="Z", c='k',
                                   xlabel_size=0.04, ylabel_size=0.04, zlabel_size=0.04)
            self.plotter.add(self.axes_actor)

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
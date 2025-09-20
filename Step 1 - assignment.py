import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel
from vedo import Plotter, load
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

SHAPEDATA_ROOT = os.path.abspath('ShapeDatabase_INFOMR-master/ShapeDatabase_INFOMR-master')

def parse_obj_info(filepath):
    vertices = []
    faces = []
    face_type = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) == 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                faces.append(parts[1:])
    if faces:
        face_type = f"{len(faces[0])}-gon"
    else:
        face_type = "N/A"
    if vertices:
        xs, ys, zs = zip(*vertices)
        bbox = f"X:[{min(xs):.2f}, {max(xs):.2f}] Y:[{min(ys):.2f}, {max(ys):.2f}] Z:[{min(zs):.2f}, {max(zs):.2f}]"
    else:
        bbox = "N/A"
    return len(vertices), len(faces), face_type, bbox

class FileExplorer(QWidget):
    def __init__(self, root_folder):
        super().__init__()
        self.root_folder = root_folder

        main_layout = QHBoxLayout(self)

        # Category list
        self.category_list = QListWidget()
        categories = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
        self.category_list.addItems(categories)
        self.category_list.itemClicked.connect(self.on_category_selected)

        category_panel = QVBoxLayout()
        category_panel.addWidget(QLabel("Categories"))
        category_panel.addWidget(self.category_list)
        main_layout.addLayout(category_panel)

        # File list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)

        file_panel = QVBoxLayout()
        file_panel.addWidget(QLabel("Files"))
        file_panel.addWidget(self.file_list)
        main_layout.addLayout(file_panel)

        # 3D Viewer and Info
        viewer_panel = QVBoxLayout()
        viewer_panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        viewer_panel.addWidget(self.viewer_widget)

        self.info_label = QLabel("Select a file to see info.")
        viewer_panel.addWidget(self.info_label)
        main_layout.addLayout(viewer_panel)

    def on_category_selected(self, item):
        self.file_list.clear()
        category_path = os.path.join(self.root_folder, item.text())
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)

    def on_file_selected(self, item):
        selected_category = self.category_list.currentItem().text()
        file_name = item.text()
        full_path = os.path.join(self.root_folder, selected_category, file_name)
        if os.path.exists(full_path):
            self.plotter.clear()
            mesh = load(full_path)
            self.plotter.show(mesh, resetcam=True)
            v_count, f_count, f_type, bbox = parse_obj_info(full_path)
            info = (
                f"Vertices: {v_count}\n"
                f"Faces: {f_count}\n"
                f"Face type: {f_type}\n"
                f"Bounding box: {bbox}"
            )
            self.info_label.setText(info)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Step 3: File Explorer with 3D Viewer")
    window.resize(1200, 600)
    layout = QHBoxLayout(window)
    layout.addWidget(FileExplorer(SHAPEDATA_ROOT))
    window.show()
    sys.exit(app.exec())

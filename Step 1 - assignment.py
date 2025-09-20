import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox
from vedo import Plotter, load, Box
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
        self.current_mesh = None
        self.bbox_actor = None

        main_layout = QHBoxLayout(self)

        # Left Panel: Categories & Files
        selection_panel = QVBoxLayout()

        category_layout = QVBoxLayout()
        category_layout.addWidget(QLabel("Categories"))
        self.category_list = QListWidget()
        categories = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
        self.category_list.addItems(categories)
        self.category_list.itemClicked.connect(self.on_category_selected)
        category_layout.addWidget(self.category_list)
        selection_panel.addLayout(category_layout)

        file_layout = QVBoxLayout()
        file_layout.addWidget(QLabel("Files"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)
        selection_panel.addLayout(file_layout)

        main_layout.addLayout(selection_panel)

        # Right Panel: 3D Viewer & Info
        viewer_panel = QVBoxLayout()
        viewer_panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        viewer_panel.addWidget(self.viewer_widget)

        self.info_label = QLabel("Select a file to see info.")
        viewer_panel.addWidget(self.info_label)

        # Bounding box checkbox
        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        viewer_panel.addWidget(self.bbox_toggle)

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
            self.current_mesh = load(full_path)
            self.plotter.show(self.current_mesh, resetcam=True)

            v_count, f_count, f_type, bbox = parse_obj_info(full_path)
            info = (
                f"Vertices: {v_count}\n"
                f"Faces: {f_count}\n"
                f"Face type: {f_type}\n"
                f"Bounding box: {bbox}"
            )
            self.info_label.setText(info)

            # Reset the checkbox and remove the old bounding box
            self.bbox_toggle.setChecked(False)
            if self.bbox_actor:
                self.plotter.remove(self.bbox_actor)
                self.bbox_actor = None

    def on_bbox_toggle(self, state):
        if self.current_mesh:
            if state:  # Checkbox is checked
                # Get the bounds of the current mesh
                bounds = self.current_mesh.bounds()
                # Create a vedo Box actor from the mesh bounds
                self.bbox_actor = Box(bounds)
                self.bbox_actor.wireframe().c('red')
                self.plotter.add(self.bbox_actor)
            else:  # Checkbox is unchecked
                if self.bbox_actor:
                    self.plotter.remove(self.bbox_actor)
                    self.bbox_actor = None
            self.plotter.render()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("3D Viewer")
    window.resize(1200, 600)
    layout = QHBoxLayout(window)
    layout.addWidget(FileExplorer(SHAPEDATA_ROOT))
    window.show()
    sys.exit(app.exec())
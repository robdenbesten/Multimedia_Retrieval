import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox
from vedo import Plotter, load, Box
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Define the directory
SHAPEDATA_ROOT = os.path.abspath('ShapeDatabase_INFOMR-master/ShapeDatabase_INFOMR-master')


# This function get information like vertex count, face count, and bounding box.
def parse_obj_info(filepath):
    vertices = []
    faces = []
    face_type = "N/A"

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # 'v' for vertex
                parts = line.strip().split()
                if len(parts) == 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # 'f' for face
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
        # Calculate the bounding box by finding the min and max coordinates for each axis
        xs, ys, zs = zip(*vertices)
        bbox = f"X:[{min(xs):.2f}, {max(xs):.2f}] Y:[{min(ys):.2f}, {max(ys):.2f}] Z:[{min(zs):.2f}, {max(zs):.2f}]"

    return len(vertices), len(faces), face_type, bbox


# The main application window for the 3D viewer.
class FileExplorer(QWidget):
    def __init__(self, root_folder):
        super().__init__()
        self.root_folder = root_folder
        self.current_mesh = None  # Holds the currently loaded 3D model
        self.bbox_actor = None  # Holds the bounding box

        # Set up the layout for the entire window
        main_layout = QHBoxLayout(self)

        # This function creates a panel
        def create_list_panel(label_text, click_handler):
            list_widget = QListWidget()
            list_widget.itemClicked.connect(click_handler)
            panel = QVBoxLayout()
            panel.addWidget(QLabel(label_text))
            panel.addWidget(list_widget)
            return list_widget, panel

        # Create the categories panel and add it to the main layout
        self.category_list, category_panel = create_list_panel("Categories", self.on_category_selected)
        categories = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
        self.category_list.addItems(categories)
        main_layout.addLayout(category_panel)

        # Create the files panel and add it to the main layout
        self.file_list, file_panel = create_list_panel("Files", self.on_file_selected)
        main_layout.addLayout(file_panel)

        # Create the 3D viewer panel
        viewer_panel = QVBoxLayout()
        viewer_panel.addWidget(QLabel("3D Viewer"))

        # This widget is where the 3D model will be drawn
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        # The vedo plotter for 3D rendering
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        viewer_panel.addWidget(self.viewer_widget)

        # This displays information about the 3D model
        self.info_label = QLabel("Select a file to see info.")
        viewer_panel.addWidget(self.info_label)

        # This checkbox will show or hide the bounding box
        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        viewer_panel.addWidget(self.bbox_toggle)

        # Add the view panel to the main layout
        main_layout.addLayout(viewer_panel)

    # This function is called when a category is clicked
    def on_category_selected(self, item):
        self.file_list.clear()  # Clear the previous files
        category_path = os.path.join(self.root_folder, item.text())
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)  # Populate the file list with .obj files

    # This function is called when a file is clicked
    def on_file_selected(self, item):
        # Get the full path to the selected OBJ file
        selected_category = self.category_list.currentItem().text()
        full_path = os.path.join(self.root_folder, selected_category, item.text())

        if os.path.exists(full_path):
            self.plotter.clear()  # Clear the previous model from the viewer
            self.current_mesh = load(full_path)  # Load the new 3D model
            self.plotter.show(self.current_mesh, resetcam=True)  # Display the model and reset the camera

            # Parse the OBJ file for information and update the info label
            v_count, f_count, f_type, bbox = parse_obj_info(full_path)
            info = f"Vertices: {v_count}\nFaces: {f_count}\nFace type: {f_type}\nBounding box: {bbox}"
            self.info_label.setText(info)

            # Reset the bounding box toggle and actor for the new model
            self.bbox_toggle.setChecked(False)
            if self.bbox_actor:
                self.plotter.remove(self.bbox_actor)
                self.bbox_actor = None

    # This function is called when the bounding box checkbox is toggled
    def on_bbox_toggle(self, state):
        if self.current_mesh:
            if state:  # If the box is checked
                # Create a Box actor that outlines the loaded mesh
                self.bbox_actor = Box(self.current_mesh.bounds()).wireframe().c('red')
                self.plotter.add(self.bbox_actor)  # Add the box to the viewer
            else:  # If the box is unchecked
                if self.bbox_actor:
                    self.plotter.remove(self.bbox_actor)  # Remove the box from the viewer
                    self.bbox_actor = None
            self.plotter.render()  # Update the display to show the changes


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("3D Viewer")
    window.resize(1200, 600)
    layout = QHBoxLayout(window)
    layout.addWidget(FileExplorer(SHAPEDATA_ROOT))
    window.show()
    sys.exit(app.exec())
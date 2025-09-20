import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *


class ObjLoader:
    """
    A simple class to load and parse a Wavefront .obj file.
    It supports vertices (v) and triangular faces (f).
    """

    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.face_indices = []
        self.vertex_count = 0
        self.face_count = 0
        try:
            with open(filename, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == 'v':
                        # Parse vertex line: 'v x y z'
                        self.vertices.append(list(map(float, parts[1:4])))
                        self.vertex_count += 1
                    elif parts[0] == 'f':
                        # Parse face line: 'f v1 v2 v3'
                        # We only handle triangular faces for simplicity.
                        indices = [int(p.split('/')[0]) for p in parts[1:]]
                        self.faces.append(indices)
                        self.face_count += 1
                        # Create a flat list of vertex indices for OpenGL
                        self.face_indices.extend(indices)
            print(f"Loaded {filename}: {self.vertex_count} vertices, {self.face_count} faces.")
        except FileNotFoundError:
            print(f"Error: File not found at {filename}")


class GLWidget(QOpenGLWidget):
    """
    This is the main OpenGL widget responsible for rendering the 3D model.
    It handles initialization, resizing, painting, and mouse interactions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_data = None
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.last_mouse_pos = None

        # Set up a timer to continuously update the GL scene for smooth rotation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS

    def set_model_data(self, model_data):
        """Sets the model to be rendered and resets the rotation."""
        self.model_data = model_data
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.update()  # Request a redraw

    def initializeGL(self):
        """
        Called once to set up the OpenGL rendering context.
        """
        glClearColor(0.2, 0.2, 0.2, 1.0)  # Set background color to dark gray
        glEnable(GL_DEPTH_TEST)  # Enable depth testing for proper 3D rendering
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Set up a simple light source
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])

    def resizeGL(self, w, h):
        """
        Called whenever the widget is resized.
        """
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (w / h), 0.1, 50.0)  # Set a perspective projection
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)  # Position the camera

    def paintGL(self):
        """
        The main rendering function, called for every frame.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera position
        gluLookAt(0, 0, 5,  # Eye position
                  0, 0, 0,  # Look at point
                  0, 1, 0)  # Up vector

        # Apply rotations based on user mouse input
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        # Draw the model if loaded
        if self.model_data:
            glColor3f(0.8, 0.8, 0.8)  # Set a light gray color for the model

            # This is the "immediate mode" drawing, simple but not the most efficient for large models.
            # It's good for demonstration purposes.
            glBegin(GL_TRIANGLES)
            for face in self.model_data.faces:
                # Get vertex coordinates from the loaded data
                for vertex_index in face:
                    v = self.model_data.vertices[vertex_index - 1]
                    glVertex3f(v[0], v[1], v[2])
            glEnd()

            # Optional: Draw the wireframe to make the mesh structure visible
            glColor3f(0.5, 0.5, 0.5)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
            glBegin(GL_TRIANGLES)
            for face in self.model_data.faces:
                for vertex_index in face:
                    v = self.model_data.vertices[vertex_index - 1]
                    glVertex3f(v[0], v[1], v[2])
            glEnd()
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def mousePressEvent(self, event):
        """Store the initial mouse position on click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """
        Rotate the model based on mouse movement.
        """
        if event.buttons() == Qt.MouseButton.LeftButton:
            # Calculate the change in position
            delta_x = event.x() - self.last_mouse_pos.x()
            delta_y = event.y() - self.last_mouse_pos.y()

            # Adjust rotation angles. The factor (0.5) controls the rotation speed.
            self.rotation_x += delta_y * 0.5
            self.rotation_y += delta_x * 0.5

            # Update the last mouse position
            self.last_mouse_pos = event.pos()

            # Request a redraw
            self.update()


class MainWindow(QMainWindow):
    """
    The main application window, holding all UI components.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6/PyOpenGL 3D Model Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for information and controls
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.open_button = QPushButton("Open .obj File")
        self.open_button.clicked.connect(self.open_file_dialog)

        self.info_label = QLabel("Model Info: No file loaded")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info_label.setFixedWidth(250)

        info_layout.addWidget(self.open_button)
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()

        # Right panel for the 3D viewer
        self.gl_widget = GLWidget()

        # Add widgets to the main layout
        main_layout.addWidget(info_panel)
        main_layout.addWidget(self.gl_widget, 1)  # Set stretch factor to 1 for the viewer

    def open_file_dialog(self):
        """
        Opens a file dialog to select and load an .obj file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open .obj File", "", "OBJ Files (*.obj)")
        if file_path:
            try:
                # Load the model and update the UI
                model_loader = ObjLoader(file_path)
                self.gl_widget.set_model_data(model_loader)

                # Update the info label
                info_text = (
                    f"<b>File:</b> {file_path.split('/')[-1]}\n"
                    f"<b>Vertices:</b> {model_loader.vertex_count}\n"
                    f"<b>Faces:</b> {model_loader.face_count}"
                )
                self.info_label.setText(info_text)
            except Exception as e:
                self.info_label.setText(f"Error loading file: {e}")
                self.gl_widget.set_model_data(None)


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    # The application needs a reference to a QApplication object
    # for QOpenGLWidget to work correctly.
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

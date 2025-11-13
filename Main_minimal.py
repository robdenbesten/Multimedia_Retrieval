"""
MINIMAL MAIN VIEWER - NO NORMALIZATION
This version works without pymeshlab/RemeshAndNormalise
Only displays meshes and features from pre-computed database
"""

import sys
import os
import shutil

# Set Qt/OpenGL environment variables BEFORE any Qt/VTK imports
os.environ.setdefault('QT_API', 'pyqt6')
os.environ.setdefault('QT_OPENGL', 'desktop')
os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '0')
os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '0')
os.environ.setdefault('VEDO_DEFAULT_BACKEND', 'vtk')
os.environ.setdefault('VTK_SILENCE_GET_VOID_POINTER_WARNINGS', '1')

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QComboBox, QListWidget, QCheckBox, QGridLayout,
                             QScrollArea, QTextEdit, QMessageBox)

# Set Qt attributes before any QApplication instance
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VTK Qt integration not available: {e}")
    VTK_AVAILABLE = False

try:
    from vedo import Plotter, load, Axes
    VEDO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: vedo not available: {e}")
    VEDO_AVAILABLE = False

# Try to import Querying for search functionality
try:
    from Querying import ShapeSearcher, MANUAL_WEIGHTS
    QUERYING_AVAILABLE = True
except Exception as e:
    print(f"Warning: Querying not available: {e}")
    QUERYING_AVAILABLE = False
    ShapeSearcher = None
    MANUAL_WEIGHTS = None

DATABASE_LOCATION = r'Normalised-objects'
NORMALISED_DATABASE_LOCATION = r'Normalised-objects'
FEATURE_CSV = 'Feature-matrix/all_features.csv'


class MinimalMeshViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.database_location = DATABASE_LOCATION
        self.searcher = None
        self.result_plotters = []
        self.result_labels = []
        self.current_features = None
        self.origin_axes = None

        self.init_searcher()
        self.init_ui()

    def init_searcher(self):
        """Initialize the ShapeSearcher."""
        if not QUERYING_AVAILABLE or not ShapeSearcher:
            print("Querying module not available - search disabled")
            return

        try:
            self.searcher = ShapeSearcher(
                feature_csv_path=FEATURE_CSV,
                weights=MANUAL_WEIGHTS,
                weighting_method='feature'
            )
            print("Search functionality enabled")
        except Exception as e:
            print(f"Failed to initialize searcher: {e}")

    def init_ui(self):
        self.setWindowTitle('Minimal Mesh Viewer (No Normalization)')
        self.setGeometry(100, 100, 1600, 900)

        if not VTK_AVAILABLE or not VEDO_AVAILABLE:
            QMessageBox.critical(
                self,
                "Missing Dependencies",
                f"VTK/vedo not available:\nVTK: {VTK_AVAILABLE}\nvedo: {VEDO_AVAILABLE}\n\n"
                "Please install: pip install vtk vedo"
            )
            self.close()
            return

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)

        # Category dropdown
        control_layout.addWidget(QLabel('Select Category:'))
        self.category_dropdown = QComboBox()
        self.category_dropdown.currentTextChanged.connect(self.load_category_objects)
        control_layout.addWidget(self.category_dropdown)

        # Object list
        control_layout.addWidget(QLabel('Objects:'))
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.load_mesh_from_list)
        control_layout.addWidget(self.object_list)

        # Status label
        self.status_label = QLabel('Select a category and object')
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        # Show axes checkbox
        self.show_axes_checkbox = QCheckBox('Show Axes')
        self.show_axes_checkbox.setChecked(True)
        self.show_axes_checkbox.stateChanged.connect(self.toggle_axes_visibility)
        control_layout.addWidget(self.show_axes_checkbox)

        # Search controls
        control_layout.addWidget(QLabel('Search Metric:'))
        self.metric_dropdown = QComboBox()
        if self.searcher:
            self.metric_dropdown.addItems(self.searcher.metrics)
        control_layout.addWidget(self.metric_dropdown)

        self.search_button = QPushButton('Find Similar')
        self.search_button.clicked.connect(self.find_similar_shapes)
        self.search_button.setEnabled(False)
        control_layout.addWidget(self.search_button)

        # Middle panel - viewer
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)

        self.mesh_info_label = QLabel('Mesh Info: Load a mesh')
        self.mesh_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mesh_info_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        middle_layout.addWidget(self.mesh_info_label)

        # Main VTK Widget
        try:
            self.vtk_widget = QVTKRenderWindowInteractor(middle_panel)
            self.plotter = Plotter(qt_widget=self.vtk_widget, N=1, bg='white')
            middle_layout.addWidget(self.vtk_widget, stretch=3)
        except Exception as e:
            error_label = QLabel(f'Error creating VTK widget:\n{e}')
            error_label.setStyleSheet("color: red; padding: 20px;")
            middle_layout.addWidget(error_label, stretch=3)

        # Results viewers
        results_widget = QWidget()
        results_layout = QGridLayout(results_widget)
        results_layout.setSpacing(5)

        self.result_labels = []
        for i in range(5):
            result_container = QWidget()
            result_container_layout = QVBoxLayout(result_container)
            result_container_layout.setContentsMargins(0, 0, 0, 0)
            result_container_layout.setSpacing(2)

            dissim_label = QLabel(f'Result {i+1}')
            dissim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dissim_label.setStyleSheet("font-size: 10px; font-weight: bold;")
            result_container_layout.addWidget(dissim_label)
            self.result_labels.append(dissim_label)

            try:
                vtk_res_widget = QVTKRenderWindowInteractor(result_container)
                plot = Plotter(qt_widget=vtk_res_widget, N=1, bg='lightgrey')
                self.result_plotters.append(plot)
                result_container_layout.addWidget(vtk_res_widget)
            except Exception as e:
                error_label = QLabel(f'VTK Error')
                error_label.setStyleSheet("color: red; font-size: 8px;")
                result_container_layout.addWidget(error_label)

            results_layout.addWidget(result_container, 0, i)

        middle_layout.addWidget(results_widget, stretch=1)

        # Right panel - features
        feature_panel = QWidget()
        feature_layout = QVBoxLayout(feature_panel)
        feature_panel.setMaximumWidth(400)

        feature_label = QLabel('Feature Visualization')
        feature_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feature_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        feature_layout.addWidget(feature_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        self.scalar_features_text = QTextEdit()
        self.scalar_features_text.setReadOnly(True)
        self.scalar_features_text.setMaximumHeight(200)
        self.scalar_features_text.setPlainText("Scalar Features:\nLoad a mesh to see features")
        scroll_layout.addWidget(QLabel("Scalar Features:"))
        scroll_layout.addWidget(self.scalar_features_text)

        self.feature_figure = Figure(figsize=(4, 8))
        self.feature_canvas = FigureCanvas(self.feature_figure)
        scroll_layout.addWidget(QLabel("Histogram Features:"))
        scroll_layout.addWidget(self.feature_canvas)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        feature_layout.addWidget(scroll)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(middle_panel, stretch=2)
        main_layout.addWidget(feature_panel)

        self.add_axes()
        self.load_categories()

        if not self.searcher:
            self.status_label.setText(f"Search disabled (missing {FEATURE_CSV})")
            self.metric_dropdown.setEnabled(False)
            self.search_button.setEnabled(False)

    def add_axes(self):
        """Add coordinate axes to the viewer"""
        try:
            self.axes_object = Axes(
                xrange=(-0.5, 0.5),
                yrange=(-0.5, 0.5),
                zrange=(-0.5, 0.5),
                xtitle='X',
                ytitle='Y',
                ztitle='Z',
                text_scale=1.2,
                c='black',
            )
            self.origin_axes = self.axes_object
            self.plotter.add(self.axes_object)
        except Exception as e:
            print(f"Warning: Could not create axes: {e}")
            self.origin_axes = None

    def load_categories(self):
        """Load all category folders from the database"""
        try:
            if os.path.exists(self.database_location):
                categories = [d for d in os.listdir(self.database_location)
                            if os.path.isdir(os.path.join(self.database_location, d))]
                categories.sort()
                self.category_dropdown.addItems(categories)
            else:
                self.status_label.setText(f'Database not found:\n{self.database_location}')
        except Exception as e:
            self.status_label.setText(f'Error loading categories:\n{e}')

    def load_category_objects(self, category):
        """Load all objects from the selected category"""
        self.object_list.clear()
        if not category:
            return

        try:
            category_path = os.path.join(self.database_location, category)
            if os.path.exists(category_path):
                objects = [f for f in os.listdir(category_path)
                          if f.endswith('.obj')]
                objects.sort()
                self.object_list.addItems(objects)
                self.status_label.setText(f'{len(objects)} objects in {category}')
            else:
                self.status_label.setText(f'Category folder not found')
        except Exception as e:
            self.status_label.setText(f'Error loading objects:\n{e}')

    def load_mesh_from_list(self, item):
        """Load the selected mesh from the list"""
        category = self.category_dropdown.currentText()
        if not category:
            return

        object_name = item.text()
        file_path = os.path.join(self.database_location, category, object_name)
        self.load_mesh(file_path)

    def load_mesh(self, file_path):
        """Load and display a mesh file"""
        if file_path:
            try:
                self.current_file = file_path
                self.display_mesh(self.current_file)

                # Update mesh info
                import trimesh
                mesh = trimesh.load(file_path)
                v_count = len(mesh.vertices)
                f_count = len(mesh.faces)
                self.mesh_info_label.setText(f'Vertices: {v_count} | Faces: {f_count}')
                self.status_label.setText(f'Loaded:\n{os.path.basename(file_path)}')

                # Extract and display features
                self.extract_and_display_features()

                if self.searcher:
                    self.search_button.setEnabled(True)

            except Exception as e:
                self.status_label.setText(f'Error loading:\n{e}')
                self.search_button.setEnabled(False)

    def extract_and_display_features(self):
        """Extract features from the current mesh and display them"""
        if not self.current_file:
            return

        try:
            category = self.category_dropdown.currentText()
            object_name = os.path.basename(self.current_file)
            # The file is already normalized with _rm.obj suffix
            query_label = f"{category}/{object_name}"

            if self.searcher and query_label in self.searcher.features_df.index:
                features = self.searcher.features_df.loc[query_label]
                self.current_features = features

                scalar_names = ['Surface area', 'Sphericity', 'Rectangularity',
                               'Diameter', 'Convexity', 'Eccentricity']
                scalar_text = "Scalar Features:\n" + "="*30 + "\n"
                for name in scalar_names:
                    if name in features.index:
                        scalar_text += f"{name:20s}: {features[name]:.4f}\n"
                self.scalar_features_text.setPlainText(scalar_text)

                self.display_histograms(features)
            else:
                self.scalar_features_text.setPlainText(f"Features not found for:\n{query_label}")

        except Exception as e:
            print(f"Error extracting features: {e}")
            self.scalar_features_text.setPlainText(f"Error extracting features:\n{e}")

    def display_histograms(self, features):
        """Display histogram features as bar charts"""
        try:
            self.feature_figure.clear()

            hist_descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
            n_bins = 20

            for idx, desc in enumerate(hist_descriptors):
                ax = self.feature_figure.add_subplot(5, 1, idx + 1)

                bin_cols = [f'{desc}_bin_{i}' for i in range(n_bins)]
                hist_values = [features[col] if col in features.index else 0 for col in bin_cols]

                ax.bar(range(n_bins), hist_values, color='steelblue', edgecolor='black', linewidth=0.5)
                ax.set_title(desc, fontsize=10, fontweight='bold')
                ax.set_xlim(-0.5, n_bins - 0.5)
                ax.set_ylim(0, max(hist_values) * 1.1 if max(hist_values) > 0 else 1)
                ax.set_ylabel('Frequency', fontsize=8)
                if idx == len(hist_descriptors) - 1:
                    ax.set_xlabel('Bin', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(axis='y', alpha=0.3)

            self.feature_figure.tight_layout()
            self.feature_canvas.draw()

        except Exception as e:
            print(f"Error displaying histograms: {e}")

    def toggle_axes_visibility(self):
        """Toggle the visibility of coordinate axes"""
        if self.origin_axes:
            if self.show_axes_checkbox.isChecked():
                self.plotter.add(self.origin_axes)
            else:
                self.plotter.remove(self.origin_axes)
            self.plotter.render()

    def find_similar_shapes(self):
        """Perform search and display the top 5 results"""
        if not self.searcher or not self.current_file:
            self.status_label.setText("Searcher not ready or no mesh loaded.")
            return

        category = self.category_dropdown.currentText()
        object_name = os.path.basename(self.current_file)
        # The file is already normalized with _rm.obj suffix
        query_label = f"{category}/{object_name}"

        metric = self.metric_dropdown.currentText()

        self.status_label.setText(f"Searching for similar shapes to {object_name}...")
        QApplication.processEvents()

        try:
            results_with_distances = self.searcher.search_with_distances(
                query_label=query_label, metric=metric, top_n=5
            )

            for i, (res_label, distance) in enumerate(results_with_distances):
                res_path = os.path.join(NORMALISED_DATABASE_LOCATION, res_label)
                self.display_result_mesh(res_path, i, distance)

            for i in range(len(results_with_distances), 5):
                self.result_plotters[i].clear().render()
                self.result_labels[i].setText(f'Result {i+1}')

            self.status_label.setText(f"Top 5 results for {object_name}")

        except Exception as e:
            self.status_label.setText(f"Search Error: {e}")

    def display_result_mesh(self, file_path, index, dissimilarity=None):
        """Display a result mesh in one of the five small viewers"""
        if index >= len(self.result_plotters):
            return
        try:
            plot = self.result_plotters[index]
            plot.clear()
            mesh = load(file_path).lighting('default').linecolor('black').linewidth(0.5)
            plot.show(mesh, resetcam=True)

            if dissimilarity is not None:
                obj_name = os.path.basename(file_path)
                self.result_labels[index].setText(f'{obj_name}\nDissimilarity: {dissimilarity:.4f}')
            else:
                self.result_labels[index].setText(f'Result {index+1}')
        except Exception as e:
            print(f"Error displaying result mesh {file_path}: {e}")

    def display_mesh(self, file_path):
        """Display mesh in the main viewer"""
        try:
            self.plotter.clear()

            if self.show_axes_checkbox.isChecked() and self.origin_axes:
                self.plotter.add(self.origin_axes)

            mesh = load(file_path).lighting('default').linecolor('black').linewidth(1)
            self.plotter.show(mesh, resetcam=True)
        except Exception as e:
            self.status_label.setText(f'Error displaying mesh:\n{e}')

    def closeEvent(self, event):
        """Clean up VTK widgets properly"""
        try:
            self.plotter.close()
            for plotter in self.result_plotters:
                plotter.close()
        except:
            pass
        event.accept()


def main():
    """Main entry point"""
    print("=" * 60)
    print("Minimal Mesh Viewer Starting")
    print("=" * 60)
    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"Platform: {sys.platform}")

    try:
        from PyQt6 import QtCore
        print(f"PyQt6 version: {getattr(QtCore, 'PYQT_VERSION_STR', 'N/A')}")
        print(f"Qt version: {getattr(QtCore, 'QT_VERSION_STR', 'N/A')}")
    except Exception as e:
        print(f"Qt version check failed: {e}")

    try:
        import vedo
        print(f"vedo version: {getattr(vedo, '__version__', 'N/A')}")
    except Exception as e:
        print(f"vedo version check failed: {e}")

    print("\nVTK Available:", VTK_AVAILABLE)
    print("vedo Available:", VEDO_AVAILABLE)
    print("Querying Available:", QUERYING_AVAILABLE)
    print("=" * 60)

    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        print("\nQApplication created successfully")
        print(f"Platform: {app.platformName()}")

        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('MeshViewer.1.0')
            except:
                pass

        print("Creating MeshViewer window...")

        viewer = MinimalMeshViewer()
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()
        app.processEvents()

        print("MeshViewer window shown")
        print(f"Window visible: {viewer.isVisible()}")
        print(f"Window size: {viewer.width()}x{viewer.height()}")
        print("Entering Qt event loop...")
        print("=" * 60)

        sys.exit(app.exec())

    except Exception as e:
        print(f"\n{'='*60}")
        print("FATAL ERROR")
        print('='*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*60)
        sys.exit(1)


if __name__ == '__main__':
    main()


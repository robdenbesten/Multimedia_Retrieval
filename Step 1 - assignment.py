import os
import sys
import vedo


class MeshViewer:
    """
    A simple 3D mesh viewer using the vedo library.
    It allows users to cycle through a directory of .obj files,
    and toggle wireframe mode using keyboard controls.
    """

    def __init__(self, folder_path):
        """
        Initializes the viewer, finds all .obj files in the specified folder,
        and sets up the vedo plotter.

        Args:
            folder_path (str): The path to the directory containing .obj files.
        """
        self.obj_files = self._find_obj_files(folder_path)
        if not self.obj_files:
            print(f"Error: No .obj files found in '{folder_path}'.")
            sys.exit(1)

        self.current_idx = 0
        self.wireframe_mode = False
        self.plotter = vedo.Plotter(interactive=True)

        # Load the initial mesh and display it
        self.current_mesh = self._load_mesh()
        self.plotter.show(self.current_mesh, interactive=False)

        # Add keyboard callbacks for user interaction
        self.plotter.add_callback('key press', self._on_key_press)

    def _find_obj_files(self, folder_path):
        """
        Walks through a directory and its subdirectories to find all .obj files.
        """
        obj_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.obj'):
                    obj_files.append(os.path.join(root, file))
        obj_files.sort()
        return obj_files

    def _load_mesh(self):
        """
        Loads the .obj file at the current index and returns a vedo.Mesh object.
        """
        mesh_path = self.obj_files[self.current_idx]
        print(f"Loading mesh: {os.path.basename(mesh_path)}")
        return vedo.load(mesh_path)

    def _switch_mesh(self, direction):
        """
        Switches to the next or previous mesh based on the direction.
        """
        num_files = len(self.obj_files)
        self.current_idx = (self.current_idx + direction) % num_files

        new_mesh = self._load_mesh()

        # Apply the current wireframe state to the new mesh
        if self.wireframe_mode:
            new_mesh.wireframe()

        # Clear the old mesh and show the new one
        self.plotter.clear()
        self.plotter.add(new_mesh)
        self.plotter.render()

    def _toggle_wireframe(self):
        """
        Toggles the wireframe mode for the current mesh.
        """
        self.wireframe_mode = not self.wireframe_mode

        # Ensure the actor is a vedo.Mesh before calling its wireframe method
        current_actor = self.plotter.actors[0]
        if isinstance(current_actor, vedo.Mesh):
            current_actor.wireframe(self.wireframe_mode)
            self.plotter.render()

    def _on_key_press(self, event):
        """
        Callback function for keyboard events.
        """
        if event.keypress == 'Right':
            self._switch_mesh(1)
        elif event.keypress == 'Left':
            self._switch_mesh(-1)
        elif event.keypress.lower() == 'w':
            self._toggle_wireframe()

    def run(self):
        """
        Starts the interactive vedo plotter.
        """
        self.plotter.interactive().close()


# Main execution block
if __name__ == "__main__":
    # Create an instance of the viewer and run it
    # Replace 'ShapeDatabase_INFOMR-master' with the path to your folder
    viewer = MeshViewer('ShapeDatabase_INFOMR-master')
    viewer.run()
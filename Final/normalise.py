import os
import numpy as np
import pymeshlab as ml
import trimesh

TARGET_VERTICES = 5000
ORIGINAL_DATABASE_LOCATION = r'ShapeDatabase_INFOMR-master\Original Database'


class Mesh:
    def __init__(self, file_path, target_vertices=TARGET_VERTICES):
        self.file_path = file_path
        self.target_vertices = target_vertices
        self.ms = ml.MeshSet()
        self.ms.load_new_mesh(file_path)
        
        if self.ms.current_mesh().vertex_number() == 0:
            raise ValueError(f"Empty mesh loaded from {file_path}")
    
    @staticmethod
    def _triangle_center(vertices, faces):
        triangle_vertices = vertices[faces]
        center = triangle_vertices.mean(axis=1)
        return center
    
    def remesh(self):
        # Clean mesh
        self.ms.apply_filter("meshing_remove_duplicate_faces")
        self.ms.apply_filter("meshing_remove_duplicate_vertices")
        self.ms.apply_filter("meshing_remove_unreferenced_vertices")
        self.ms.apply_filter("meshing_remove_null_faces")
        self.ms.apply_filter("meshing_repair_non_manifold_edges")
        self.ms.apply_filter("meshing_repair_non_manifold_vertices")
        
        # Remesh to target
        counter = consecutive_failures = 0
        while (self.ms.current_mesh().vertex_number() != self.target_vertices and 
               counter < 20 and consecutive_failures < 3):
            counter += 1
            current_vertices = self.ms.current_mesh().vertex_number()
            try:
                if current_vertices < self.target_vertices:
                    self.ms.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
                    consecutive_failures = 0
                elif current_vertices > self.target_vertices:
                    estimated_faces = int(self.ms.current_mesh().face_number() * 
                                        (self.target_vertices / current_vertices))
                    if estimated_faces > 0:
                        self.ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                                            targetfacenum=estimated_faces, qualitythr=0.5,
                                            preservenormal=True, preserveboundary=True,
                                            preservetopology=True, optimalplacement=True, 
                                            autoclean=True)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
            except Exception as e:
                consecutive_failures += 1
                print(f"Remeshing iteration {counter} failed: {e}")
        
        if consecutive_failures >= 3:
            print(f"Stopped remeshing after {consecutive_failures} consecutive failures. "
                  "Proceeding with current mesh.")
        
        return self
    
    def normalize(self):
        """
        Apply all normalization transformations using trimesh and numpy:
        1. Center to origin
        2. PCA alignment
        3. Scale to unit cube
        4. Flip based on moment test
        """
        # Get vertices and faces from PyMeshLab mesh after remeshing
        ml_mesh = self.ms.current_mesh()
        vertices = ml_mesh.vertex_matrix()
        faces = ml_mesh.face_matrix()
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # 1. CENTER: Translate to origin
        centralized = mesh.centroid
        mesh.apply_translation(-centralized)
        
        # 2. PCA ALIGNMENT
        covariance_matrix = np.cov(mesh.vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues (descending)
        index = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, index]
        
        # Normalize eigenvectors
        eigenvectors[:, 0] = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
        eigenvectors[:, 1] = eigenvectors[:, 1] / np.linalg.norm(eigenvectors[:, 1])
        eigenvectors[:, 2] = eigenvectors[:, 2] / np.linalg.norm(eigenvectors[:, 2])
        
        e1, e2 = eigenvectors[:, 0], eigenvectors[:, 1]
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        
        # Apply rotation
        xi_updated = np.dot(mesh.vertices, e1)
        yi_updated = np.dot(mesh.vertices, e2)
        zi_updated = np.dot(mesh.vertices, e3)
        
        aligned_vertices = np.column_stack((xi_updated, yi_updated, zi_updated))
        aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=mesh.faces, process=False)
        
        # 3. SCALE: Scale to unit cube
        bounding_box = aligned_mesh.bounds
        size = bounding_box[1] - bounding_box[0]
        largest = size.max()
        
        if largest == 0:
            print("Warning: Cannot scale mesh with zero dimensions")
            return self
        
        scale_factor = 1.0 / largest
        aligned_mesh.apply_scale(scale_factor)
        
        # 4. FLIP: Moment test flipping
        center_of_triangle = self._triangle_center(aligned_mesh.vertices, aligned_mesh.faces)
        
        list_of_f = []
        for i in [0, 1, 2]:
            fi = np.sum(np.sign(center_of_triangle[:, i]) * (center_of_triangle[:, i] ** 2))
            list_of_f.append(fi)
        
        list_of_f = np.array(list_of_f)
        value = np.sign(list_of_f)
        
        # Apply flipping to vertices
        xi_updated = aligned_mesh.vertices[:, 0] * value[0]
        yi_updated = aligned_mesh.vertices[:, 1] * value[1]
        zi_updated = aligned_mesh.vertices[:, 2] * value[2]
        flipped_vertices = np.column_stack((xi_updated, yi_updated, zi_updated))
        
        # Flip faces if determinant is negative
        faces = aligned_mesh.faces
        if value[0] * value[1] * value[2] == -1:
            faces = np.fliplr(aligned_mesh.faces)

        # Create final normalized mesh
        normalized_mesh = trimesh.Trimesh(vertices=flipped_vertices, faces=faces, process=False)
        
        # Update PyMeshLab mesh by saving and reloading
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            normalized_mesh.export(temp_path)
            self.ms.clear()
            self.ms.load_new_mesh(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return self
    
    def full_normalize(self):
        """Apply remeshing followed by normalization."""
        self.remesh()
        self.normalize()
        return self
    
    def save(self, output_dir="remeshed_output", suffix="_rm"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the original file path to construct output name
        original_basename = os.path.basename(self.file_path)
        base_name = os.path.splitext(original_basename)[0]
        output_filename = base_name + suffix + ".obj"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            self.ms.save_current_mesh(output_path)
            print(f"Saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving mesh: {e}")
            return None
    
    def vertex_count(self):
        return self.ms.current_mesh().vertex_number()
    


if __name__ == "__main__":
    # Process all .obj files from the original database
    database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 ORIGINAL_DATABASE_LOCATION)

    for root, dirs, files in os.walk(database_path):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                print(f"\nProcessing: {file_path}")
                try:
                    # Create a new Mesh object and process it
                    mesh = Mesh(file_path)
                    mesh.full_normalize()
                    mesh.save()
                except Exception as e:
                    print(f"Failed to process {file}: {e}")

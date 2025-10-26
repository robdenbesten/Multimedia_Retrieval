import os
import numpy as np
import pymeshlab as ml

TARGET_VERTICES = 5000
ORIGINAL_DATABASE_LOCATION = r'ShapeDatabase_INFOMR-master\Original Database'


def triangle_center(vertices, faces):
    triangle_vertices = vertices[faces]
    center = triangle_vertices.mean(axis=1)
    return center


def remesh(obj):
    ms = ml.MeshSet()
    ms.load_new_mesh(obj)
    
    if ms.current_mesh().vertex_number() == 0:
        print(f"Empty mesh loaded from {obj}")
        return None
    
    # Clean mesh
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    
    # # Additional cleaning to ensure manifoldness
    # ms.apply_filter("meshing_remove_connected_component_by_face_number", mincomponentsize=50)
    # ms.apply_filter("meshing_close_holes", maxholesize=30)
    
    # Remesh to target
    counter = consecutive_failures = 0
    while (ms.current_mesh().vertex_number() != TARGET_VERTICES and 
           counter < 20 and consecutive_failures < 3):
        counter += 1
        current_vertices = ms.current_mesh().vertex_number()
        print(current_vertices)
        try:
            if current_vertices < TARGET_VERTICES:
                ms.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
                consecutive_failures = 0
            elif current_vertices > TARGET_VERTICES:
                estimated_faces = int(ms.current_mesh().face_number() * (TARGET_VERTICES / current_vertices))
                if estimated_faces > 0:
                    ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                                        targetfacenum=estimated_faces, qualitythr=0.5,
                                        preservenormal=True, preserveboundary=True,
                                        preservetopology=True, optimalplacement=True, autoclean=True)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
        except Exception as e:
            consecutive_failures += 1
            print(f"Remeshing iteration {counter} failed: {e}")
    
    if consecutive_failures >= 3:
        print(f"Stopped remeshing after {consecutive_failures} consecutive failures. Proceeding with current mesh.")
    
    return ms


def center(ms):
    # Get centroid and translate to origin
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    centralized = vertices.mean(axis=0)
    vertices -= centralized
    return mesh


def scale(ms):
    # Scale by largest bounding box dimension
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    bbox = mesh.bounding_box()
    
    # Calculate dimensions from bbox min and max
    grootte = bbox.max() - bbox.min()
    langste = grootte.max()
    
    if langste == 0:
        return mesh
    
    scale_factor = 1.0 / langste
    vertices *= scale_factor
    return mesh


def PCA_align(ms):
    # PCA alignment using covariance matrix
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    
    covariance_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues
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
    xi_updated = np.dot(vertices, e1)
    yi_updated = np.dot(vertices, e2)
    zi_updated = np.dot(vertices, e3)
    
    aligned_vertices = np.column_stack((xi_updated, yi_updated, zi_updated))
    vertices[:] = aligned_vertices
    return mesh


def flip(ms):
    # Flipping test based on triangle centers
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    
    center_of_triangle = triangle_center(vertices, faces)
    
    list_of_f = []
    for i in [0, 1, 2]:
        fi = np.sum(np.sign(center_of_triangle[:, i]) * (center_of_triangle[:, i] ** 2))
        list_of_f.append(fi)
    
    list_of_f = np.array(list_of_f)
    value = np.sign(list_of_f)
    
    # Apply flipping
    vertices[:, 0] *= value[0]
    vertices[:, 1] *= value[1]
    vertices[:, 2] *= value[2]
    
    # Flip faces if needed
    if value[0] * value[1] * value[2] == -1:
        face_matrix = mesh.face_matrix()
        face_matrix[:] = np.fliplr(face_matrix)
    return mesh


def full_normalise(ms):
    ms = remesh(ms)
    ms = center(ms)
    ms = scale(ms)
    ms = PCA_align(ms)
    ms = flip(ms)
    return ms


def save(obj):
    output_dir = "remeshed_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original filename from the mesh label
    mesh_label = obj.current_mesh().label()
    base_name = os.path.splitext(mesh_label)[0]
    output_filename = base_name + "_rm.obj"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        obj.save_current_mesh(output_path)
        print(f"Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return None


if __name__ == "__main__":
    # Process all .obj files from the original database
    database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ORIGINAL_DATABASE_LOCATION)

    for root, dirs, files in os.walk(database_path):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                print(f"\nProcessing: {file_path}")
                try:
                    full_normalise(file_path)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")




   
"""
ADVANCED NORMALIZATION WITH FLIPPING (OLD VERSION)
This version adds orientation correction to normalization.
It detects if shapes are flipped and corrects their orientation.
It uses PCA (principal component analysis) to find the main axes.
It creates histograms to analyze shape alignment before and after.
"""

# python
import os
import math
import trimesh
import matplotlib.pyplot as plt
import numpy as np



def triangle_center(vertices, faces):

    triangle_vertices = vertices[faces]
    center = triangle_vertices.mean(axis=1)
    return center

histogram_distance_centre_before =[]
histogram_distance_centre_after = []
largest_bounding_box_dimension_before = []
largest_bounding_box_dimension_after = []

histogram_dot_x_before = []
histogram_dot_y_before = []
histogram_dot_z_before = []

histogram_dot_x_after = []
histogram_dot_y_after = []
histogram_dot_z_after = []

histogram_fliptest_x_before = []
histogram_fliptest_y_before = []
histogram_fliptest_z_before = []
histogram_fliptest_x_after = []
histogram_fliptest_y_after = []
histogram_fliptest_z_after = []

def fullnormalising_mesh(mesh_path, out_path=None):
    try:
        mesh = trimesh.load_mesh(mesh_path)
    except Exception as e:
        print(f"FAILED to load: {mesh_path} -> {e}")
        return False

    if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
        print(f"SKIP empty/unreadable mesh: {mesh_path}")
        return False

    # original prints and transforms kept
    #histogram_distance_centre_before.append(math.dist(mesh.centroid, [0,0,0]))
    centralized = mesh.centroid
    mesh.apply_translation(-centralized)
    #histogram_distance_centre_after.append(math.dist(mesh.centroid, [0, 0, 0]))

    # vertices = mesh.vertices

    #2. alignment
    covariance_matrix = np.cov(mesh.vertices.T)

    matrix_eigenvalandvec = np.linalg.eigh(covariance_matrix)
    eigenvalues = matrix_eigenvalandvec[0]
    eigenvectors = matrix_eigenvalandvec[1]
    # #####sort eigenvectors according to eigenvalues
    index = np.argsort(eigenvalues)[::-1]

    eigenvectors = eigenvectors[:, index]
    ##############eigenvectors = [x for _, x in sorted(zip(eigenvalues, eigenvectors), reverse=True)]



    #     # ##########rotation to vertices
    # ######## aligned_vertices = vertices @ eigenvectors
    eigenvectors[:][0] = eigenvectors[:][0] / np.linalg.norm(eigenvectors[:][0])
    eigenvectors[:][1] = eigenvectors[:][1] / np.linalg.norm(eigenvectors[:][1])
    eigenvectors[:][2] = eigenvectors[:][2] / np.linalg.norm(eigenvectors[:][2])
    e1, e2 = eigenvectors[:, 0], eigenvectors[:, 1]
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    x_axis = abs(np.dot(e1, [1, 0, 0]))
    y_axis = abs(np.dot(e2, [0, 1, 0]))
    z_axis = abs(np.dot(e3, [0, 0, 1]))

    xi_updated = np.dot(mesh.vertices, e1)
    yi_updated = np.dot(mesh.vertices, e2)
    zi_updated = np.dot(mesh.vertices, e3)

    aligned_vertices = np.column_stack((xi_updated, yi_updated, zi_updated))
    # #################vertices to mesh
    aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=mesh.faces, process=False)

    # 3. scaling
    afbaken_box = aligned_mesh.bounds
    grootte = afbaken_box[1] - afbaken_box[0]
    langste = grootte.max()
    # largest_bounding_box_dimension_before.append(langste)
    if langste == 0:
        print(f"SKIP invalid bbox: {mesh_path}")
        return False

    scale = 1.0 / langste
    aligned_mesh.apply_scale(scale)
    afbaken_box = aligned_mesh.bounds
    grootte = afbaken_box[1] - afbaken_box[0]
    langste = grootte.max()

    # covariance_matrix = np.cov(aligned_mesh.vertices.T)
    #
    # matrix_eigenvalandvec = np.linalg.eigh(covariance_matrix)

    # eigenvalues = matrix_eigenvalandvec[0]
    # eigenvectors = matrix_eigenvalandvec[1]
    #
    # index = np.argsort(eigenvalues)[::-1]
    #
    # eigenvectors = eigenvectors[:, index]
    #
    # e1, e2 = eigenvectors[:, 0], eigenvectors[:, 1]
    # e1 = e1 / np.linalg.norm(e1)
    # e2 = e2 / np.linalg.norm(e2)
    # e3 = np.cross(e1, e2)
    #
    # x_axis = abs(np.dot(e1, [1, 0, 0]))
    # y_axis = abs(np.dot(e2, [0, 1, 0]))
    # z_axis = abs(np.dot(e3, [0, 0, 1]))
    #histogram_dot_x_after.append(x_axis)
    #histogram_dot_y_after.append(y_axis)
    #histogram_dot_z_after.append(z_axis)

    # #######flipping test
    #
    # ############# center of the triangles
    center_of_triangle = triangle_center(aligned_mesh.vertices, aligned_mesh.faces)

    list_of_f = []
    for i in [0, 1, 2]:
        fi = np.sum(np.sign(center_of_triangle[:, i]) * (center_of_triangle[:, i] ** 2))
        list_of_f.append(fi)

    list_of_f = np.array(list_of_f)

    # # ############flipping based on f
    value = np.sign(list_of_f)

    xi_updated = aligned_vertices[:, 0] * value[0]
    yi_updated = aligned_vertices[:, 1] * value[1]
    zi_updated = aligned_vertices[:, 2] * value[2]
    aligned_vertices = np.column_stack((xi_updated, yi_updated, zi_updated))




   #

    faces = aligned_mesh.faces
    if value[0] * value[1] * value[2] == -1:
        faces = np.fliplr(aligned_mesh.faces)
    aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=faces, process=False)

    ######for testing:
   #  center_of_triangle = triangle_center(aligned_mesh.vertices, aligned_mesh.faces)
   #
   #  list_of_f = []
   #  for i in [0, 1, 2]:
   #      fi = np.sum(np.sign(center_of_triangle[:, i]) * (center_of_triangle[:, i] ** 2))
   #      list_of_f.append(fi)
   #
   #  list_of_f = np.array(list_of_f)
   #
   # #  ######### flipping based on f
   #  value = np.sign(list_of_f)


    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            aligned_mesh.export(out_path)
        except Exception as e:
            print(f"FAILED to export {out_path}: {e}")
            return False
        return True
    else:
        aligned_mesh.show()
        return True

def normalize_database(src_root='copy5000/copy5000',
                       out_root='copy5000/remeshing_flipping'):
    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)

    files = []
    for root, _, filenames in os.walk(src_root):
        # skip any files already in the output folder (defensive)
        if os.path.commonpath([os.path.abspath(root), out_root]) == out_root:
            continue
        for fn in filenames:
            if fn.lower().endswith('.obj'):
                in_path = os.path.join(root, fn)
                rel_dir = os.path.relpath(root, src_root)
                out_path = os.path.join(out_root, rel_dir, fn)
                files.append((in_path, out_path))

    if not files:
        print("No .obj files found to process.")
        return

    print(f"Found {len(files)} .obj files. Output: {out_root}")

    processed = 0
    skipped = 0
    for i, (in_path, out_path) in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {in_path}")
        ok = fullnormalising_mesh(in_path, out_path)
        if ok:
            processed += 1
        else:
            skipped += 1

    print(f"Done. Processed: {processed}, Skipped/Failed: {skipped}")

if __name__ == '__main__':
    normalize_database()


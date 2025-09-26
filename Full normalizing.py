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
histogram_distance_centre = []
largest_bounding_box_dimension = []


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

    centralized = mesh.centroid
    mesh.apply_translation(-centralized)
    afbaken_box = mesh.bounds
    grootte = afbaken_box[1] - afbaken_box[0]
    langste = grootte.max()
    if langste == 0:
        print(f"SKIP invalid bbox: {mesh_path}")
        return False
    scale = 1.0 / langste
    mesh.apply_scale(scale)

    vertices = mesh.vertices
    ####why transposed??

    covariance_matrix = np.cov(mesh.vertices.T)

    matrix_eigenvalandvec = np.linalg.eigh(covariance_matrix)
    eigenvalues = matrix_eigenvalandvec[0]
    eigenvectors = matrix_eigenvalandvec[1]
    ##sort eigenvectors according to eigenvalues
    index = np.argsort(eigenvalues)[::-1]

    eigenvectors = eigenvectors[:, index]

    # matrix rotation according to axis
    axis_for_alignment = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rotated_matrix = eigenvectors @ np.linalg.inv(axis_for_alignment)

    # Ensure right-handed system (det(R) should be +1)
    if np.linalg.det(rotated_matrix) < 0:
        eigenvectors[:, -1] = eigenvectors[:, -1] * -1
        rotated_matrix = eigenvectors @ np.linalg.inv(axis_for_alignment)

        # rotation to vertices
    aligned_vertices = vertices @ rotated_matrix

    # vertices to mesh
    aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=mesh.faces, process=False)

    ##flipping test

    # center of the triangles
    center_of_triangle = triangle_center(aligned_mesh.vertices, aligned_mesh.faces)

    list_of_f = []
    for i in [0, 1, 2]:
        fi = np.sum(np.sign(center_of_triangle[:, i]) * (center_of_triangle[:, i] ** 2))
        list_of_f.append(fi)

    list_of_f = np.array(list_of_f)

    # flipping based on f
    value = np.sign(list_of_f)

    if value[0] == -1:
        flip_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],

        ])
        aligned_vertices = aligned_vertices @ flip_matrix

    if value[1] == -1:
        flip_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],

        ])
        aligned_vertices = aligned_vertices @ flip_matrix


    if value[2] == -1:
        flip_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],

        ])
        aligned_vertices = aligned_vertices @ flip_matrix


        #
    aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=aligned_mesh.faces, process=False)

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

def normalize_database(src_root='ShapeDatabase_INFOMR-master/Original Database',
                       out_root='ShapeDatabase_INFOMR-master/normalized_database'):
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




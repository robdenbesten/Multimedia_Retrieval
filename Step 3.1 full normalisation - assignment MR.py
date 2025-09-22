import trimesh
import math
import numpy as np

def normalising_mesh(mesh):
    mesh = trimesh.load_mesh(mesh)

    ####why this?
    vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    print(vertices)

    centralized = mesh.centroid
    mesh.apply_translation(-centralized)
    afbaken_box = mesh.bounds
    grootte = afbaken_box[1] - afbaken_box[0]
    langste = grootte.max()
    scale = 1/langste
    mesh.apply_scale(scale)

    ####why transposed??

    covariance_matrix = np.cov(mesh.vertices.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)


    index = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, index]

    # We want eigenvector 0 -> x-axis, eigenvector 1 -> y-axis
    target_axes = np.eye(3)
    R = eigenvectors @ np.linalg.inv(target_axes)

    # Ensure right-handed system (det(R) should be +1)
    if np.linalg.det(R) < 0:
        eigenvectors[:, -1] *= -1
        R = eigenvectors @ np.linalg.inv(target_axes)

    # Apply rotation
    align_vertices = vertices @ R

    # Return new aligned mesh
    align_mesh = trimesh.Trimesh(vertices=align_vertices, faces=mesh.faces, process=False)
    align_mesh.show()
    ##find largest




normalising_mesh("bottle.obj")
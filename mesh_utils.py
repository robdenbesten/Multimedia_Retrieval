"""
Shared utilities for 3D mesh processing and analysis.
"""
import os
import math
import threading
import shutil
from typing import Tuple, Dict, List, Optional
import trimesh
import pymeshlab as ml


# Constants
TARGET_VERTICES = 5000
TEMP_REMESH_DIR = os.path.abspath('temp_remesh')


def parse_obj_info(filepath: str) -> Tuple[int, int, str, str]:
    """Parse OBJ file and extract basic information."""
    vertices, faces = [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                if len(parts) == 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append(parts[1:])
    
    # Determine face types
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
    
    # Calculate bounding box
    bbox = "N/A"
    if vertices:
        xs, ys, zs = zip(*vertices)
        bbox = f"X:[{min(xs):.2f},{max(xs):.2f}] Y:[{min(ys):.2f},{max(ys):.2f}] Z:[{min(zs):.2f},{max(zs):.2f}]"
    
    return len(vertices), len(faces), face_type, bbox


def clean_mesh(mesh_set: ml.MeshSet) -> None:
    """Apply standard mesh cleaning operations."""
    cleaning_filters = [
        "meshing_remove_duplicate_faces",
        "meshing_remove_duplicate_vertices", 
        "meshing_remove_unreferenced_vertices",
        "meshing_remove_null_faces",
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices"
    ]
    
    for filter_name in cleaning_filters:
        try:
            mesh_set.apply_filter(filter_name)
        except ml.PyMeshLabException:
            pass  # Continue if filter fails


def decrease_vertices(mesh_set: ml.MeshSet, target_vertices: int) -> bool:
    """Decrease vertex count using quadric edge collapse."""
    try:
        estimated_faces = int(mesh_set.current_mesh().face_number() * 
                             (target_vertices / mesh_set.current_mesh().vertex_number()))
        
        mesh_set.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=estimated_faces,
            qualitythr=0.5,
            preservenormal=True,
            preserveboundary=True,
            preservetopology=True,
            optimalplacement=True,
            autoclean=True
        )
        return True
    except ml.PyMeshLabException:
        return False


def increase_vertices(mesh_set: ml.MeshSet) -> bool:
    """Increase vertex count using midpoint subdivision."""
    try:
        mesh_set.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
        return True
    except ml.PyMeshLabException:
        return False


def remesh_to_target_vertices(input_path: str, output_path: str, target_vertices: int = TARGET_VERTICES) -> bool:
    """Remesh a mesh to target vertex count."""
    if not os.path.exists(input_path):
        return False
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mesh_set = ml.MeshSet()
    mesh_set.load_new_mesh(input_path)
    
    # Clean the mesh
    clean_mesh(mesh_set)
    
    # Remeshing loop
    counter = 0
    max_iterations = 20
    
    while (mesh_set.current_mesh().vertex_number() != target_vertices and 
           counter < max_iterations):
        counter += 1
        current_vertices = mesh_set.current_mesh().vertex_number()
        
        if current_vertices < target_vertices:
            if not increase_vertices(mesh_set):
                break
        elif current_vertices > target_vertices:
            if not decrease_vertices(mesh_set, target_vertices):
                break
    
    # Save result
    try:
        mesh_set.save_current_mesh(output_path)
        return True
    except ml.PyMeshLabException:
        return False


def normalize_mesh(input_path: str, output_path: str) -> bool:
    """Normalize mesh (center and scale to unit size)."""
    if not os.path.exists(input_path):
        return False
    
    try:
        mesh = trimesh.load_mesh(input_path)
    except Exception:
        return False
    
    if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
        return False
    
    # Center at origin
    mesh.apply_translation(-mesh.centroid)
    
    # Scale to unit size
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    max_dimension = size.max()
    
    if max_dimension == 0:
        return False
    
    mesh.apply_scale(1.0 / max_dimension)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        mesh.export(output_path)
        return True
    except Exception:
        return False


def find_obj_files(folder_path: str) -> List[str]:
    """Find all OBJ files in a folder recursively."""
    obj_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    return sorted(obj_files)


def cleanup_temp_folder() -> None:
    """Delete contents of temporary folder asynchronously."""
    if not os.path.exists(TEMP_REMESH_DIR):
        return
    
    def _delete_contents():
        try:
            for filename in os.listdir(TEMP_REMESH_DIR):
                file_path = os.path.join(TEMP_REMESH_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass
    
    threading.Thread(target=_delete_contents, daemon=True).start()


def get_vertex_count(file_path: str) -> int:
    """Get vertex count from OBJ file."""
    try:
        mesh_set = ml.MeshSet()
        mesh_set.load_new_mesh(file_path)
        return mesh_set.current_mesh().vertex_number()
    except Exception:
        return 0

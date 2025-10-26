# -- coding: utf-8 --
"""
Created on Tue Oct 14 17:32:39 2025
@author: ekaza

This script combines a three-stage mesh processing pipeline:
1. Remeshing: Takes raw .obj files and remeshes them to a target vertex count using pymeshlab.
2. Normalization: Takes the remeshed files and normalizes them (center, align, scale, flip correction) using trimesh.
3. Feature Extraction: Computes histogram-based shape descriptors from normalized meshes and saves them to a CSV file.
"""

import os
import shutil
import math
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re

import pymeshlab as ml
import trimesh
import numpy as np
import pandas as pd

# ----- Settings -----

# Step 1: Remeshing Settings
ORIGINAL_INPUT_FOLDER = 'ShapeDatabase_INFOMR-master/Original Database'
REMESHED_OUTPUT_FOLDER = 'ShapeDatabase_INFOMR-master/remeshed_5000'

# Step 2: Normalization Settings
NORMALIZED_OUTPUT_FOLDER = 'ShapeDatabase_INFOMR-master/normalized_5000'

# Step 3: Feature Extraction Settings
FEATURES_CSV_PATH = 'features_raw.csv'
HISTOGRAM_BINS = 32
DESCRIPTOR_SAMPLES = 10000

# General Parameters
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Remeshing Parameters
TARGET_VERTICES = 5000
TOLERANCE = 50
MAX_ITERATIONS = 40
MIN_PROGRESS = 10
FINAL_CORRECTION_MAX = 12


# ----- Remeshing Functions (using pymeshlab) -----

def _safe_apply_filter(ms, filter_name: str, **params) -> bool:
    """Safely apply a pymeshlab filter, returning True if successful."""
    try:
        if ms.current_mesh().face_number() > 0:
            ms.apply_filter(filter_name, **params)
            return True
        return False
    except Exception:
        return False


def _initial_aggressive_clean(ms) -> None:
    """Apply aggressive cleaning once at the start for very messy meshes."""
    _safe_apply_filter(ms, 'meshing_remove_duplicate_vertices')
    _safe_apply_filter(ms, 'meshing_remove_duplicate_faces')
    _safe_apply_filter(ms, 'meshing_remove_null_faces')
    _safe_apply_filter(ms, 'meshing_remove_unreferenced_vertices')
    _safe_apply_filter(ms, 'meshing_repair_non_manifold_edges', method='Remove Faces')
    _safe_apply_filter(ms, 'meshing_repair_non_manifold_vertices', vertdispratio=0.0)
    _safe_apply_filter(ms, 'meshing_close_holes', maxholesize=100, newfaceselected=False)
    _safe_apply_filter(ms, 'meshing_repair_self_intersections')
    _safe_apply_filter(ms, 'meshing_triangulation')


def _maintenance_clean(ms) -> None:
    """Apply a lighter, faster cleaning pass during iterations."""
    _safe_apply_filter(ms, 'meshing_remove_unreferenced_vertices')
    _safe_apply_filter(ms, 'meshing_repair_non_manifold_edges', method='Remove Faces')


def _adaptive_subdivide(ms, iters: int = 1) -> bool:
    """Subdivide mesh adaptively, choosing the best method."""
    for _ in range(iters):
        if _safe_apply_filter(ms, 'meshing_surface_subdivision_midpoint', iterations=1):
            continue
        if not _safe_apply_filter(ms, 'meshing_surface_subdivision_loop', iterations=1):
            return False
    return True


def _smart_decimate(ms, target_v: int, aggressive: bool = False) -> bool:
    """Decimate intelligently toward target vertex count with quality preservation."""
    current_v = ms.current_mesh().vertex_number()
    if current_v <= target_v:
        return True

    current_f = ms.current_mesh().face_number()
    target_f = max(10, int(current_f * (target_v / max(1, current_v))))

    qualitythr = 0.0 if aggressive else 0.3
    preservetopology = not aggressive

    return _safe_apply_filter(
        ms,
        'meshing_decimation_quadric_edge_collapse',
        targetfacenum=target_f,
        targetperc=0.0,
        qualitythr=qualitythr,
        preserveboundary=True,
        boundaryweight=1.0,
        preservenormal=True,
        preservetopology=preservetopology,
        optimalplacement=True,
        planarquadric=True,
        qualityweight=False,
        autoclean=True
    )


def _edge_length_estimate(ms, target_v: int, scale: float = 1.5) -> float:
    """Estimate target edge length for isotropic remeshing."""
    bbox = ms.current_mesh().bounding_box()
    diag = bbox.diagonal()
    if diag == 0:
        return 0.0
    return (diag / (max(1, target_v) ** 0.5)) * scale


def _remesh_isotropic(ms, target_v: int, iterations: int = 15, scale: float = 1.5) -> bool:
    """Use isotropic remeshing as a powerful recovery strategy."""
    target_edge = _edge_length_estimate(ms, target_v, scale=scale)
    if target_edge <= 0.0:
        return False
    return _safe_apply_filter(
        ms,
        'meshing_isotropic_explicit_remeshing',
        iterations=iterations,
        adaptive=True,
        targetlen=target_edge
    )


def _tune_isotropic_to_target(ms, target_v: int, max_trials: int = 6) -> bool:
    """Binary search over targetlen to drive vertex count into tolerance."""
    base = _edge_length_estimate(ms, target_v, scale=1.5)
    if base <= 0.0:
        return False
    low = base / 3.0
    high = base * 3.0

    ok_any = False
    for _ in range(max_trials):
        tl = 0.5 * (low + high)
        if not _safe_apply_filter(ms, 'meshing_isotropic_explicit_remeshing', iterations=5, targetlen=tl):
            high = tl
            continue
        ok_any = True
        cv = ms.current_mesh().vertex_number()
        if abs(cv - target_v) <= TOLERANCE:
            return True
        if cv > target_v:
            low = tl
        else:
            high = tl
    return ok_any


def _precision_correction(ms, target_v: int) -> None:
    """Deterministic push into tolerance with escalation."""
    prev_v = -1
    stagnation = 0
    for i in range(FINAL_CORRECTION_MAX):
        cv = ms.current_mesh().vertex_number()
        if abs(cv - target_v) <= TOLERANCE:
            return

        if cv > target_v + TOLERANCE:
            _smart_decimate(ms, max(target_v, cv - (cv - target_v) // 2))
        else:
            _adaptive_subdivide(ms, 1)

        new_v = ms.current_mesh().vertex_number()
        if abs(new_v - prev_v) < max(5, MIN_PROGRESS // 2):
            stagnation += 1
        else:
            stagnation = 0
        prev_v = new_v

        if i % 3 == 2:
            _maintenance_clean(ms)

        if stagnation >= 2:
            _remesh_isotropic(ms, target_v, iterations=5)
            stagnation = 0


def remesh_to_target_vertices(input_file: str, output_file: str, target_v: int = TARGET_VERTICES) -> Tuple[bool, str]:
    """Robustly remesh a single file, returning a status tuple."""
    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(input_file)

        mesh = ms.current_mesh()
        if mesh.vertex_number() < 10 or mesh.face_number() < 5:
            return False, 'Mesh is too small or degenerate'

        initial_v = mesh.vertex_number()
        _initial_aggressive_clean(ms)

        iteration = 0
        last_v = -1
        stagnation_remesh_done = False

        while iteration < MAX_ITERATIONS:
            current_v = ms.current_mesh().vertex_number()
            if ms.current_mesh().face_number() < 3:
                return False, 'Mesh became degenerate during processing'

            if abs(current_v - target_v) <= TOLERANCE:
                break

            if abs(current_v - last_v) < MIN_PROGRESS:
                if not stagnation_remesh_done:
                    _remesh_isotropic(ms, target_v, iterations=10)
                    stagnation_remesh_done = True
                else:
                    break
            else:
                stagnation_remesh_done = False

            last_v = current_v

            if current_v < target_v - TOLERANCE:
                if not _adaptive_subdivide(ms, 1):
                    break
            else:
                if not _smart_decimate(ms, target_v):
                    break

            iteration += 1
            if iteration % 10 == 0:
                _maintenance_clean(ms)

        _precision_correction(ms, target_v)
        if abs(ms.current_mesh().vertex_number() - target_v) > TOLERANCE:
            _tune_isotropic_to_target(ms, target_v)

        _maintenance_clean(ms)
        _safe_apply_filter(ms, 'meshing_triangulation')

        final_v = ms.current_mesh().vertex_number()
        deviation = abs(final_v - target_v)
        status = '✓' if deviation <= TOLERANCE else '~'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ms.save_current_mesh(output_file)
        return True, f'{status} {initial_v} → {final_v} vertices (Δ{deviation})'
    except Exception as e:
        return False, str(e)


# ----- Normalization Functions (using trimesh) -----

def triangle_center(vertices, faces):
    """Calculate the center of each triangle in a mesh."""
    triangle_vertices = vertices[faces]
    center = triangle_vertices.mean(axis=1)
    return center


def full_normalising_mesh(mesh_path: str, out_path: str) -> Tuple[bool, str]:
    """Load, normalize, and save a mesh, returning a status tuple."""
    try:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.shape[0] == 0:
            return False, "Empty or invalid mesh"

        # 1. Center the mesh at the origin using its centroid
        mesh.apply_translation(-mesh.centroid)

        # 2. Align with principal components (PCA)
        if mesh.vertices.shape[0] < 3:
            return False, "Not enough vertices for PCA"
        covariance_matrix = np.cov(mesh.vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        index = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, index]

        e1, e2 = eigenvectors[:, 0], eigenvectors[:, 1]
        e3 = np.cross(e1, e2)
        rotation_matrix_3x3 = np.column_stack((e1, e2, e3))

        # Create a 4x4 homogeneous transformation matrix for rotation
        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = rotation_matrix_3x3.T
        mesh.apply_transform(transform_4x4)

        # 3. Scale to unit size
        max_dim = mesh.extents.max()
        if max_dim < 1e-6:
            return False, "Degenerate mesh with zero size"
        mesh.apply_scale(1.0 / max_dim)

        # 4. Re-center the mesh using the center of its new bounding box
        mesh.apply_translation(-mesh.bounding_box.centroid)

        # 5. Flipping test for canonical pose
        if mesh.faces.shape[0] == 0:
            return False, "Mesh has no faces for flip test"
        center_of_triangle = triangle_center(mesh.vertices, mesh.faces)
        f_values = np.sum(np.sign(center_of_triangle) * (center_of_triangle ** 2), axis=0)
        flip_signs = np.sign(f_values)
        flip_signs[flip_signs == 0] = 1.0

        mesh.vertices *= flip_signs
        if np.prod(flip_signs) == -1:
            mesh.faces = np.fliplr(mesh.faces)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh.export(out_path)
        return True, "Normalized successfully"
    except Exception as e:
        return False, str(e)


# ----- Feature Extraction Functions -----

def extract_histogram_features(mesh: trimesh.Trimesh, n_samples: int, n_bins: int) -> Dict[str, np.ndarray]:
    """Extracts raw histogram counts for A3, D1, D2, D3, D4 shape descriptors."""
    features = {}
    v = mesh.vertices
    n_v = len(v)
    if n_v < 4:
        return {}

    # A3: Angle between 3 random vertices
    p1 = v[np.random.randint(0, n_v, n_samples)]
    p2 = v[np.random.randint(0, n_v, n_samples)]
    p3 = v[np.random.randint(0, n_v, n_samples)]
    v1 = p2 - p1
    v2 = p3 - p1
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    dots = np.einsum('ij,ij->i', v1, v2)
    # Clip to avoid domain errors with arccos
    angles = np.arccos(np.clip(dots / np.where(norms > 1e-9, norms, 1.0), -1.0, 1.0))
    features['A3'], _ = np.histogram(angles, bins=n_bins, range=(0, np.pi))

    # D1: Distance between barycenter and a random vertex
    barycenter = mesh.centroid
    dists_d1 = np.linalg.norm(v[np.random.randint(0, n_v, n_samples)] - barycenter, axis=1)
    features['D1'], _ = np.histogram(dists_d1, bins=n_bins, range=(0, 1.0))

    # D2: Distance between 2 random vertices
    p1 = v[np.random.randint(0, n_v, n_samples)]
    p2 = v[np.random.randint(0, n_v, n_samples)]
    dists_d2 = np.linalg.norm(p1 - p2, axis=1)
    features['D2'], _ = np.histogram(dists_d2, bins=n_bins, range=(0, 1.5))

    # D3: Sqrt of area of triangle from 3 random vertices
    p1 = v[np.random.randint(0, n_v, n_samples)]
    p2 = v[np.random.randint(0, n_v, n_samples)]
    p3 = v[np.random.randint(0, n_v, n_samples)]
    areas = np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1) / 2.0
    features['D3'], _ = np.histogram(np.sqrt(areas), bins=n_bins, range=(0, 1.0))

    # D4: Cbrt of volume of tetrahedron from 4 random vertices
    p1 = v[np.random.randint(0, n_v, n_samples)]
    p2 = v[np.random.randint(0, n_v, n_samples)]
    p3 = v[np.random.randint(0, n_v, n_samples)]
    p4 = v[np.random.randint(0, n_v, n_samples)]
    volumes = np.abs(np.einsum('ij,ij->i', p4 - p1, np.cross(p2 - p1, p3 - p1))) / 6.0
    features['D4'], _ = np.histogram(np.cbrt(volumes), bins=n_bins, range=(0, 0.5))

    return features


def extract_and_store_features(normalized_dir: str, csv_path: str):
    """Walks a directory of normalized meshes, extracts features, and saves to CSV."""
    print(f"\n{'=' * 20} STAGE: FEATURE EXTRACTION {'=' * 20}")
    tasks = _get_files_for_stage(normalized_dir, '', '.obj')
    if not tasks:
        print("No normalized files found to extract features from.")
        return

    print(f"Found {len(tasks)} normalized meshes. Extracting features...")
    all_features = []
    for i, (mesh_path, _, rel_path) in enumerate(tasks, 1):
        try:
            # Infer category and object name from path
            path_parts = os.path.normpath(rel_path).split(os.sep)
            category = path_parts[0] if len(path_parts) > 1 else 'Uncategorized'
            object_name = os.path.splitext(path_parts[-1])[0]

            mesh = trimesh.load_mesh(mesh_path)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
                print(f"[{i}/{len(tasks)}] ✗ Skipping {object_name}: Not enough vertices.")
                continue

            hist_features = extract_histogram_features(mesh, DESCRIPTOR_SAMPLES, HISTOGRAM_BINS)

            # Flatten features into a single dictionary for the CSV row
            row = {'ObjectName': object_name, 'Category': category}
            for name, hist in hist_features.items():
                for bin_idx, value in enumerate(hist):
                    row[f'{name}__{bin_idx+1}'] = value
            all_features.append(row)
            print(f"[{i}/{len(tasks)}] ✓ Extracted features for {object_name}")

        except Exception as e:
            print(f"[{i}/{len(tasks)}] ✗ Error processing {os.path.basename(mesh_path)}: {e}")

    if not all_features:
        print("\nFeature extraction complete: No features were generated.")
        return

    # Create DataFrame and save to CSV
    try:
        df = pd.DataFrame(all_features)
        # Ensure columns are ordered deterministically
        cols = sorted([c for c in df.columns if c not in ['ObjectName', 'Category']])
        df = df[['ObjectName', 'Category'] + cols]
        df.to_csv(csv_path, index=False)
        print(f"\nFeature extraction complete: Saved {len(df)} records to `{csv_path}`.")
    except Exception as e:
        print(f"\nError saving features to CSV: {e}")


# ----- Combined Processing Function -----

def process_mesh(input_path: str, output_path: str) -> Tuple[bool, str]:
    """Process a single mesh: remesh to target vertices, then normalize."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Step 1: Remesh
        remesh_success, remesh_msg = remesh_to_target_vertices(input_path, temp_path, TARGET_VERTICES)
        if not remesh_success:
            return False, f"Remeshing failed: {remesh_msg}"

        # Step 2: Normalize
        norm_success, norm_msg = full_normalising_mesh(temp_path, output_path)
        if not norm_success:
            return False, f"Normalization failed: {norm_msg}"

        return True, f"Processed successfully: {remesh_msg}, {norm_msg}"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ----- Main Pipeline Orchestration -----

def _get_files_for_stage(input_dir: str, output_dir: str, ext: str) -> List[Tuple[str, str, str]]:
    """Scans for files and creates a list of (input, output, relative_path) tuples."""
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(ext):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path) if output_dir else ''
                tasks.append((in_path, out_path, rel_path))
    return tasks


def run_parallel_stage(stage_name: str, tasks: list, worker_fn, *args):
    """Generic function to run a pipeline stage in parallel."""
    print(f"\n{'=' * 20} STAGE: {stage_name.upper()} {'=' * 20}")
    if not tasks:
        print(f"No files found to process for {stage_name}.")
        return

    print(f"Found {len(tasks)} files. Processing with {MAX_WORKERS} workers...")
    processed, failed = 0, 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(worker_fn, task[0], task[1], *args): task for task in tasks}
        for i, future in enumerate(as_completed(future_to_task), 1):
            in_path, _, _ = future_to_task[future]
            try:
                ok, message = future.result()
                if ok:
                    processed += 1
                    print(f"[{i}/{len(tasks)}] ✓ {os.path.basename(in_path)}: {message}")
                else:
                    failed += 1
                    print(f"[{i}/{len(tasks)}] ✗ {os.path.basename(in_path)}: {message}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(tasks)}] ✗ Error: {os.path.basename(in_path)} -> {e}")

    print(f"\n{stage_name} Complete: {processed} processed, {failed} failed/skipped.")


def main():
    """Run the full remeshing, normalization, and feature extraction pipeline."""
    # Stage 1 & 2: Process all files from the original database
    tasks = _get_files_for_stage(ORIGINAL_INPUT_FOLDER, NORMALIZED_OUTPUT_FOLDER, '.obj')
    run_parallel_stage("Processing (Remeshing + Normalization)", tasks, process_mesh)

    # Stage 3: Extract features from normalized meshes
    extract_and_store_features(NORMALIZED_OUTPUT_FOLDER, FEATURES_CSV_PATH)

    print("\nPipeline finished.")


if __name__ == '__main__':
    main()
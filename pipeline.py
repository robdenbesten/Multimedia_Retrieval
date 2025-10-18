# -- coding: utf-8 --
"""
Created on Tue Oct 14 17:32:39 2025
@author: ekaza

This script combines a three-stage mesh processing pipeline:
1. Remeshing: Takes raw .obj files and remeshes them to a target vertex count using pymeshlab.
2. Normalization: Takes the remeshed files and normalizes them (center, align, scale, flip correction) using trimesh.
3. Feature Extraction: Extracts shape descriptors and metrics from the normalized files.
"""

import os
import shutil
import math
import hashlib
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import pymeshlab as ml
import trimesh
import numpy as np

# ----- Settings -----

# Step 1: Remeshing Settings
ORIGINAL_INPUT_FOLDER = 'ShapeDatabase_INFOMR-master/Original Database'
REMESHED_OUTPUT_FOLDER = 'ShapeDatabase_INFOMR-master/remeshed_5000'

# Step 2: Normalization Settings
NORMALIZED_OUTPUT_FOLDER = 'ShapeDatabase_INFOMR-master/normalized_5000'

# Step 3: Feature Extraction Settings
FEATURES_OUTPUT_FOLDER = 'ShapeDatabase_INFOMR-master/Features'
FEATURES_JSON_PATH = os.path.join(FEATURES_OUTPUT_FOLDER, 'features.json')


# General Parameters
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Remeshing Parameters
TARGET_VERTICES = 5000
TOLERANCE = 50
MAX_ITERATIONS = 40
MIN_PROGRESS = 10
FINAL_CORRECTION_MAX = 12

# Feature Extraction Parameters
FEAT_N_SAMPLES = 250000
FEAT_SURFACE_POINTS = 5000


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
                return False, 'Mesh degenerated to less than 3 faces'

            if abs(current_v - target_v) <= TOLERANCE:
                break

            if abs(current_v - last_v) < MIN_PROGRESS:
                if not stagnation_remesh_done:
                    _remesh_isotropic(ms, target_v)
                    stagnation_remesh_done = True
                else:
                    break
            else:
                stagnation_remesh_done = False

            last_v = current_v

            if current_v < target_v - TOLERANCE:
                _adaptive_subdivide(ms, 1)
            else:
                _smart_decimate(ms, target_v)

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
        # This is the key step to ensure it fits in the [-0.5, 0.5] box
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


# ----- Feature Extraction Functions (using trimesh and numpy) -----

RANGES: Dict[str, Tuple[float, float]] = {
    'D1': (0.0, 0.7), 'D2': (0.0, 1.0), 'A3': (0.0, 180.0),
    'D3': (0.0, 0.7), 'D4': (0.0, 0.5),
}
DEFAULT_BINS: Dict[str, int] = {k: 20 for k in RANGES.keys()}


def repair_mesh(mesh: trimesh.Trimesh) -> None:
    """Run a sequence of repair functions on a mesh, ignoring errors."""
    for fn in (trimesh.repair.fill_holes, trimesh.repair.fix_winding, trimesh.repair.fix_normals):
        try: fn(mesh)
        except Exception: pass
    try: mesh.remove_unreferenced_vertices()
    except Exception: pass


def deterministic_rng_from_relpath(rel_path: str) -> np.random.Generator:
    h = hashlib.sha256(rel_path.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], byteorder='little', signed=False)
    return np.random.default_rng(seed)


def sample_surface_points_weighted(mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(mesh.faces) == 0 or len(mesh.triangles) == 0:
        idx = rng.integers(0, len(mesh.vertices), size=n_points)
        return mesh.vertices[idx]
    areas = mesh.area_faces
    probs = areas / (areas.sum() + 1e-18)
    face_idx = rng.choice(len(mesh.faces), size=n_points, p=probs)
    tri = mesh.triangles[face_idx]
    u, v = rng.random(n_points), rng.random(n_points)
    r1, r2 = np.sqrt(u), v
    return (1.0 - r1)[:, None] * tri[:, 0] + (r1 * (1.0 - r2))[:, None] * tri[:, 1] + (r1 * r2)[:, None] * tri[:, 2]


def descriptor_d1_distance_to_origin(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(points), size=n_samples)
    return np.linalg.norm(points[idx], axis=1)


def descriptor_d2_pairwise_distance(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1, i2 = rng.integers(0, len(points), size=n_samples), rng.integers(0, len(points), size=n_samples)
    return np.linalg.norm(points[i1] - points[i2], axis=1)


def descriptor_a3_triangle_angle_deg(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1, i2, i3 = rng.integers(0, len(points), size=n_samples), rng.integers(0, len(points), size=n_samples), rng.integers(0, len(points), size=n_samples)
    v1, v2, v3 = points[i1], points[i2], points[i3]
    u, w = v1 - v2, v3 - v2
    cos_theta = np.clip(np.einsum('ij,ij->i', u, w) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-18), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def descriptor_d3_sqrt_triangle_area(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1, i2, i3 = rng.integers(0, len(points), size=n_samples), rng.integers(0, len(points), size=n_samples), rng.integers(0, len(points), size=n_samples)
    a, b, c = points[i1], points[i2], points[i3]
    return np.sqrt(np.maximum(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1), 0.0))


def descriptor_d4_cuberoot_tetra_volume(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    p = points[rng.integers(0, len(points), size=(n_samples, 4))]
    return np.cbrt(np.maximum(np.abs(np.einsum('ij,ij->i', (p[:,1] - p[:,0]), np.cross(p[:,2] - p[:,0], p[:,3] - p[:,0]))) / 6.0, 0.0))


def compute_convex_hull_metrics(original_mesh: trimesh.Trimesh) -> dict:
    try: original_volume = abs(float(original_mesh.volume)) if np.isfinite(original_mesh.volume) else float('nan')
    except Exception: original_volume = float('nan')
    try: hull = original_mesh.convex_hull
    except Exception: hull = original_mesh
    try: hull_volume = abs(float(hull.volume)) if np.isfinite(hull.volume) else float('nan')
    except Exception: hull_volume = float('nan')
    try: hull_area = float(hull.area)
    except Exception: hull_area = float('nan')
    hull_extents = getattr(hull, 'extents', np.array([float('nan')] * 3))
    diameter = float(np.linalg.norm(hull_extents)) if np.all(np.isfinite(hull_extents)) else float('nan')
    compactness = (hull_area ** 3) / (36.0 * math.pi * (hull_volume ** 2)) if hull_volume and hull_volume > 0 else float('nan')
    try: rectangularity = (hull_volume / hull.bounding_box_oriented.volume) if hull.bounding_box_oriented.volume > 0 else float('nan')
    except Exception: rectangularity = float('nan')
    convexity = max(0.0, min(original_volume / hull_volume, 1.0)) if hull_volume and hull_volume > 0 and np.isfinite(original_volume) else float('nan')
    try: moments = hull.principal_inertia_components; eccentricity = float(moments[0] / moments[2]) if moments[2] > 1e-9 else float('nan')
    except Exception: eccentricity = float('nan')
    return {"Mesh volume": hull_volume, "Surface area": hull_area, "Diameter": diameter, "Compactness": compactness,
            "Rectangularity": rectangularity, "Convexity": convexity, "Eccentricity": eccentricity,
            "Sphericity": 1.0 / compactness if compactness and compactness > 0 else float('nan'), "extents": hull_extents}


def make_fixed_bin_edges(bins_dict: Dict[str, int]) -> Dict[str, np.ndarray]:
    return {k: np.linspace(RANGES[k][0], RANGES[k][1], bins_dict[k] + 1) for k in RANGES.keys()}


def l1_normalized_histogram(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.clip(x, edges[0] + 1e-12, edges[-1] - 1e-12), bins=edges)
    return counts.astype(float) / (counts.sum() or 1.0)


def extract_features_for_single_mesh(
        obj_path: str, rel_path: str, edges: Dict[str, np.ndarray], n_samples: int, surface_points: int
) -> Tuple[str, Optional[Dict[str, any]], str]:
    """Extracts features and returns them as a dictionary, ready for JSON serialization."""
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            return rel_path, None, "Failed to load as a Trimesh object"

        repair_mesh(mesh)
        metrics = compute_convex_hull_metrics(mesh)

        # Convert numpy arrays in metrics to lists for JSON compatibility
        if "extents" in metrics and isinstance(metrics["extents"], np.ndarray):
            metrics["extents"] = metrics["extents"].tolist()

        rng = deterministic_rng_from_relpath(rel_path)
        points = sample_surface_points_weighted(mesh, surface_points, rng)

        hists = {
            'D1': l1_normalized_histogram(descriptor_d1_distance_to_origin(points, n_samples, rng), edges['D1']),
            'D2': l1_normalized_histogram(descriptor_d2_pairwise_distance(points, n_samples, rng), edges['D2']),
            'A3': l1_normalized_histogram(descriptor_a3_triangle_angle_deg(points, n_samples, rng), edges['A3']),
            'D3': l1_normalized_histogram(descriptor_d3_sqrt_triangle_area(points, n_samples, rng), edges['D3']),
            'D4': l1_normalized_histogram(descriptor_d4_cuberoot_tetra_volume(points, n_samples, rng), edges['D4']),
        }

        # Convert histogram numpy arrays to lists
        hist_lists = {name: hist.tolist() for name, hist in hists.items()}

        all_features = {"metrics": metrics, "histograms": hist_lists}

        return rel_path, all_features, ""
    except Exception as e:
        return rel_path, None, str(e)


# ----- Main Pipeline Orchestration -----

def _get_files_for_stage(input_dir: str, output_dir: str, ext: str) -> List[Tuple[str, str, str]]:
    """Scans for files and creates a list of (input, output, relative_path) tuples."""
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(ext):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
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
                    print(f"[{i}/{len(tasks)}] ✓ Success: {os.path.basename(in_path)} -> {message}")
                else:
                    failed += 1
                    print(f"[{i}/{len(tasks)}] ✗ Failed: {os.path.basename(in_path)} -> {message}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(tasks)}] ✗ Error: {os.path.basename(in_path)} -> {e}")

    print(f"\n{stage_name} Complete: {processed} processed, {failed} failed/skipped.")


def run_feature_extraction_step(input_dir: str, output_json_path: str):
    """Process all .obj files and save all features to a single JSON file."""
    print(f"\n{'=' * 20} STAGE: FEATURE EXTRACTION {'=' * 20}")
    # We only need the input path and relative path for this stage
    tasks = [(t[0], t[2]) for t in _get_files_for_stage(input_dir, FEATURES_OUTPUT_FOLDER, '.obj')]
    if not tasks:
        print('No .obj files found to extract features from.')
        return

    print(f'Found {len(tasks)} meshes. Processing with {MAX_WORKERS} workers...')
    edges = make_fixed_bin_edges(DEFAULT_BINS)

    all_results = {}
    done, failed = 0, 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [
            ex.submit(extract_features_for_single_mesh, in_path, rel_path, edges, FEAT_N_SAMPLES, FEAT_SURFACE_POINTS)
            for in_path, rel_path in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rel_path, features, err = fut.result()
            base_name = os.path.basename(rel_path)
            if features is not None:
                all_results[rel_path] = features
                done += 1
                print(f'[{i}/{len(tasks)}] ✓ Extracted features for {base_name}')
            else:
                failed += 1
                print(f'[{i}/{len(tasks)}] ✗ FAILED for {base_name}: {err}')

    print(f"\nFeature Extraction Complete: {done} processed, {failed} failed.")

    if all_results:
        print(f"Writing {len(all_results)} feature sets to '{output_json_path}'...")
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print("Successfully saved JSON file.")


def main():
    """Run the full remeshing, normalization, and feature extraction pipeline."""
    # Step 1: Remesh all files from the original database
    #remesh_tasks = _get_files_for_stage(ORIGINAL_INPUT_FOLDER, REMESHED_OUTPUT_FOLDER, '.obj')
    #run_parallel_stage("Remeshing", remesh_tasks, remesh_to_target_vertices, TARGET_VERTICES)

    # Step 2: Normalize all files that were just remeshed
    #norm_tasks = _get_files_for_stage(REMESHED_OUTPUT_FOLDER, NORMALIZED_OUTPUT_FOLDER, '.obj')
    #run_parallel_stage("Normalization", norm_tasks, full_normalising_mesh)

    # Step 3: Extract features from all normalized files into a single JSON
    run_feature_extraction_step(NORMALIZED_OUTPUT_FOLDER, FEATURES_JSON_PATH)

    print("\nPipeline finished.")


if __name__ == '__main__':
    main()
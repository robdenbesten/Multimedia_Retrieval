import os
import math
import hashlib
from typing import Dict, List, Tuple
import csv

import numpy as np
import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed

from unicodedata import category

# -----------------------------
# Fixed descriptor ranges (unit AABB-diagonal scaling)
# -----------------------------
RANGES: Dict[str, Tuple[float, float]] = {
    'D1': (0.0, 0.6),    # distance to centroid (origin)
    'D2': (0.0, 1.0),    # pairwise point distance
    'A3': (0.0, 180.0),  # triangle angle (deg)
    'D3': (0.0, 0.7),    # sqrt(area), padded (tight ~0.658)
    'D4': (0.0, 0.5),    # cbrt(volume), padded (tight ~0.488)
}

# Increased bin count for higher feature resolution.
DEFAULT_BINS: Dict[str, int] = {k: 20 for k in RANGES.keys()}


# -----------------------------
# Geometry utilities and repair
# -----------------------------

def repair_mesh(mesh: trimesh.Trimesh) -> None:
    # Fill holes first if mesh is not watertight
    try:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # Remove broken/duplicate topology
    for fn in (
        getattr(mesh, 'remove_unreferenced_vertices', None),
        getattr(mesh, 'remove_duplicate_faces', None),
        getattr(mesh, 'remove_degenerate_faces', None),
        getattr(mesh, 'merge_vertices', None),
    ):
        try:
            if fn is not None:
                fn()
        except Exception:
            pass

    # Fix normals and winding
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass

    # Final cleanup attempt
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass


def deterministic_rng_from_relpath(rel_path: str) -> np.random.Generator:
    h = hashlib.sha256(rel_path.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], byteorder='little', signed=False)
    return np.random.default_rng(seed)


def sample_surface_points_weighted(mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator) -> np.ndarray:
    # Area-weighted triangle sampling; fallback to vertices for faceless meshes.
    if len(mesh.faces) == 0 or len(mesh.triangles) == 0:
        idx = rng.integers(0, len(mesh.vertices), size=n_points)
        return mesh.vertices[idx]
    tris = mesh.triangles  # (F, 3, 3)
    areas = mesh.area_faces
    probs = areas / (areas.sum() + 1e-18)
    face_idx = rng.choice(len(mesh.faces), size=n_points, p=probs)
    tri = tris[face_idx]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    u = rng.random(n_points)
    v = rng.random(n_points)
    r1 = np.sqrt(u)
    r2 = v
    pts = (1.0 - r1)[:, None] * a + (r1 * (1.0 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    return pts


# -----------------------------
# Descriptors
# -----------------------------
def descriptor_d1_distance_to_origin(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(points), size=n_samples)
    return np.linalg.norm(points[idx], axis=1)


def descriptor_d2_pairwise_distance(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1 = rng.integers(0, len(points), size=n_samples)
    i2 = rng.integers(0, len(points), size=n_samples)
    return np.linalg.norm(points[i1] - points[i2], axis=1)


def descriptor_a3_triangle_angle_deg(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1 = rng.integers(0, len(points), size=n_samples)
    i2 = rng.integers(0, len(points), size=n_samples)
    i3 = rng.integers(0, len(points), size=n_samples)
    v1, v2, v3 = points[i1], points[i2], points[i3]
    u = v1 - v2
    w = v3 - v2
    nu = np.linalg.norm(u, axis=1)
    nw = np.linalg.norm(w, axis=1)
    cos_theta = np.clip(np.einsum('ij,ij->i', u, w) / (nu * nw + 1e-18), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def descriptor_d3_sqrt_triangle_area(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1 = rng.integers(0, len(points), size=n_samples)
    i2 = rng.integers(0, len(points), size=n_samples)
    i3 = rng.integers(0, len(points), size=n_samples)
    a, b, c = points[i1], points[i2], points[i3]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.sqrt(np.maximum(area, 0.0))


def descriptor_d4_cuberoot_tetra_volume(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1 = rng.integers(0, len(points), size=n_samples)
    i2 = rng.integers(0, len(points), size=n_samples)
    i3 = rng.integers(0, len(points), size=n_samples)
    i4 = rng.integers(0, len(points), size=n_samples)
    p1, p2, p3, p4 = points[i1], points[i2], points[i3], points[i4]
    vol6 = np.abs(np.einsum('ij,ij->i', (p2 - p1), np.cross(p3 - p1, p4 - p1)))
    volume = vol6 / 6.0
    return np.cbrt(np.maximum(volume, 0.0))


# -----------------------------
# Metrics (convex hull based)
# -----------------------------
def compute_convex_hull_metrics(original_mesh: trimesh.Trimesh) -> dict:
    """
    Convex hull based metrics. Repair is applied only to a copy when computing
    the original mesh volume (used for convexity), so general mesh state is not mutated.
    """
    # Original (possibly non‑watertight) mesh volume (signed -> make positive)
    try:
        # Use a repaired copy to get a more stable original volume for convexity
        mesh_for_volume = original_mesh.copy()
        try:
            repair_mesh(mesh_for_volume)
        except Exception:
            # If repair fails, continue with the copy as-is
            pass

        original_volume = float(mesh_for_volume.volume)
        if np.isfinite(original_volume):
            original_volume = abs(original_volume)
        else:
            original_volume = float('nan')
    except Exception:
        original_volume = float('nan')

    # Convex hull (use the original mesh for hull computation)
    try:
        hull = original_mesh.convex_hull
    except Exception:
        hull = original_mesh

    try:
        hull_volume = float(hull.volume)
        if np.isfinite(hull_volume):
            hull_volume = abs(hull_volume)
        else:
            hull_volume = float('nan')
    except Exception:
        hull_volume = float('nan')

    try:
        hull_area = float(hull.area)
    except Exception:
        hull_area = float('nan')

    hull_extents = getattr(hull, 'extents', np.array([float('nan')] * 3))
    diameter = float(np.linalg.norm(hull_extents)) if np.all(np.isfinite(hull_extents)) else float('nan')

    if hull_volume > 0:
        compactness = (hull_area ** 3) / (36.0 * math.pi * (hull_volume ** 2)) if hull_area > 0 else float('nan')
        sphericity = 1.0 / compactness if compactness and compactness > 0 else float('nan')
    else:
        compactness = float('nan')
        sphericity = float('nan')

    try:
        obb_volume = float(hull.bounding_box_oriented.volume)
        rectangularity = (hull_volume / obb_volume) if (obb_volume and obb_volume > 0) else float('nan')
    except Exception:
        rectangularity = float('nan')

    if hull_volume > 0 and np.isfinite(original_volume):
        convexity = original_volume / hull_volume
        if not np.isfinite(convexity):
            convexity = float('nan')
        else:
            convexity = max(0.0, min(convexity, 1.0))
    else:
        convexity = float('nan')

    try:
        moments = hull.principal_inertia_components
        eccentricity = float(moments[0] / moments[2]) if (moments[2] and moments[2] > 1e-9) else float('nan')
    except Exception:
        eccentricity = float('nan')

    return {
        "Mesh volume": hull_volume,
        "Surface area": hull_area,
        "Diameter": diameter,
        "Compactness": compactness,
        "Rectangularity": rectangularity,
        "Convexity": convexity,
        "Eccentricity": eccentricity,
        "Sphericity": sphericity,
        "extents": hull_extents,
    }


# -----------------------------
# Histogram helpers
# -----------------------------
def make_fixed_bin_edges(bins_dict: Dict[str, int]) -> Dict[str, np.ndarray]:
    return {k: np.linspace(RANGES[k][0], RANGES[k][1], bins_dict[k] + 1) for k in RANGES.keys()}


def l1_normalized_histogram(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    eps = 1e-12
    x = np.clip(x, edges[0] + eps, edges[-1] - eps)
    counts, _ = np.histogram(x, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


# -----------------------------
# Single-file worker
# -----------------------------
def extract_features_for_single_mesh(
        obj_path: str,
        rel_path: str,
        out_path: str,
        edges: Dict[str, np.ndarray],
        n_samples: int,
        surface_points: int
) -> Tuple[str, bool, object]:
    try:
        # Split relative path into category and object name
        if '/' in rel_path:
            category, obj_name = rel_path.split('/', 1)
        else:
            category, obj_name = 'Unknown', rel_path

        mesh = trimesh.load(obj_path, force='mesh')
        metrics = compute_convex_hull_metrics(mesh)
        rng = deterministic_rng_from_relpath(rel_path)
        points = sample_surface_points_weighted(mesh, surface_points, rng)

        d1 = descriptor_d1_distance_to_origin(points, n_samples, rng)
        d2 = descriptor_d2_pairwise_distance(points, n_samples, rng)
        a3 = descriptor_a3_triangle_angle_deg(points, n_samples, rng)
        d3 = descriptor_d3_sqrt_triangle_area(points, n_samples, rng)
        d4 = descriptor_d4_cuberoot_tetra_volume(points, n_samples, rng)

        d1_hist = l1_normalized_histogram(d1, edges['D1'])
        d2_hist = l1_normalized_histogram(d2, edges['D2'])
        a3_hist = l1_normalized_histogram(a3, edges['A3'])
        d3_hist = l1_normalized_histogram(d3, edges['D3'])
        d4_hist = l1_normalized_histogram(d4, edges['D4'])

        metric_keys = ["Mesh volume", "Surface area", "Diameter", "Compactness",
                       "Rectangularity", "Convexity", "Eccentricity", "Sphericity"]

        # Start row with object name and category
        row: List[object] = [obj_name, category]

        for k in metric_keys:
            v = metrics.get(k)
            try:
                row.append(float(v))
            except (ValueError, TypeError):
                row.append(float('nan'))

        ext = metrics.get('extents', np.array([float('nan')] * 3))
        for i in range(3):
            try:
                row.append(float(ext[i]))
            except (ValueError, TypeError, IndexError):
                row.append(float('nan'))

        for arr in (d1_hist, d2_hist, a3_hist, d3_hist, d4_hist):
            row.extend([float(x) for x in np.asarray(arr, dtype=float).tolist()])

        return (obj_path, True, row)
    except Exception as e:
        return (obj_path, False, str(e))


def extract_features_for_all_meshes(
        base_dir: str = 'ShapeDatabase_INFOMR-master/normalized_5000',
        features_dir: str = 'ShapeDatabase_INFOMR-master',
        n_samples: int = 250000,
        bins_dict: Dict[str, int] = None,
        surface_points: int = 20000,
        max_workers: int = max(1, os.cpu_count() or 1)
) -> None:
    if bins_dict is None:
        bins_dict = DEFAULT_BINS.copy()
    edges = make_fixed_bin_edges(bins_dict)

    tasks: List[Tuple[str, str, str]] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_path = os.path.join(root, file)
                rel_path = os.path.relpath(obj_path, base_dir).replace('\\', '/')  # Normalize path separators
                out_path = os.path.join(features_dir, os.path.splitext(rel_path)[0] + '.csv')
                tasks.append((obj_path, rel_path, out_path))

    if not tasks:
        print('No .obj files found.')
        return

    # Add 'Relative Path' as the first column in the header
    metric_keys = ["Mesh volume", "Surface area", "Diameter", "Compactness",
                   "Rectangularity", "Convexity", "Eccentricity", "Sphericity"]
    header = ["Object", "Category"] + metric_keys + ["extents_0", "extents_1", "extents_2"]

    hist_order = ['D1', 'D2', 'A3', 'D3', 'D4']
    for k in hist_order:
        n_bins = len(edges[k]) - 1
        header += [f'{k}_bin_{i}' for i in range(n_bins)]

    os.makedirs(features_dir, exist_ok=True)
    out_csv = os.path.join(features_dir, 'all_features.csv')

    work_total = len(tasks)
    print(f'Processing {work_total} meshes with {max_workers} workers...')
    print(f'Settings: surface_points={surface_points}, n_samples={n_samples}, bins={list(bins_dict.values())[0]}')

    done = 0
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(extract_features_for_single_mesh, obj, rel, outp, edges, n_samples, surface_points) for
                    (obj, rel, outp) in tasks]
            for fut in as_completed(futs):
                obj_path, ok, payload = fut.result()
                done += 1
                if ok and isinstance(payload, list):
                    writer.writerow(payload)
                    print(f'[{done}/{work_total}] Saved features for {os.path.basename(obj_path)}')
                else:
                    print(f'[{done}/{work_total}] FAILED for {os.path.basename(obj_path)}: {payload}')


if __name__ == '__main__':
    # Windows-safe entry point
    extract_features_for_all_meshes(
        base_dir='Normalised-objects',
        features_dir='Normalised-objects',
        # Higher quality sampling settings
        n_samples=250000,
        surface_points=5000,
        bins_dict=DEFAULT_BINS,
        max_workers=max(1, (os.cpu_count() or 2) - 1)
    )
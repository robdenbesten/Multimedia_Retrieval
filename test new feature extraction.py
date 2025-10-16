import os
import math
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# Fixed descriptor ranges (unit AABB-diagonal scaling)
# -----------------------------
RANGES: Dict[str, Tuple[float, float]] = {
    'D1': (0.0, 0.5),  # distance to centroid (origin)
    'D2': (0.0, 1.0),  # pairwise point distance
    'A3': (0.0, 180.0),  # triangle angle (deg)
    'D3': (0.0, 0.7),  # sqrt(area), padded (tight ~0.658)
    'D4': (0.0, 0.5),  # cbrt(volume), padded (tight ~0.488)
}

# Increased bin count for higher feature resolution.
DEFAULT_BINS: Dict[str, int] = {k: 20 for k in RANGES.keys()}


# -----------------------------
# Geometry utilities and repair
# -----------------------------
def robust_original_volume(mesh: trimesh.Trimesh) -> float:
    # Best-effort estimation for non-watertight meshes.
    try:
        if mesh.is_watertight and np.isfinite(mesh.volume):
            return float(mesh.volume)
    except Exception:
        pass
    try:
        m_filled = mesh.copy()
        trimesh.repair.fill_holes(m_filled)
        if m_filled.is_watertight and np.isfinite(m_filled.volume):
            return float(m_filled.volume)
    except Exception:
        pass
    try:
        ext = float(np.max(mesh.extents))
        if ext > 0 and np.isfinite(ext):
            pitch = ext / 128.0
            vg = mesh.voxelized(pitch)
            solid = vg.fill()
            vox_count = int(solid.points.shape[0])
            if vox_count > 0:
                return vox_count * (float(solid.pitch) ** 3)
    except Exception:
        pass
    # Fallback: if all else fails, use the convex hull volume.
    # This provides an upper bound on the volume.
    try:
        hull_vol = float(mesh.convex_hull.volume)
        if np.isfinite(hull_vol) and hull_vol > 0:
            return hull_vol
    except Exception:
        pass
    return float('nan')


def repair_mesh_inplace(mesh: trimesh.Trimesh) -> None:
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
    # Fill holes if not watertight
    try:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    # Cleanup
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass


def normalize_mesh_unit_diameter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Enforce centroid at origin and scale so AABB diagonal is 1. Idempotent if already normalized.
    """
    m = mesh.copy()
    center = m.centroid
    if np.linalg.norm(center) > 1e-12:
        m.apply_translation(-center)
    diameter = float(np.linalg.norm(m.extents)) + 1e-18
    if not np.isfinite(diameter) or diameter <= 0:
        return m
    m.apply_scale(1.0 / diameter)
    # Re-center to suppress numerical drift
    m.apply_translation(-m.centroid)
    return m


def _rng_from_relpath(rel_path: str) -> np.random.Generator:
    h = hashlib.sha256(rel_path.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], byteorder='little', signed=False)
    return np.random.default_rng(seed)


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator) -> np.ndarray:
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
def d1_descriptor(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(points), size=n_samples)
    return np.linalg.norm(points[idx], axis=1)


def d2_descriptor(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1 = rng.integers(0, len(points), size=n_samples)
    i2 = rng.integers(0, len(points), size=n_samples)
    return np.linalg.norm(points[i1] - points[i2], axis=1)


def a3_descriptor(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
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


def d3_descriptor(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    i1 = rng.integers(0, len(points), size=n_samples)
    i2 = rng.integers(0, len(points), size=n_samples)
    i3 = rng.integers(0, len(points), size=n_samples)
    a, b, c = points[i1], points[i2], points[i3]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.sqrt(np.maximum(area, 0.0))


def d4_descriptor(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
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
def compute_base_metrics(original_mesh: trimesh.Trimesh) -> dict:
    """
    Convex hull based metrics.
    Fix: ensure volumes are non‑negative (abs) and clamp convexity to [0,1].
    """
    # Original (possibly non‑watertight) mesh volume (signed -> make positive)
    try:
        original_volume = float(original_mesh.volume)
        if np.isfinite(original_volume):
            original_volume = abs(original_volume)
        else:
            original_volume = float('nan')
    except Exception:
        original_volume = float('nan')

    # Convex hull
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
        # Numerical tolerance; enforce valid range
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
def fixed_bin_edges(bins_dict: Dict[str, int]) -> Dict[str, np.ndarray]:
    return {k: np.linspace(RANGES[k][0], RANGES[k][1], bins_dict[k] + 1) for k in RANGES.keys()}


def hist_l1_normalized(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
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
def process_one(
        obj_path: str,
        rel_path: str,
        out_path: str,
        edges: Dict[str, np.ndarray],
        n_samples: int,
        surface_points: int
) -> Tuple[str, bool, str]:
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            try:
                mesh = mesh.dump().sum()
            except Exception:
                geoms = getattr(mesh, 'geometry', {})
                mesh = trimesh.util.concatenate(list(geoms.values()))
        repair_mesh_inplace(mesh)

        # Metrics now from convex hull; convexity uses original filled mesh volume
        metrics = compute_base_metrics(mesh)

        # Normalize (original mesh) for descriptors
        mesh_n = normalize_mesh_unit_diameter(mesh)

        rng = _rng_from_relpath(rel_path)
        points = sample_surface_points(mesh_n, surface_points, rng)

        d1 = d1_descriptor(points, n_samples, rng)
        d2 = d2_descriptor(points, n_samples, rng)
        a3 = a3_descriptor(points, n_samples, rng)
        d3 = d3_descriptor(points, n_samples, rng)
        d4 = d4_descriptor(points, n_samples, rng)

        d1_hist = hist_l1_normalized(d1, edges['D1'])
        d2_hist = hist_l1_normalized(d2, edges['D2'])
        a3_hist = hist_l1_normalized(a3, edges['A3'])
        d3_hist = hist_l1_normalized(d3, edges['D3'])
        d4_hist = hist_l1_normalized(d4, edges['D4'])

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        def to_csv(arr: np.ndarray) -> str:
            return ','.join(map(str, arr.tolist()))

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('Metrics:\n')
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    f.write(f'{k}: {to_csv(v)}\n')
                else:
                    f.write(f'{k}: {v}\n')

            f.write('\nD1_hist:\n' + to_csv(d1_hist) + '\n')
            f.write('\nD2_hist:\n' + to_csv(d2_hist) + '\n')
            f.write('\nA3_hist:\n' + to_csv(a3_hist) + '\n')
            f.write('\nD3_hist:\n' + to_csv(d3_hist) + '\n')
            f.write('\nD4_hist:\n' + to_csv(d4_hist) + '\n')

        return (obj_path, True, '')
    except Exception as e:
        return (obj_path, False, str(e))


# -----------------------------
# Parallel extraction driver
# -----------------------------
def save_features_for_all_objects_txt(
        base_dir: str = 'ShapeDatabase_INFOMR-master/after_remeshing_normalise',
        features_dir: str = 'ShapeDatabase_INFOMR-master/features_test',
        n_samples: int = 250000,
        bins_dict: Dict[str, int] = None,
        surface_points: int = 20000,
        max_workers: int = max(1, os.cpu_count() or 1)
) -> None:
    if bins_dict is None:
        bins_dict = DEFAULT_BINS.copy()
    edges = fixed_bin_edges(bins_dict)

    tasks: List[Tuple[str, str, str]] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_path = os.path.join(root, file)
                rel_path = os.path.relpath(obj_path, base_dir)
                out_path = os.path.join(features_dir, os.path.splitext(rel_path)[0] + '.txt')
                tasks.append((obj_path, rel_path, out_path))

    if not tasks:
        print('No .obj files found.')
        return

    work_total = len(tasks)
    print(f'Processing {work_total} meshes with {max_workers} workers...')
    print(f'Settings: surface_points={surface_points}, n_samples={n_samples}, bins={list(bins_dict.values())[0]}')

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one, obj, rel, outp, edges, n_samples, surface_points) for (obj, rel, outp) in tasks]
        for fut in as_completed(futs):
            obj_path, ok, err = fut.result()
            done += 1
            if ok:
                print(f'[{done}/{work_total}] Saved features for {os.path.basename(obj_path)}')
            else:
                print(f'[{done}/{work_total}] FAILED for {os.path.basename(obj_path)}: {err}')


if __name__ == '__main__':
    # Windows-safe entry point
    save_features_for_all_objects_txt(
        base_dir='ShapeDatabase_INFOMR-master/after_remeshing_normalise',
        features_dir='ShapeDatabase_INFOMR-master/features_test',
        # Higher quality sampling settings
        n_samples=250000,
        surface_points=5000,
        bins_dict=DEFAULT_BINS,
        max_workers=max(1, (os.cpu_count() or 2) - 1)
    )

"""
Compact 3D Shape Browser and Processing GUI v3
"""
import sys
import os
import math
import threading
import shutil
import json
import hashlib
from typing import Tuple, Optional, List, Dict, Any
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QCheckBox, QComboBox, QPushButton
from PyQt6.QtGui import QPalette, QColor
from vedo import Plotter, load, Box, Line
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import trimesh
import pymeshlab as ml
import numpy as np

# Constants
TARGET_VERTICES = 5000
SHAPEDATA_PARENT = os.path.abspath('ShapeDatabase_INFOMR-master/Original Database')
TEMP_REMESH_DIR = os.path.abspath('temp_remesh')

# Feature extraction constants and ranges from Feature extraction.py
RANGES: Dict[str, Tuple[float, float]] = {
    'D1': (0.0, 0.5),    # distance to centroid (origin)
    'D2': (0.0, 1.0),    # pairwise point distance
    'A3': (0.0, 180.0),  # triangle angle (deg)
    'D3': (0.0, 0.7),    # sqrt(area), padded (tight ~0.658)
    'D4': (0.0, 0.5),    # cbrt(volume), padded (tight ~0.488)
}

DEFAULT_BINS: Dict[str, int] = {k: 20 for k in RANGES.keys()}


class FeatureMatcher:
    """Feature matching class based on Feature extraction.py"""
    
    def __init__(self, features_dict_path: str):
        self.features_dict = {}
        self.load_features_dictionary(features_dict_path)
        self.n_samples = 25000
        self.surface_points = 2000
        self.bins_dict = DEFAULT_BINS.copy()
        self.edges = self._make_fixed_bin_edges()
    
    def _make_fixed_bin_edges(self) -> Dict[str, np.ndarray]:
        """Create fixed bin edges for histograms."""
        return {k: np.linspace(RANGES[k][0], RANGES[k][1], self.bins_dict[k] + 1) 
                for k in RANGES.keys()}
    
    def load_features_dictionary(self, features_dict_path: str) -> None:
        """Load the features dictionary from JSON file."""
        try:
            with open(features_dict_path, 'r', encoding='utf-8') as f:
                self.features_dict = json.load(f)
            print(f"Loaded features dictionary with {len(self.features_dict)} categories")
        except Exception as e:
            print(f"Error loading features dictionary: {e}")
            self.features_dict = {}
    
    def _repair_mesh(self, mesh: trimesh.Trimesh) -> None:
        """Repair mesh topology and geometry."""
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
        
        # Final cleanup
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
    
    def _deterministic_rng_from_path(self, file_path: str) -> np.random.Generator:
        """Create deterministic random number generator from file path."""
        h = hashlib.sha256(file_path.encode('utf-8')).digest()
        seed = int.from_bytes(h[:8], byteorder='little', signed=False)
        return np.random.default_rng(seed)
    
    def _sample_surface_points_weighted(self, mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Sample points on mesh surface weighted by triangle area."""
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
    
    def _descriptor_d1_distance_to_origin(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """D1: Distance from points to origin."""
        idx = rng.integers(0, len(points), size=self.n_samples)
        return np.linalg.norm(points[idx], axis=1)
    
    def _descriptor_d2_pairwise_distance(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """D2: Pairwise distance between random point pairs."""
        i1 = rng.integers(0, len(points), size=self.n_samples)
        i2 = rng.integers(0, len(points), size=self.n_samples)
        return np.linalg.norm(points[i1] - points[i2], axis=1)
    
    def _descriptor_a3_triangle_angle_deg(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A3: Angle in random triangles (in degrees)."""
        i1 = rng.integers(0, len(points), size=self.n_samples)
        i2 = rng.integers(0, len(points), size=self.n_samples)
        i3 = rng.integers(0, len(points), size=self.n_samples)
        v1, v2, v3 = points[i1], points[i2], points[i3]
        u = v1 - v2
        w = v3 - v2
        nu = np.linalg.norm(u, axis=1)
        nw = np.linalg.norm(w, axis=1)
        cos_theta = np.clip(np.einsum('ij,ij->i', u, w) / (nu * nw + 1e-18), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    
    def _descriptor_d3_sqrt_triangle_area(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """D3: Square root of triangle area."""
        i1 = rng.integers(0, len(points), size=self.n_samples)
        i2 = rng.integers(0, len(points), size=self.n_samples)
        i3 = rng.integers(0, len(points), size=self.n_samples)
        a, b, c = points[i1], points[i2], points[i3]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return np.sqrt(np.maximum(area, 0.0))
    
    def _descriptor_d4_cuberoot_tetra_volume(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """D4: Cube root of tetrahedron volume."""
        i1 = rng.integers(0, len(points), size=self.n_samples)
        i2 = rng.integers(0, len(points), size=self.n_samples)
        i3 = rng.integers(0, len(points), size=self.n_samples)
        i4 = rng.integers(0, len(points), size=self.n_samples)
        p1, p2, p3, p4 = points[i1], points[i2], points[i3], points[i4]
        vol6 = np.abs(np.einsum('ij,ij->i', (p2 - p1), np.cross(p3 - p1, p4 - p1)))
        volume = vol6 / 6.0
        return np.cbrt(np.maximum(volume, 0.0))
    
    def _compute_convex_hull_metrics(self, original_mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Compute geometric metrics based on convex hull."""
        # Original mesh volume
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
        
        # Compactness and sphericity
        if hull_volume > 0:
            compactness = (hull_area ** 3) / (36.0 * math.pi * (hull_volume ** 2)) if hull_area > 0 else float('nan')
            sphericity = 1.0 / compactness if compactness and compactness > 0 else float('nan')
        else:
            compactness = float('nan')
            sphericity = float('nan')
        
        # Rectangularity
        try:
            obb_volume = float(hull.bounding_box_oriented.volume)
            rectangularity = (hull_volume / obb_volume) if (obb_volume and obb_volume > 0) else float('nan')
        except Exception:
            rectangularity = float('nan')
        
        # Convexity
        if hull_volume > 0 and np.isfinite(original_volume):
            convexity = original_volume / hull_volume
            if not np.isfinite(convexity):
                convexity = float('nan')
            else:
                convexity = max(0.0, min(convexity, 1.0))
        else:
            convexity = float('nan')
        
        # Eccentricity
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
            "extents": hull_extents.tolist() if hasattr(hull_extents, 'tolist') else hull_extents,
        }
    
    def _l1_normalized_histogram(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Create L1-normalized histogram."""
        eps = 1e-12
        x = np.clip(x, edges[0] + eps, edges[-1] - eps)
        counts, _ = np.histogram(x, bins=edges)
        total = counts.sum()
        if total == 0:
            return np.zeros_like(counts, dtype=float)
        return counts.astype(float) / float(total)
    
    def extract_features_from_mesh(self, mesh: trimesh.Trimesh, file_path: str = "") -> Dict[str, Any]:
        """Extract all features from a mesh."""
        # Make a copy and repair
        mesh_copy = mesh.copy()
        self._repair_mesh(mesh_copy)
        
        # Compute geometric metrics
        metrics = self._compute_convex_hull_metrics(mesh_copy)
        
        # Sample surface points
        rng = self._deterministic_rng_from_path(file_path or "default")
        points = self._sample_surface_points_weighted(mesh_copy, self.surface_points, rng)
        
        # Compute descriptors
        d1 = self._descriptor_d1_distance_to_origin(points, rng)
        d2 = self._descriptor_d2_pairwise_distance(points, rng)
        a3 = self._descriptor_a3_triangle_angle_deg(points, rng)
        d3 = self._descriptor_d3_sqrt_triangle_area(points, rng)
        d4 = self._descriptor_d4_cuberoot_tetra_volume(points, rng)
        
        # Create histograms
        d1_hist = self._l1_normalized_histogram(d1, self.edges['D1'])
        d2_hist = self._l1_normalized_histogram(d2, self.edges['D2'])
        a3_hist = self._l1_normalized_histogram(a3, self.edges['A3'])
        d3_hist = self._l1_normalized_histogram(d3, self.edges['D3'])
        d4_hist = self._l1_normalized_histogram(d4, self.edges['D4'])
        
        return {
            "Metrics": metrics,
            "D1_hist": d1_hist.tolist(),
            "D2_hist": d2_hist.tolist(),
            "A3_hist": a3_hist.tolist(),
            "D3_hist": d3_hist.tolist(),
            "D4_hist": d4_hist.tolist()
        }
    
    def extract_features_from_file(self, obj_path: str) -> Optional[Dict[str, Any]]:
        """Extract features from an OBJ file."""
        try:
            mesh = trimesh.load(obj_path, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                try:
                    mesh = mesh.dump().sum()
                except Exception:
                    geoms = getattr(mesh, 'geometry', {})
                    mesh = trimesh.util.concatenate(list(geoms.values()))
            
            return self.extract_features_from_mesh(mesh, obj_path)
            
        except Exception as e:
            print(f"Error extracting features from {obj_path}: {e}")
            return None
    
    def _compute_histogram_distance(self, hist1: List[float], hist2: List[float]) -> float:
        """Compute Earth Mover's Distance (EMD) between two histograms."""
        if len(hist1) != len(hist2):
            print(f"HIST_DEBUG: Length mismatch {len(hist1)} vs {len(hist2)}")
            return 1.0  # Return default distance instead of inf
        
        try:
            hist1 = np.array(hist1, dtype=float)
            hist2 = np.array(hist2, dtype=float)
            
            print(f"HIST_DEBUG: Original sums: {np.sum(hist1):.6f} vs {np.sum(hist2):.6f}")
            print(f"HIST_DEBUG: Are identical? {np.array_equal(hist1, hist2)}")
            
            # Check for invalid values
            if not np.all(np.isfinite(hist1)) or not np.all(np.isfinite(hist2)):
                print(f"HIST_DEBUG: Non-finite values detected")
                return 1.0
            
            # Normalize histograms to sum to 1 (if they don't already)
            sum1, sum2 = np.sum(hist1), np.sum(hist2)
            if sum1 > 0:
                hist1 = hist1 / sum1
            if sum2 > 0:
                hist2 = hist2 / sum2
            
            # Simple EMD approximation using cumulative difference
            cum1 = np.cumsum(hist1)
            cum2 = np.cumsum(hist2)
            emd = np.sum(np.abs(cum1 - cum2))
            
            print(f"HIST_DEBUG: EMD result = {emd}")
            
            # Ensure result is finite
            if np.isfinite(emd):
                return emd
            else:
                print(f"HIST_DEBUG: EMD not finite, returning 1.0")
                return 1.0
            
        except Exception as e:
            print(f"HIST_DEBUG: Error in histogram distance calculation: {e}")
            return 1.0
    
    def _compute_metrics_distance(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> float:
        """Compute normalized distance between metric sets."""
        # Key metrics to compare (excluding extents which is handled separately)
        key_metrics = ["Mesh volume", "Surface area", "Diameter", "Compactness", 
                      "Rectangularity", "Convexity", "Eccentricity", "Sphericity"]
        
        distances = []
        for metric in key_metrics:
            val1 = metrics1.get(metric, float('nan'))
            val2 = metrics2.get(metric, float('nan'))
            
            if not (np.isfinite(val1) and np.isfinite(val2)):
                continue
            
            # Normalize by the maximum value to make distances comparable
            max_val = max(abs(val1), abs(val2), 1e-9)
            dist = abs(val1 - val2) / max_val
            distances.append(dist)
        
        return np.mean(distances) if distances else float('inf')
    
    def compute_feature_distance(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Compute distance between two feature sets."""
        # Histogram distances
        hist_distances = []
        for hist_name in ["D1_hist", "D2_hist", "A3_hist", "D3_hist", "D4_hist"]:
            if hist_name in features1 and hist_name in features2:
                dist = self._compute_histogram_distance(features1[hist_name], features2[hist_name])
                hist_distances.append(dist)
        
        hist_distance = np.mean(hist_distances) if hist_distances else float('inf')
        
        # Metrics distance
        metrics_distance = float('inf')
        if "Metrics" in features1 and "Metrics" in features2:
            metrics_distance = self._compute_metrics_distance(features1["Metrics"], features2["Metrics"])
        
        # Combined distance (weighted average - low histogram weight, high metrics weight)
        if np.isfinite(hist_distance) and np.isfinite(metrics_distance):
            return 0.1 * hist_distance + 0.9 * metrics_distance
        elif np.isfinite(hist_distance):
            return hist_distance
        elif np.isfinite(metrics_distance):
            return metrics_distance
        else:
            return float('inf')
    
    def compute_feature_distance_detailed(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, float]:
        """Compute detailed distance breakdown between two feature sets."""
        result = {}
        
        # Individual histogram distances
        for hist_name in ["D1_hist", "D2_hist", "A3_hist", "D3_hist", "D4_hist"]:
            if hist_name in features1 and hist_name in features2:
                try:
                    hist1 = features1[hist_name]
                    hist2 = features2[hist_name]
                    
                    # Debug: print actual histogram info
                    print(f"DEBUG {hist_name}: len1={len(hist1) if hist1 else 'None'}, len2={len(hist2) if hist2 else 'None'}")
                    if hist1 and hist2:
                        print(f"  First few values h1: {hist1[:3] if len(hist1) >= 3 else hist1}")
                        print(f"  First few values h2: {hist2[:3] if len(hist2) >= 3 else hist2}")
                        print(f"  Sum h1: {sum(hist1)}, Sum h2: {sum(hist2)}")
                    
                    # Check histogram properties
                    if len(hist1) != len(hist2):
                        print(f"WARNING: {hist_name} length mismatch: {len(hist1)} vs {len(hist2)}")
                        result[hist_name.replace('_hist', '')] = 1.0  # Use default distance instead of inf
                    elif not hist1 or not hist2:
                        print(f"WARNING: {hist_name} is empty")
                        result[hist_name.replace('_hist', '')] = 1.0
                    else:
                        dist = self._compute_histogram_distance(hist1, hist2)
                        print(f"DEBUG {hist_name}: computed distance = {dist}")
                        if np.isfinite(dist):
                            result[hist_name.replace('_hist', '')] = dist
                        else:
                            print(f"WARNING: {hist_name} produced infinite distance")
                            result[hist_name.replace('_hist', '')] = 1.0
                except Exception as e:
                    print(f"ERROR computing {hist_name}: {e}")
                    result[hist_name.replace('_hist', '')] = 1.0
            else:
                print(f"WARNING: {hist_name} missing in one or both feature sets")
                result[hist_name.replace('_hist', '')] = 1.0
        
        # Average histogram distance
        hist_distances = [v for v in result.values() if np.isfinite(v)]
        result['Histograms'] = np.mean(hist_distances) if hist_distances else 1.0
        
        # Metrics distance
        if "Metrics" in features1 and "Metrics" in features2:
            try:
                metrics_dist = self._compute_metrics_distance(features1["Metrics"], features2["Metrics"])
                result['Metrics'] = metrics_dist if np.isfinite(metrics_dist) else 1.0
            except Exception as e:
                print(f"ERROR computing metrics distance: {e}")
                result['Metrics'] = 1.0
        else:
            print("WARNING: Metrics missing in one or both feature sets")
            result['Metrics'] = 1.0
        
        # Total combined distance (low histogram weight, high metrics weight)
        result['Total'] = 0.1 * result['Histograms'] + 0.9 * result['Metrics']
        
        return result
    
    def find_best_matches(self, query_features: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Find the best matching shapes for a query."""
        matches = []
        
        for category, objects in self.features_dict.items():
            for object_name, features in objects.items():
                distance = self.compute_feature_distance(query_features, features)
                matches.append((category, object_name, distance))
        
        # Sort by distance and return top k
        matches.sort(key=lambda x: x[2])
        return matches[:top_k]


def parse_obj_info(filepath: str) -> Tuple[int, int, str, str]:
    """Parse OBJ file and extract basic information."""
    vertices, faces = [], []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v' and len(parts) == 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append(parts[1:])
    
    # Determine face types and bounding box in one pass
    face_types = {"triangles" if len(f) == 3 else "quads" if len(f) == 4 else "other" for f in faces}
    face_type = " and ".join(sorted(face_types)) if face_types else "unknown"
    
    bbox = "N/A"
    if vertices:
        xs, ys, zs = zip(*vertices)
        bbox = f"X:[{min(xs):.2f},{max(xs):.2f}] Y:[{min(ys):.2f},{max(ys):.2f}] Z:[{min(zs):.2f},{max(zs):.2f}]"
    
    return len(vertices), len(faces), face_type, bbox


def remesh_to_target_vertices(input_path: str, output_path: str) -> bool:
    """Remesh a mesh to target vertex count with robust error handling."""
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return False
    
    mesh_set = None
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh_set = ml.MeshSet()
        mesh_set.load_new_mesh(input_path)
        
        if mesh_set.current_mesh().vertex_number() == 0:
            print(f"Empty mesh loaded from {input_path}")
            return False
        
        # Clean mesh
        for filter_name in ["meshing_remove_duplicate_faces", "meshing_remove_duplicate_vertices",
                           "meshing_remove_unreferenced_vertices", "meshing_remove_null_faces",
                           "meshing_repair_non_manifold_edges", "meshing_repair_non_manifold_vertices"]:
            try:
                mesh_set.apply_filter(filter_name)
            except Exception as e:
                print(f"Warning: {filter_name} failed: {e}")
        
        # Remesh to target
        counter = consecutive_failures = 0
        while (mesh_set.current_mesh().vertex_number() != TARGET_VERTICES and 
               counter < 20 and consecutive_failures < 3):
            counter += 1
            current_vertices = mesh_set.current_mesh().vertex_number()
            
            try:
                if current_vertices < TARGET_VERTICES:
                    mesh_set.apply_filter("meshing_surface_subdivision_midpoint", iterations=1)
                    consecutive_failures = 0
                elif current_vertices > TARGET_VERTICES:
                    estimated_faces = int(mesh_set.current_mesh().face_number() * (TARGET_VERTICES / current_vertices))
                    if estimated_faces > 0:
                        mesh_set.apply_filter("meshing_decimation_quadric_edge_collapse",
                                            targetfacenum=estimated_faces, qualitythr=0.5,
                                            preservenormal=True, preserveboundary=True,
                                            preservetopology=True, optimalplacement=True, autoclean=True)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                else:
                    break
            except Exception as e:
                consecutive_failures += 1
                print(f"Remeshing iteration {counter} failed: {e}")
                if consecutive_failures >= 3:
                    print("Too many consecutive failures, stopping remeshing")
                    break
        
        mesh_set.save_current_mesh(output_path)
        print(f"Remeshing completed: {mesh_set.current_mesh().vertex_number()} vertices (target: {TARGET_VERTICES})")
        return True
        
    except Exception as e:
        print(f"Error during remeshing: {e}")
        return False
    finally:
        if mesh_set: del mesh_set


def normalize_mesh(input_path: str, output_path: str) -> bool:
    """Normalize mesh (center, scale, align, and flip) with robust error handling."""
    if not os.path.exists(input_path):
        print(f"Input file not found for normalization: {input_path}")
        return False
    
    mesh = None
    try:
        import numpy as np
        mesh = trimesh.load_mesh(input_path)
        
        # Validate mesh
        if not mesh or not hasattr(mesh, "vertices") or mesh.vertices is None or mesh.vertices.size == 0 or len(mesh.vertices) < 3:
            print(f"Invalid mesh for {input_path}")
            return False
        
        # Step 1: Center at origin
        centroid = mesh.centroid
        if not all(not math.isnan(x) and not math.isinf(x) for x in centroid):
            print(f"Invalid centroid for mesh {input_path}: {centroid}")
            return False
        mesh.apply_translation(-centroid)
        
        # Step 2: Scale to unit size
        bounds = mesh.bounds
        if bounds is None or len(bounds) != 2:
            print(f"Invalid bounds for mesh {input_path}")
            return False
        size = bounds[1] - bounds[0]
        max_dimension = size.max()
        if max_dimension <= 0 or math.isnan(max_dimension) or math.isinf(max_dimension):
            print(f"Invalid mesh dimensions for {input_path}: {max_dimension}")
            return False
        mesh.apply_scale(1.0 / max_dimension)
        
        # Step 3: Alignment using PCA
        covariance_matrix = np.cov(mesh.vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Normalize eigenvectors and ensure right-handed coordinate system
        e1, e2, e3 = [sorted_eigenvectors[:, i] / np.linalg.norm(sorted_eigenvectors[:, i]) for i in range(3)]
        if np.dot(e3, np.cross(e1, e2)) < 0: e3 = -e3
        
        # Transform vertices
        rotation_matrix = np.column_stack([e1, e2, e3])
        aligned_vertices = mesh.vertices @ rotation_matrix
        mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=mesh.faces, process=False)
        
        # Step 4: Flipping based on triangle center analysis
        triangle_centers = mesh.vertices[mesh.faces].mean(axis=1)
        flip_factors = np.array([np.sign(np.sum(np.sign(triangle_centers[:, axis]) * (triangle_centers[:, axis] ** 2))) for axis in range(3)])
        
        # Apply flipping
        flipped_vertices = mesh.vertices * flip_factors
        faces = np.fliplr(mesh.faces) if np.prod(flip_factors) == -1 else mesh.faces
        mesh = trimesh.Trimesh(vertices=flipped_vertices, faces=faces, process=False)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        print(f"Full normalization completed for {os.path.basename(input_path)}")
        return True
        
    except Exception as e:
        print(f"Error during normalization: {e}")
        return False
    finally:
        if mesh: del mesh


def cleanup_temp_folder() -> None:
    """Delete contents of temporary folder synchronously with timeout."""
    if not os.path.exists(TEMP_REMESH_DIR):
        return
    
    try:
        # Try to delete synchronously first (faster for small folders)
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
        # If synchronous fails, try async as fallback
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
        
        thread = threading.Thread(target=_delete_contents, daemon=True)
        thread.start()
        # Don't wait for thread to complete - let it finish in background


class Shape:
    """3D mesh shape with loading, remeshing, and normalization capabilities."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vertices = None
        self.faces = None
        self.mesh = None
        self.temp_copy_path = None
        os.makedirs(TEMP_REMESH_DIR, exist_ok=True)

    def load(self) -> None:
        """Load mesh using vedo and parse info."""
        self.mesh = load(self.file_path)
        self.vertices, self.faces, _, _ = parse_obj_info(self.file_path)

    def resample(self) -> bool:
        """Remesh to target vertex count and save to temp folder."""
        if not os.path.exists(self.file_path):
            print(f"File not found for resampling: {self.file_path}")
            return False

        try:
            name, ext = os.path.splitext(os.path.basename(self.file_path))
            self.temp_copy_path = os.path.join(TEMP_REMESH_DIR, f"{name}_processed{ext}")
            
            if remesh_to_target_vertices(self.file_path, self.temp_copy_path):
                self._load_processed_mesh()
                return True
            else:
                print(f"Remeshing failed for {self.file_path}")
                return False
        except Exception as e:
            print(f"Error during resampling: {e}")
            return False

    def normalize(self) -> bool:
        """Normalize mesh (center, scale, align, flip) in place on temp file."""
        input_path = self.temp_copy_path if self.temp_copy_path and os.path.exists(self.temp_copy_path) else self.file_path
        
        if not os.path.exists(input_path):
            print(f"File not found for normalization: {input_path}")
            return False

        try:
            if normalize_mesh(input_path, self.temp_copy_path):
                self._load_processed_mesh()
                return True
            else:
                print(f"Normalization failed for {input_path}")
                return False
        except Exception as e:
            print(f"Error during normalization: {e}")
            return False

    def _load_processed_mesh(self) -> None:
        """Load processed mesh from temp file."""
        try:
            self.mesh = load(self.temp_copy_path)
            self.vertices, self.faces, _, _ = parse_obj_info(self.temp_copy_path)
        except Exception as e:
            print(f"Failed to load processed file: {e}")
            raise


class CBSRApp(QWidget):
    """3D Shape Browser and Processing GUI."""
    
    def __init__(self, parent_folder: str):
        super().__init__()
        self.parent_folder = parent_folder
        self.loaded_shapes = []
        self.current_mesh_actor = None
        self.bbox_actor = None
        self.bbox_labels = []
        self.origin_axes = None
        self.show_bbox_preference = False
        self.show_reference_preference = True
        self.dark_mode_enabled = False
        
        # Initialize score tracking
        self.current_gallery_scores = []
        self.current_detailed_scores = []
        self.current_selected_score = None
        self.current_selected_detailed_scores = None
        self.score_labels = []
        
        # Initialize feature matcher
        self.feature_matcher = None
        features_dict_path = os.path.join(os.path.dirname(__file__), 'features_dictionary.json')
        if os.path.exists(features_dict_path):
            self.feature_matcher = FeatureMatcher(features_dict_path)
        else:
            print(f"Warning: Features dictionary not found at {features_dict_path}")
            print("Gallery will show random objects instead of feature-based matches")

        self.setWindowTitle("CBSR Debug GUI")
        self.resize(1200, 900)
        
        # Setup UI: top row (file panel + main viewer), bottom row (gallery spans full width)
        root_layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        top_row.addLayout(self._create_file_panel())
        top_row.addLayout(self._create_viewer_panel())
        root_layout.addLayout(top_row)
        root_layout.addLayout(self._create_gallery_panel())

        # Initialize with first category
        if self.categories:
            self.on_category_changed(self.categories[0])

    def _create_origin_axes(self) -> None:
        """Create origin axes (X=red, Y=green, Z=blue)."""
        # X-axis: red line
        x_axis = Line([0, 0, 0], [0.5, 0, 0]).c('red').lw(1)
        
        # Y-axis: green line
        y_axis = Line([0, 0, 0], [0, 0.5, 0]).c('green').lw(1)
        
        # Z-axis: blue line
        z_axis = Line([0, 0, 0], [0, 0, 0.5]).c('blue').lw(1)
        
        # Combine all reference objects (only axes, no cube)
        self.origin_axes = [x_axis, y_axis, z_axis]
        
        # Add reference objects to plotter
        for obj in self.origin_axes:
            self.plotter.add(obj)

    def _create_file_panel(self) -> QVBoxLayout:
        """Create file browser panel."""
        panel = QVBoxLayout()
        
        # Categories dropdown (replaces main folder selection)
        panel.addWidget(QLabel("Categories"))
        self.category_combo = QComboBox()
        self.categories = [d for d in os.listdir(self.parent_folder) 
                          if os.path.isdir(os.path.join(self.parent_folder, d))]
        self.category_combo.addItems(self.categories)
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        panel.addWidget(self.category_combo)

        # Files list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        panel.addWidget(QLabel("Files"))
        panel.addWidget(self.file_list)
        
        return panel

    def _create_viewer_panel(self) -> QVBoxLayout:
        """Create 3D viewer panel."""
        panel = QVBoxLayout()
        
        panel.addWidget(QLabel("3D Viewer"))
        self.viewer_widget = QVTKRenderWindowInteractor(self)
        self.plotter = Plotter(qt_widget=self.viewer_widget)
        panel.addWidget(self.viewer_widget)

        # Create and add origin axes
        self._create_origin_axes()

        self.info_label = QLabel("Select a file to see info.")
        panel.addWidget(self.info_label)

        self.bbox_toggle = QCheckBox("Show Bounding Box")
        self.bbox_toggle.stateChanged.connect(self.on_bbox_toggle)
        panel.addWidget(self.bbox_toggle)

        self.reference_toggle = QCheckBox("Show Reference Axes")
        self.reference_toggle.stateChanged.connect(self.on_reference_toggle)
        self.reference_toggle.setChecked(self.show_reference_preference)
        panel.addWidget(self.reference_toggle)

        self.darkmode_toggle = QCheckBox("Dark Mode")
        self.darkmode_toggle.stateChanged.connect(self.on_darkmode_toggle)
        panel.addWidget(self.darkmode_toggle)

        self.auto_normalize_toggle = QCheckBox("Normalize")
        self.auto_normalize_toggle.stateChanged.connect(self.on_auto_normalize_toggle)
        panel.addWidget(self.auto_normalize_toggle)
        return panel

    def _create_gallery_panel(self) -> QVBoxLayout:
        """Create bottom gallery panel that spans the full window width."""
        panel = QVBoxLayout()
        
        # Gallery title and selected object score
        self.gallery_title = QLabel("Similar Objects Gallery")
        panel.addWidget(self.gallery_title)
        
        self.selected_score_label = QLabel("Selected object score: N/A")
        panel.addWidget(self.selected_score_label)
        
        # Gallery viewers
        self.gallery_layout = QHBoxLayout()
        self.gallery_widgets: List[QVTKRenderWindowInteractor] = []
        self.gallery_plotters: List[Plotter] = []
        for _ in range(5):
            w = QVTKRenderWindowInteractor(self)
            p = Plotter(qt_widget=w)
            self.gallery_widgets.append(w)
            self.gallery_plotters.append(p)
            self.gallery_layout.addWidget(w)
        panel.addLayout(self.gallery_layout)
        
        # Score labels under each viewer
        self.score_labels_layout = QHBoxLayout()
        for i in range(5):
            label = QLabel(f"Viewer {i+1}: N/A")
            label.setWordWrap(True)
            label.setStyleSheet("font-family: monospace; font-size: 9px;")
            self.score_labels.append(label)
            self.score_labels_layout.addWidget(label)
        panel.addLayout(self.score_labels_layout)
        return panel

    def on_category_changed(self, category_name: str) -> None:
        """Handle category selection change."""
        self.file_list.clear()
        self.current_category = category_name
        category_path = os.path.join(self.parent_folder, category_name)
        files = [f for f in os.listdir(category_path) if f.endswith('.obj')]
        self.file_list.addItems(files)
        # Clear gallery when category changes (will be populated when user selects an object)
        self._clear_gallery()

    def on_file_selected(self, item) -> None:
        """Handle file selection and display mesh."""
        full_path = os.path.join(self.parent_folder, self.current_category, item.text())

        shape = Shape(full_path)
        shape.load()
        self.loaded_shapes.append(shape)

        # Always create a normalized copy for feature extraction (independent of display)
        normalized_shape = None
        self.info_label.setText("Normalizing for feature extraction, please wait...")
        QApplication.processEvents()
        
        normalized_shape = Shape(full_path)
        normalized_shape.load()
        
        # First resample to target vertices
        if not normalized_shape.resample():
            self.info_label.setText("Normalize failed: Remeshing step failed.")
            return
        
        # Then normalize
        if not normalized_shape.normalize():
            self.info_label.setText("Normalize failed: Normalization step failed.")
            return

        # If auto-normalize is enabled, also normalize the display shape
        if self.auto_normalize_toggle.isChecked():
            self.info_label.setText("Normalizing display, please wait...")
            QApplication.processEvents()
            
            # First resample to target vertices
            if not shape.resample():
                self.info_label.setText("Normalize failed: Remeshing step failed.")
                return
            
            # Then normalize
            if not shape.normalize():
                self.info_label.setText("Normalize failed: Normalization step failed.")
                return

        self.plotter.clear()
        # Re-add origin axes after clearing (only if reference toggle is on)
        if self.show_reference_preference:
            for axis in self.origin_axes:
                self.plotter.add(axis)
        
        # Display mesh with lighting enabled (like pressing 'L' key)
        shape.mesh.lighting('default').linecolor('black').linewidth(1)
        self.plotter.show(shape.mesh, resetcam=True)
        self.current_mesh_actor = shape.mesh

        status_text = "(Normalized) " if self.auto_normalize_toggle.isChecked() else ""
        self.info_label.setText(f"{status_text}File: {item.text()}\nVertices: {shape.vertices}")
        self.bbox_toggle.setChecked(self.show_bbox_preference)
        self.reference_toggle.setChecked(self.show_reference_preference)
        
        # If preference is to show bounding box, trigger it
        if self.show_bbox_preference:
            self.on_bbox_toggle(True)
        
        # If preference is to hide reference objects, remove them
        if not self.show_reference_preference:
            self.on_reference_toggle(False)
        
        # Extract features from NORMALIZED mesh and load similar shapes in gallery
        # Use the same relative path format as used when building the features dictionary
        rel_path = os.path.join(self.current_category, item.text())
        self.load_similar_gallery_from_normalized_mesh(normalized_shape, rel_path)

    def on_bbox_toggle(self, state) -> None:
        """Toggle bounding box display with dimension labels."""
        if not self.current_mesh_actor:
            return
        
        # Remember the user's preference
        self.show_bbox_preference = bool(state)
            
        # Always remove existing dimension info first (to prevent stacking)
        current_info = self.info_label.text()
        lines = current_info.split('\n')
        filtered_lines = [line for line in lines if not (line.startswith('BBox:') or line.startswith('[X='))]
        base_info = '\n'.join(filtered_lines)
        
        if state:
            try:
                # Create bounding box
                bounds = self.current_mesh_actor.bounds()
                self.bbox_actor = Box(bounds).wireframe().c('grey')
                self.plotter.add(self.bbox_actor)
                
                # Calculate dimensions
                x_size = bounds[1] - bounds[0]  # xmax - xmin
                y_size = bounds[3] - bounds[2]  # ymax - ymin
                z_size = bounds[5] - bounds[4]  # zmax - zmin
                
                # Add dimension info to clean base info
                dimension_info = f"\n[X={x_size:.2f}, Y={y_size:.2f}, Z={z_size:.2f}]"
                self.info_label.setText(base_info + dimension_info)
                
                # Store that we have labels (for cleanup)
                self.bbox_labels = ['info_updated']
                
            except Exception as e:
                print(f"Error creating bounding box: {e}")
                self.info_label.setText(f"Bounding box error: {str(e)}")
        else:
            # Remove bounding box
            if self.bbox_actor:
                try:
                    self.plotter.remove(self.bbox_actor)
                except Exception as e:
                    print(f"Error removing bounding box: {e}")
                self.bbox_actor = None
            
            # Set info label to clean base info (dimensions already removed above)
            self.info_label.setText(base_info)
            self.bbox_labels = []
            
        try:
            self.plotter.render()
        except Exception as e:
            print(f"Error rendering: {e}")

    def on_reference_toggle(self, state) -> None:
        """Toggle reference objects (unit cube and axes) display."""
        # Remember the user's preference
        self.show_reference_preference = bool(state)
        
        if state:
            # Add reference objects back
            for obj in self.origin_axes:
                self.plotter.add(obj)
        else:
            # Remove reference objects
            for obj in self.origin_axes:
                self.plotter.remove(obj)
        self.plotter.render()

    def on_darkmode_toggle(self, state) -> None:
        """Toggle dark mode for the entire app and 3D viewers."""
        self.dark_mode_enabled = bool(state)
        self._apply_qt_palette(self.dark_mode_enabled)
        # Update backgrounds of all plotters
        try:
            bg = 'black' if self.dark_mode_enabled else 'white'
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.background(bg)
                self.plotter.render()
            if hasattr(self, 'gallery_plotters'):
                for p in self.gallery_plotters:
                    try:
                        p.background(bg)
                        p.render()
                    except Exception:
                        pass
        except Exception:
            pass

    def _apply_qt_palette(self, dark: bool) -> None:
        """Apply a dark or default palette to the Qt application."""
        app = QApplication.instance()
        if app is None:
            return
        if not dark:
            app.setPalette(QPalette())
            return
        palette = QPalette()
        # Window
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        # Base/Alternate
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        # Tooltips
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(53, 53, 53))
        # Text/Button
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        # Bright/Dark/Shadow
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(240, 240, 240))
        app.setPalette(palette)

    def on_clean_clicked(self) -> None:
        """Clean mesh (remesh + normalize) with robust error handling."""
        if not self.loaded_shapes:
            self.info_label.setText("No shape loaded to clean!")
            return

        shape = self.loaded_shapes[-1]
        self.info_label.setText("Cleaning (remesh + normalize), please wait...")
        QApplication.processEvents()

        try:
            # Step 1: Remesh
            self.info_label.setText("Step 1/2: Remeshing...")
            QApplication.processEvents()
            
            if not shape.resample():
                self.info_label.setText("Cleaning failed: Remeshing step failed.\nCheck console for details.")
                return

            # Step 2: Normalize (in place on the same temp file)
            self.info_label.setText("Step 2/2: Normalizing...")
            QApplication.processEvents()
            
            if not shape.normalize():
                self.info_label.setText("Cleaning failed: Normalization step failed.\nCheck console for details.")
                return

            # Display result
            self.plotter.clear()
            # Re-add origin axes after clearing (only if reference toggle is on)
            if self.show_reference_preference:
                for axis in self.origin_axes:
                    self.plotter.add(axis)
            
            # Display cleaned mesh with lighting enabled (like pressing 'L' key)
            shape.mesh.lighting('default').linecolor('black').linewidth(1)
            self.plotter.show(shape.mesh, resetcam=True)
            self.current_mesh_actor = shape.mesh
            
            # Re-apply bounding box if it was previously shown
            if self.show_bbox_preference:
                self.bbox_actor = Box(self.current_mesh_actor.bounds()).wireframe().c('red')
                self.plotter.add(self.bbox_actor)
            
            self.info_label.setText(f"Cleaned successfully!\n"
                                   f"File: {os.path.basename(shape.file_path)}\n"
                                   f"Vertices: {shape.vertices}")
            
        except Exception as e:
            print(f"Unexpected error during cleaning: {e}")
            self.info_label.setText(f"Cleaning failed: Unexpected error.\nCheck console for details.\nError: {str(e)}")
            return

    def on_auto_normalize_toggle(self, state) -> None:
        """Handle auto-normalize checkbox state change and immediately update viewer."""
        # If no shape is currently loaded, just update the status message
        if not self.loaded_shapes:
            if state:
                self.info_label.setText("Normalize enabled. Select a file to see normalized version.")
            else:
                self.info_label.setText("Normalize disabled. Objects will be shown as original.")
            return
        
        # Get the current shape
        current_shape = self.loaded_shapes[-1]
        
        if state:
            # Checkbox checked - show normalized version
            self.info_label.setText("Normalizing current object, please wait...")
            QApplication.processEvents()
            
            # Create a fresh copy of the shape for normalization
            temp_shape = Shape(current_shape.file_path)
            temp_shape.load()
            
            # Normalize the temp shape
            if not temp_shape.resample():
                self.info_label.setText("Normalize failed: Remeshing step failed.")
                return
            
            if not temp_shape.normalize():
                self.info_label.setText("Normalize failed: Normalization step failed.")
                return
            
            # Update the loaded shape with normalized version
            self.loaded_shapes[-1] = temp_shape
            current_shape = temp_shape
            status_prefix = "(Normalized) "
        else:
            # Checkbox unchecked - reload original version
            self.info_label.setText("Loading original object...")
            QApplication.processEvents()
            
            # Reload the original shape
            original_shape = Shape(current_shape.file_path)
            original_shape.load()
            
            # Update the loaded shape with original version
            self.loaded_shapes[-1] = original_shape
            current_shape = original_shape
            status_prefix = ""
        
        # Clear and redisplay the mesh
        self.plotter.clear()
        
        # Re-add origin axes after clearing (only if reference toggle is on)
        if self.show_reference_preference:
            for axis in self.origin_axes:
                self.plotter.add(axis)
        
        # Display mesh with lighting enabled
        current_shape.mesh.lighting('default').linecolor('black').linewidth(1)
        self.plotter.show(current_shape.mesh, resetcam=True)
        self.current_mesh_actor = current_shape.mesh
        
        # Update info label
        filename = os.path.basename(current_shape.file_path)
        self.info_label.setText(f"{status_prefix}File: {filename}\nVertices: {current_shape.vertices}")
        
        # Re-apply bounding box if it was previously shown
        if self.show_bbox_preference:
            self.on_bbox_toggle(True)

    def _list_obj_files_in_current_category(self) -> List[str]:
        """List absolute paths to .obj files in the current category."""
        if not getattr(self, 'current_category', None):
            return []
        category_path = os.path.join(self.parent_folder, self.current_category)
        try:
            files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.lower().endswith('.obj')]
            return files
        except Exception:
            return []



    def load_similar_gallery(self, query_obj_path: str) -> None:
        """Load 5 most similar objects into gallery viewers based on feature matching."""
        if not self.feature_matcher:
            # Clear gallery if no feature matcher available
            print("No feature matcher available, clearing gallery")
            self._clear_gallery()
            return
        
        try:
            # Extract features from the selected object
            print(f"Extracting features from {query_obj_path}...")
            query_features = self.feature_matcher.extract_features_from_file(query_obj_path)
            
            if not query_features:
                print("Failed to extract features, clearing gallery")
                self._clear_gallery()
                return
            
            # Find similar objects
            print("Finding similar objects...")
            matches = self.feature_matcher.find_best_matches(query_features, top_k=5)
            
            if not matches:
                print("No matches found, clearing gallery")
                self._clear_gallery()
                return
            
            # Build paths to matched objects
            gallery_paths = []
            for category, object_name, distance in matches:
                # Find the actual .obj file for this object
                category_path = os.path.join(self.parent_folder, category)
                if os.path.exists(category_path):
                    for file in os.listdir(category_path):
                        if file.endswith('.obj') and file.startswith(object_name.replace('_copy', '')):
                            full_path = os.path.join(category_path, file)
                            gallery_paths.append(full_path)
                            break
            
            # Fill gallery with matched objects
            if gallery_paths:
                # Ensure we have up to 5 paths (no padding with random objects)
                gallery_paths = gallery_paths[:5]
                
                # Load into gallery viewers and clear unused ones
                for i, plotter in enumerate(self.gallery_plotters):
                    try:
                        plotter.clear()
                        if i < len(gallery_paths):
                            mesh = load(gallery_paths[i])
                            mesh.lighting('default').linecolor('black').linewidth(1)
                            plotter.show(mesh, resetcam=True)
                            print(f"Loaded similar object: {os.path.basename(gallery_paths[i])}")
                        else:
                            plotter.render()  # Clear unused viewer
                    except Exception as e:
                        if i < len(gallery_paths):
                            print(f"Error loading {gallery_paths[i]}: {e}")
                        try:
                            plotter.clear()
                            plotter.render()
                        except Exception:
                            pass
                
                print(f"Gallery loaded with {len(gallery_paths)} similar objects")
            else:
                print("No valid paths found for matches, clearing gallery")
                self._clear_gallery()
                
        except Exception as e:
            print(f"Error in similarity matching: {e}")
            self._clear_gallery()
        
        # Enforce square viewers after content is shown
        self._enforce_gallery_square_sizes()

    def load_similar_gallery_from_normalized_mesh(self, normalized_shape, rel_path: str) -> None:
        """Load 5 most similar objects into gallery viewers based on feature matching from normalized mesh."""
        if not self.feature_matcher:
            # Clear gallery if no feature matcher available
            print("No feature matcher available, clearing gallery")
            self._clear_gallery()
            return
        
        try:
            # Convert vedo mesh to trimesh object for feature extraction
            print(f"Extracting features from normalized mesh (using seed: {rel_path})...")
            
            # Get vertices and faces from vedo mesh
            vedo_mesh = normalized_shape.mesh
            vertices = vedo_mesh.vertices
            faces = vedo_mesh.cells
            
            print(f"Vedo mesh: {len(vertices)} vertices, {len(faces)} faces")
            
            # Create trimesh object
            import trimesh
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            print(f"Trimesh object: {len(trimesh_mesh.vertices)} vertices, {len(trimesh_mesh.faces)} faces")
            print(f"Trimesh is watertight: {trimesh_mesh.is_watertight}")
            
            # Use the same relative path as used when building features dictionary for deterministic RNG
            query_features = self.feature_matcher.extract_features_from_mesh(trimesh_mesh, rel_path)
            print(f"Features extracted: {query_features is not None}")
            if query_features:
                print(f"Feature keys: {list(query_features.keys())}")
                # Print histogram lengths to verify compatibility
                for hist_name in ["D1_hist", "D2_hist", "A3_hist", "D3_hist", "D4_hist"]:
                    if hist_name in query_features:
                        print(f"{hist_name} length: {len(query_features[hist_name])}")
            
            if not query_features:
                print("Failed to extract features, clearing gallery")
                self._clear_gallery()
                return
            
            # Run identity test first - check if this object matches itself in the dictionary
            self._run_identity_test(query_features, rel_path)
            
            # Find similar objects
            print("Finding similar objects...")
            matches = self.feature_matcher.find_best_matches(query_features, top_k=5)
            
            if not matches:
                print("No matches found, clearing gallery")
                self._clear_gallery()
                return
            
            print(f"Found {len(matches)} matches:")
            for i, (category, object_name, distance) in enumerate(matches):
                print(f"  {i+1}. {category}/{object_name} (distance: {distance:.4f})")
            
            # Store matches for score display
            self.current_matches = matches
            
            # Compute detailed scores for each match
            detailed_scores = []
            for category, object_name, distance in matches:
                if category in self.feature_matcher.features_dict:
                    if object_name in self.feature_matcher.features_dict[category]:
                        stored_features = self.feature_matcher.features_dict[category][object_name]
                        detailed_score = self.feature_matcher.compute_feature_distance_detailed(query_features, stored_features)
                        detailed_scores.append(detailed_score)
                    else:
                        # Fallback if object not found
                        detailed_scores.append({'D1': distance, 'D2': distance, 'A3': distance, 'D3': distance, 'D4': distance, 'Histograms': distance, 'Metrics': distance, 'Total': distance})
                else:
                    # Fallback if category not found
                    detailed_scores.append({'D1': distance, 'D2': distance, 'A3': distance, 'D3': distance, 'D4': distance, 'Histograms': distance, 'Metrics': distance, 'Total': distance})
            
            self.current_detailed_scores = detailed_scores
            
            # Build paths to matched objects
            gallery_paths = []
            gallery_scores = []
            for category, object_name, distance in matches:
                # Find the actual .obj file for this object
                category_path = os.path.join(self.parent_folder, category)
                if os.path.exists(category_path):
                    for file in os.listdir(category_path):
                        if file.endswith('.obj') and file.startswith(object_name.replace('_copy', '')):
                            full_path = os.path.join(category_path, file)
                            gallery_paths.append(full_path)
                            gallery_scores.append(distance)
                            print(f"  Found match file: {full_path}")
                            break
            
            # Fill gallery with matched objects
            if gallery_paths:
                # Ensure we have up to 5 paths (no padding with random objects)
                gallery_paths = gallery_paths[:5]
                gallery_scores = gallery_scores[:5]
                
                # Store scores for display - this is crucial!
                self.current_gallery_scores = gallery_scores[:]
                print(f"DEBUG: Stored gallery scores: {self.current_gallery_scores}")
                
                # Load into gallery viewers and clear unused ones
                for i, plotter in enumerate(self.gallery_plotters):
                    try:
                        plotter.clear()
                        if i < len(gallery_paths):
                            mesh = load(gallery_paths[i])
                            mesh.lighting('default').linecolor('black').linewidth(1)
                            plotter.show(mesh, resetcam=True)
                            print(f"Loaded similar object: {os.path.basename(gallery_paths[i])}")
                        else:
                            plotter.render()  # Clear unused viewer
                    except Exception as e:
                        if i < len(gallery_paths):
                            print(f"Error loading {gallery_paths[i]}: {e}")
                        try:
                            plotter.clear()
                            plotter.render()
                        except Exception:
                            pass
                
                # Update score labels
                self._update_gallery_score_labels()
                
                print(f"Gallery loaded with {len(gallery_paths)} similar objects")
            else:
                print("No valid paths found for matches, clearing gallery")
                self._clear_gallery()
                
        except Exception as e:
            print(f"Error in similarity matching: {e}")
            import traceback
            traceback.print_exc()
            self._clear_gallery()
        
        # Enforce square viewers after content is shown
        self._enforce_gallery_square_sizes()

    def _run_identity_test(self, query_features, rel_path: str) -> None:
        """Test if the query features match the stored features for the same object."""
        try:
            # Convert rel_path to match dictionary format (category/filename_copy)
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                category = parts[0]
                filename = parts[1]
                # Remove .obj extension and add _copy suffix to match dictionary format
                base_name = os.path.splitext(filename)[0] + '_copy'
                
                # Look for this object in the features dictionary
                if category in self.feature_matcher.features_dict:
                    if base_name in self.feature_matcher.features_dict[category]:
                        stored_features = self.feature_matcher.features_dict[category][base_name]
                        identity_distance = self.feature_matcher.compute_feature_distance(query_features, stored_features)
                        identity_detailed = self.feature_matcher.compute_feature_distance_detailed(query_features, stored_features)
                        print(f"IDENTITY TEST: Distance to self ({category}/{base_name}): {identity_distance:.6f}")
                        
                        # Store selected object score for display
                        self.current_selected_score = identity_distance
                        self.current_selected_detailed_scores = identity_detailed
                        
                        # Check histogram compatibility
                        for hist_name in ["D1_hist", "D2_hist", "A3_hist", "D3_hist", "D4_hist"]:
                            if hist_name in query_features and hist_name in stored_features:
                                q_len = len(query_features[hist_name])
                                s_len = len(stored_features[hist_name])
                                if q_len != s_len:
                                    print(f"WARNING: {hist_name} length mismatch - query: {q_len}, stored: {s_len}")
                                else:
                                    print(f"{hist_name} lengths match: {q_len}")
                    else:
                        print(f"Identity test: {base_name} not found in {category} category")
                        # Print available objects in this category for debugging
                        available = list(self.feature_matcher.features_dict[category].keys())[:5]
                        print(f"Available objects in {category}: {available}")
                        # Set selected score to None since we couldn't find identity match
                        self.current_selected_score = None
                else:
                    print(f"Identity test: Category {category} not found in features dictionary")
                    # Print available categories for debugging
                    available_cats = list(self.feature_matcher.features_dict.keys())[:10]
                    print(f"Available categories: {available_cats}")
                    self.current_selected_score = None
        except Exception as e:
            print(f"Identity test failed: {e}")
            self.current_selected_score = None

    def _update_gallery_score_labels(self) -> None:
        """Update score labels under each gallery viewer."""
        try:
            print(f"Updating score labels. Gallery scores: {getattr(self, 'current_gallery_scores', [])}")
            
            # Update selected object score with detailed breakdown
            if hasattr(self, 'current_selected_detailed_scores') and self.current_selected_detailed_scores is not None:
                scores = self.current_selected_detailed_scores
                selected_score_text = (
                    f"Selected Object: D1: {scores.get('D1', 0):.3f} | D2: {scores.get('D2', 0):.3f} | "
                    f"A3: {scores.get('A3', 0):.3f} | D3: {scores.get('D3', 0):.3f} | D4: {scores.get('D4', 0):.3f} | "
                    f"Hist: {scores.get('Histograms', 0):.3f} | Met: {scores.get('Metrics', 0):.3f} | "
                    f"Total: {scores.get('Total', 0):.3f}"
                )
                self.selected_score_label.setText(selected_score_text)
                print(f"Setting detailed selected score")
            elif hasattr(self, 'current_selected_score') and self.current_selected_score is not None:
                print(f"Setting simple selected score: {self.current_selected_score}")
                self.selected_score_label.setText(f"Selected object score: {self.current_selected_score:.6f}")
            else:
                print("No selected score available")
                self.selected_score_label.setText("Selected object score: N/A")
            
            # Update gallery viewer scores with detailed breakdown
            if hasattr(self, 'score_labels') and hasattr(self, 'current_detailed_scores'):
                for i, label in enumerate(self.score_labels):
                    if i < len(self.current_detailed_scores):
                        scores = self.current_detailed_scores[i]
                        # Format detailed score breakdown
                        score_text = (
                            f"D1: {scores.get('D1', 0):.3f}\n"
                            f"D2: {scores.get('D2', 0):.3f}\n"
                            f"A3: {scores.get('A3', 0):.3f}\n"
                            f"D3: {scores.get('D3', 0):.3f}\n"
                            f"D4: {scores.get('D4', 0):.3f}\n"
                            f"Hist: {scores.get('Histograms', 0):.3f}\n"
                            f"Met: {scores.get('Metrics', 0):.3f}\n"
                            f"Tot: {scores.get('Total', 0):.3f}"
                        )
                        label.setText(score_text)
                        print(f"Set detailed label {i}")
                    else:
                        label.setText("D1: N/A\nD2: N/A\nA3: N/A\nD3: N/A\nD4: N/A\nHist: N/A\nMet: N/A\nTot: N/A")
                        print(f"Set label {i} to: N/A")
            elif hasattr(self, 'score_labels') and hasattr(self, 'current_gallery_scores'):
                # Fallback to simple scores if detailed not available
                for i, label in enumerate(self.score_labels):
                    if i < len(self.current_gallery_scores):
                        score = self.current_gallery_scores[i]
                        label.setText(f"Total: {score:.4f}")
                        print(f"Set simple label {i} to: Total: {score:.4f}")
                    else:
                        label.setText("Total: N/A")
            else:
                print("Score labels or scores not available")
        except Exception as e:
            print(f"Error updating score labels: {e}")
            import traceback
            traceback.print_exc()

    def _clear_gallery(self) -> None:
        """Clear all gallery viewers."""
        for plotter in getattr(self, 'gallery_plotters', []):
            try:
                plotter.clear()
                plotter.render()
            except Exception:
                pass
        
        # Clear score labels
        if hasattr(self, 'score_labels'):
            for label in self.score_labels:
                label.setText("D1: N/A\nD2: N/A\nA3: N/A\nD3: N/A\nD4: N/A\nHist: N/A\nMet: N/A\nTot: N/A")
        
        if hasattr(self, 'selected_score_label'):
            self.selected_score_label.setText("Selected object score: N/A")
        
        # Clear stored scores
        self.current_gallery_scores = []
        self.current_detailed_scores = []
        self.current_selected_score = None
        self.current_selected_detailed_scores = None

    def _enforce_gallery_square_sizes(self) -> None:
        """Adjust gallery viewer heights to keep them square (height = width)."""
        if not hasattr(self, 'gallery_widgets'):
            return
        for w in self.gallery_widgets:
            try:
                w.setFixedHeight(max(0, w.width()))
            except Exception:
                pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep gallery viewers square on window resize
        self._enforce_gallery_square_sizes()

    def closeEvent(self, event) -> None:
        """Handle application close with proper cleanup."""
        try:
            # Clear the plotter and release 3D resources
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.clear()
                self.plotter.close()
            # Close gallery plotters
            if hasattr(self, 'gallery_plotters') and self.gallery_plotters:
                for p in self.gallery_plotters:
                    try:
                        p.clear()
                        p.close()
                    except Exception:
                        pass
            
            # Clear loaded shapes to free memory
            self.loaded_shapes.clear()
            self.current_mesh_actor = None
            self.bbox_actor = None
            self.bbox_labels = []
            self.origin_axes = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Cleanup temp folder
            cleanup_temp_folder()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            event.accept()


if __name__ == '__main__':
    import signal
    
    def signal_handler(signum, frame):
        """Handle system signals for clean shutdown."""
        print("Received signal, shutting down...")
        cleanup_temp_folder()
        sys.exit(0)
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)  # Ensure app quits when window is closed
    
    window = CBSRApp(SHAPEDATA_PARENT)
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Interrupted by user")
        cleanup_temp_folder()
        sys.exit(0)
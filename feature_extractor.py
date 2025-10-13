"""
Feature extraction utilities for 3D meshes.

This module defines a FeatureExtractor class that can compute a set of
geometric and statistical features from a mesh. It is designed to work
with .obj files used elsewhere in the project and returns results as a
dictionary with stable keys for later vectorization/comparison.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import trimesh


class FeatureExtractor:
    """Compute features for 3D meshes.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract_all(path_or_mesh)
        vector, names = extractor.to_vector(features)
    """

    def __init__(self) -> None:
        pass

    def _ensure_mesh(self, input_data: Union[str, trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
        if isinstance(input_data, trimesh.Trimesh):
            return input_data
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                return None
            try:
                mesh = trimesh.load_mesh(input_data, process=False)
                # Ensure triangular mesh for some properties if possible
                if hasattr(mesh, "faces") and mesh.faces is not None and mesh.faces.shape[1] != 3:
                    try:
                        mesh = mesh.triangulate()
                    except Exception:
                        pass
                return mesh
            except Exception:
                return None
        return None

    def extract_all(self, input_data: Union[str, trimesh.Trimesh]) -> Dict[str, float]:
        """Extract a baseline set of features from a mesh.

        Returns a flat dict of scalar features. NaNs/Infs are sanitized to 0.0
        to facilitate downstream vectorization and storage.
        """
        mesh = self._ensure_mesh(input_data)
        if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
            return {}

        features: Dict[str, float] = {}

        # Basic counts
        num_vertices = float(len(mesh.vertices))
        num_faces = float(len(mesh.faces)) if getattr(mesh, "faces", None) is not None else 0.0
        features["num_vertices"] = num_vertices
        features["num_faces"] = num_faces

        # Bounding box extents and ratios
        bounds = mesh.bounds
        extents = (bounds[1] - bounds[0]) if bounds is not None else np.array([0.0, 0.0, 0.0])
        ex, ey, ez = [float(x) for x in extents]
        features["bbox_extent_x"] = ex
        features["bbox_extent_y"] = ey
        features["bbox_extent_z"] = ez
        max_extent = max(ex, ey, ez, 1e-12)
        features["bbox_aspect_xy"] = (ex / max(ey, 1e-12))
        features["bbox_aspect_xz"] = (ex / max(ez, 1e-12))
        features["bbox_aspect_yz"] = (ey / max(ez, 1e-12))
        features["bbox_max_extent"] = max_extent

        # Surface area and volume (volume may be 0 for open meshes)
        try:
            features["surface_area"] = float(mesh.area)
        except Exception:
            features["surface_area"] = 0.0
        try:
            features["volume"] = float(mesh.volume) if mesh.is_volume else 0.0
        except Exception:
            features["volume"] = 0.0

        # PCA eigenvalues on vertex positions
        try:
            cov = np.cov(mesh.vertices.T)
            evals, _ = np.linalg.eigh(cov)
            evals = np.sort(np.maximum(evals, 0.0))[::-1]
            # Pad to length 3 if degenerate
            if evals.shape[0] < 3:
                evals = np.pad(evals, (0, 3 - evals.shape[0]), mode="constant")
            features["pca_eig_1"] = float(evals[0])
            features["pca_eig_2"] = float(evals[1])
            features["pca_eig_3"] = float(evals[2])
            denom = max(evals[0], 1e-12)
            features["pca_eig2_over_eig1"] = float(evals[1] / denom)
            features["pca_eig3_over_eig1"] = float(evals[2] / denom)
        except Exception:
            features["pca_eig_1"] = 0.0
            features["pca_eig_2"] = 0.0
            features["pca_eig_3"] = 0.0
            features["pca_eig2_over_eig1"] = 0.0
            features["pca_eig3_over_eig1"] = 0.0

        # Mean/variance of edge lengths
        try:
            # Build unique edges from faces
            faces = mesh.faces if getattr(mesh, "faces", None) is not None else np.empty((0, 3), dtype=int)
            edges = np.vstack({tuple(sorted(e)) for e in np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0)}) if faces.size else np.empty((0, 2), dtype=int)
            if edges.size:
                edge_vecs = mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]
                edge_lengths = np.linalg.norm(edge_vecs, axis=1)
                features["edge_length_mean"] = float(np.mean(edge_lengths))
                features["edge_length_std"] = float(np.std(edge_lengths))
            else:
                features["edge_length_mean"] = 0.0
                features["edge_length_std"] = 0.0
        except Exception:
            features["edge_length_mean"] = 0.0
            features["edge_length_std"] = 0.0

        # Sanitize NaN/Inf
        for k, v in list(features.items()):
            if not np.isfinite(v):
                features[k] = 0.0

        return features

    def feature_names(self) -> List[str]:
        """Stable ordering of feature names for vectorization."""
        return [
            "num_vertices",
            "num_faces",
            "bbox_extent_x",
            "bbox_extent_y",
            "bbox_extent_z",
            "bbox_aspect_xy",
            "bbox_aspect_xz",
            "bbox_aspect_yz",
            "bbox_max_extent",
            "surface_area",
            "volume",
            "pca_eig_1",
            "pca_eig_2",
            "pca_eig_3",
            "pca_eig2_over_eig1",
            "pca_eig3_over_eig1",
            "edge_length_mean",
            "edge_length_std",
        ]

    def to_vector(self, features: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """Convert a feature dict to a vector using stable feature_names order."""
        names = self.feature_names()
        vec = np.array([float(features.get(name, 0.0)) for name in names], dtype=float)
        return vec, names



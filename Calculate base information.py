import trimesh
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess_mesh(mesh):
    if not mesh.is_watertight:
        print("Mesh is not watertight, attempting to fill holes.")
        trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh, multibody=False)
    trimesh.repair.fix_winding(mesh)
    return mesh


def convex_hull_with_n_points(mesh, target_points=5000, max_subdiv=5):
    # Subdivide until reaching or exceeding the target number of vertices
    subdivided = mesh.copy()
    for _ in range(max_subdiv):
        if len(subdivided.vertices) >= target_points:
            break
        subdivided = subdivided.subdivide()
    # Optionally, randomly sample if there are too many points
    if len(subdivided.vertices) > target_points:
        idx = np.random.choice(len(subdivided.vertices), target_points, replace=False)
        sampled_vertices = subdivided.vertices[idx]
        # Create a new mesh with sampled vertices (faces are ignored)
        hull = trimesh.convex.convex_hull(sampled_vertices)
    else:
        hull = subdivided.convex_hull

    trimesh.repair.fix_normals(hull, multibody=False)
    trimesh.repair.fix_winding(hull)
    return hull

def mesh_volume(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # Compute the signed volume of each tetrahedron (triangle + origin)
    cross = np.cross(v0, v1)
    dot = np.einsum('ij,ij->i', cross, v2)
    volume = np.sum(dot) / 6.0
    return abs(volume)

def compute_metrics_from_convex_hull(mesh):
    original_volume = mesh.volume
    # Compute convex hull
    hull = mesh.convex_hull
    trimesh.repair.fix_winding(hull)

    volume = hull.volume
    area = hull.area
    diameter = np.linalg.norm(hull.extents)

    # Compactness (with respect to a sphere)
    compactness = (area ** 3) / (36 * np.pi * (volume ** 2)) if volume > 0 else np.nan

    # Sphericity (reciprocal of compactness)
    sphericity = 1 / compactness if compactness > 0 else np.nan

    # 3D Rectangularity (shape volume divided by OBB volume)
    try:
        obb_volume = hull.bounding_box_oriented.volume
        rectangularity = volume / obb_volume if obb_volume > 0 else np.nan
    except Exception:
        rectangularity = np.nan

    # Convexity (shape volume divided by convex hull volume, always 1 for convex hull)
    convexity = original_volume /volume if volume > 0 else np.nan

    # Eccentricity (ratio of largest to smallest eigenvalues of covariance matrix)
    moments = hull.principal_inertia_components
    eccentricity = moments[0] / moments[2] if moments[2] > 1e-6 else np.nan

    return {
        "Mesh volume": volume,
        "Surface area": area,
        "Diameter": diameter,
        "Compactness": compactness,
        "Rectangularity": rectangularity,
        "Convexity": convexity,
        "Eccentricity": eccentricity,
        "Sphericity": sphericity,
        "extents": hull.extents,
    }

def compute_metrics(mesh, original_mesh):
    volume = mesh.volume
    area = mesh.area
    diameter = np.linalg.norm(mesh.extents)
    try:
        original_volume = original_mesh.volume
        hull_volume = original_mesh.convex_hull.volume
        convexity = original_volume / hull_volume if hull_volume > 0 else np.nan
    except Exception:
        convexity = np.nan
    moments = mesh.principal_inertia_components
    eccentricity = moments[0] / moments[2] if moments[2] > 1e-6 else np.nan
    compactness = (area ** 3) / (36 * np.pi * (volume ** 2)) if volume > 0 else np.nan #(6 * np.sqrt(np.pi) * volume) / (area ** 1.5) if area > 0 else np.nan
    sphericity = 1 / compactness if compactness > 0 else np.nan

    try:
        obb_volume = mesh.bounding_box_oriented.volume
        rectangularity = volume / obb_volume if obb_volume > 0 else np.nan
    except ValueError:
        rectangularity = np.nan

    return {
        "Mesh volume": volume,
        "Surface area": area,
        "Diameter": diameter,
        "Compactness": compactness,
        "Rectangularity": rectangularity,
        "Convexity": convexity,
        "Eccentricity": eccentricity,
        "Sphericity": sphericity,
        "extents": mesh.extents,
    }

def d2_descriptor(vertices, n_samples):
    n_vertices = len(vertices)
    idx1 = np.random.randint(0, n_vertices, n_samples)
    idx2 = np.random.randint(0, n_vertices, n_samples)
    return np.linalg.norm(vertices[idx1] - vertices[idx2], axis=1)

def d1_descriptor(vertices, n_samples):
    barycenter = np.mean(vertices, axis=0)
    n_vertices = len(vertices)
    idx = np.random.randint(0, n_vertices, n_samples)
    return np.linalg.norm(vertices[idx] - barycenter, axis=1)

def a3_descriptor(vertices, n_samples):
    n_vertices = len(vertices)
    idx1 = np.random.randint(0, n_vertices, n_samples)
    idx2 = np.random.randint(0, n_vertices, n_samples)
    idx3 = np.random.randint(0, n_vertices, n_samples)
    v1, v2, v3 = vertices[idx1], vertices[idx2], vertices[idx3]
    vec1, vec2 = v1 - v2, v3 - v2
    norms1, norms2 = np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1)
    dot_product = np.sum(vec1 * vec2, axis=1)
    epsilon = 1e-8
    cos_angle = np.clip(dot_product / (norms1 * norms2 + epsilon), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def d3_descriptor(vertices, n_samples):
    n_vertices = len(vertices)
    idx1 = np.random.randint(0, n_vertices, n_samples)
    idx2 = np.random.randint(0, n_vertices, n_samples)
    idx3 = np.random.randint(0, n_vertices, n_samples)
    v1, v2, v3 = vertices[idx1], vertices[idx2], vertices[idx3]
    a = np.linalg.norm(v2 - v1, axis=1)
    b = np.linalg.norm(v3 - v2, axis=1)
    c = np.linalg.norm(v1 - v3, axis=1)
    s = (a + b + c) / 2
    area = np.sqrt(np.abs(s * (s - a) * (s - b) * (s - c)))
    return np.sqrt(area)

def d4_descriptor(vertices, n_samples):
    n_vertices = len(vertices)
    idx1 = np.random.randint(0, n_vertices, n_samples)
    idx2 = np.random.randint(0, n_vertices, n_samples)
    idx3 = np.random.randint(0, n_vertices, n_samples)
    idx4 = np.random.randint(0, n_vertices, n_samples)
    v1, v2, v3, v4 = vertices[idx1], vertices[idx2], vertices[idx3], vertices[idx4]
    vec_a = v2 - v1
    vec_b = v3 - v1
    vec_c = v4 - v1
    volumes = np.abs(np.einsum('ij,ij->i', vec_a, np.cross(vec_b, vec_c))) / 6.0
    return np.cbrt(volumes)

def plot_histogram(data, bins, color, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

def save_features_for_all_objects_txt(
    base_dir='ShapeDatabase_INFOMR-master/after_remeshing_normalise',
    features_dir='ShapeDatabase_INFOMR-master/features_3',
    n_samples=10000,
    bins_dict={'D1': 10, 'D2': 10, 'A3': 10, 'D3': 10, 'D4': 10}
):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.lower().endswith('.obj'):
                continue
            obj_path = os.path.join(root, file)
            rel_path = os.path.relpath(obj_path, base_dir)
            feature_path = os.path.join(features_dir, os.path.splitext(rel_path)[0] + '.txt')
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)

            try:
                mesh = trimesh.load(obj_path)
                #preprocessed = preprocess_mesh(mesh.copy())
                # hull = convex_hull_with_n_points(mesh.copy())
                metrics = compute_metrics_from_convex_hull(mesh)
                vertices = mesh.vertices

                d1 = d1_descriptor(vertices, n_samples)
                d2 = d2_descriptor(vertices, n_samples)
                a3 = a3_descriptor(vertices, n_samples)
                d3 = d3_descriptor(vertices, n_samples)
                d4 = d4_descriptor(vertices, n_samples)

                # Compute histograms
                d1_hist, _ = np.histogram(d1, bins=bins_dict['D1'])
                d2_hist, _ = np.histogram(d2, bins=bins_dict['D2'])
                a3_hist, _ = np.histogram(a3, bins=bins_dict['A3'])
                d3_hist, _ = np.histogram(d3, bins=bins_dict['D3'])
                d4_hist, _ = np.histogram(d4, bins=bins_dict['D4'])

                with open(feature_path, 'w') as f:
                    f.write('Metrics:\n')
                    for k, v in metrics.items():
                        f.write(f'{k}: {v}\n')
                    f.write('\nD1_hist:\n' + ','.join(map(str, d1_hist)) + '\n')
                    f.write('\nD2_hist:\n' + ','.join(map(str, d2_hist)) + '\n')
                    f.write('\nA3_hist:\n' + ','.join(map(str, a3_hist)) + '\n')
                    f.write('\nD3_hist:\n' + ','.join(map(str, d3_hist)) + '\n')
                    f.write('\nD4_hist:\n' + ','.join(map(str, d4_hist)) + '\n')

                print(f"Saved features for {obj_path} to {feature_path}")
            except Exception as e:
                print(f"Failed for {obj_path}: {e}")

def test():
    mesh_path = 'ShapeDatabase_INFOMR-master/after_remeshing_normalise/Tree/D00096_copy.obj'
    original_mesh = trimesh.load(mesh_path)
    preprocessed_mesh = preprocess_mesh(original_mesh.copy())

    # Compute metrics on convex hull of preprocessed mesh
    hull_mesh = convex_hull_with_n_points(preprocessed_mesh.copy())
    metrics = compute_metrics(hull_mesh, preprocessed_mesh)
    print(mesh_volume(hull_mesh.vertices, hull_mesh.faces))
    print("\n--- Mesh Properties ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Compute descriptors on preprocessed mesh
    vertices = preprocessed_mesh.vertices
    n_samples = 10000

    d2 = d2_descriptor(vertices, n_samples)
    plot_histogram(d2, 10, 'skyblue', 'Histogram Distance Between 2 Random Vertices', 'Distance', 'Frequency')

    d1 = d1_descriptor(vertices, n_samples)
    plot_histogram(d1, 10, 'orange', 'Histogram of Distance Between Barycenter and Random Vertex', 'Distance', 'Frequency')

    a3 = a3_descriptor(vertices, n_samples)
    plot_histogram(a3, 10, 'lightgreen', 'Histogram of Angle Between 3 Random Vertices', 'Angle (Degrees)', 'Frequency')

    d3 = d3_descriptor(vertices, n_samples)
    plot_histogram(d3, 10, 'violet', 'Histogram of Sqrt Area of Triangle from 3 Random Vertices', 'Sqrt(Area)', 'Frequency')

    d4 = d4_descriptor(vertices, n_samples)
    plot_histogram(d4, 10, 'teal', 'Histogram of Cube Root of Tetrahedron Volume from 4 Random Vertices', 'Cube Root of Volume', 'Frequency')

    plt.show()


if __name__ == "__main__":
    save_features_for_all_objects_txt()
    #test()



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
    compactness = (6 * np.sqrt(np.pi) * volume) / (area ** 1.5) if area > 0 else np.nan
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
        "Eccentricity": eccentricity
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

def main():
    mesh_path = 'ShapeDatabase_INFOMR-master/Normalised_models/ClassicPiano/D00009.obj'
    original_mesh = trimesh.load(mesh_path)
    mesh = preprocess_mesh(original_mesh.copy())
    metrics = compute_metrics(mesh, original_mesh)
    print("\n--- Mesh Properties ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    vertices = mesh.vertices
    n_samples = 10000

    d2 = d2_descriptor(vertices, n_samples)
    plot_histogram(d2, 15, 'skyblue', 'Histogram Distance Between 2 Random Vertices', 'Distance', 'Frequency')

    d1 = d1_descriptor(vertices, n_samples)
    plot_histogram(d1, 20, 'orange', 'Histogram of Distance Between Barycenter and Random Vertex', 'Distance', 'Frequency')

    a3 = a3_descriptor(vertices, n_samples)
    plot_histogram(a3, 15, 'lightgreen', 'Histogram of Angle Between 3 Random Vertices', 'Angle (Degrees)', 'Frequency')

    d3 = d3_descriptor(vertices, n_samples)
    plot_histogram(d3, 20, 'violet', 'Histogram of Sqrt Area of Triangle from 3 Random Vertices', 'Sqrt(Area)', 'Frequency')

    d4 = d4_descriptor(vertices, n_samples)
    plot_histogram(d4, 20, 'teal', 'Histogram of Cube Root of Tetrahedron Volume from 4 Random Vertices', 'Cube Root of Volume', 'Frequency')

    plt.show()

def find_most_similar_objects(query_path, models_dir, top_n_metrics=100, top_n_final=5, n_samples=10000):
    def metrics_to_vec(metrics):
        return np.array([metrics[k] for k in sorted(metrics.keys()) if np.isfinite(metrics[k])])

    print("Loading query mesh and computing metrics...")
    query_mesh = trimesh.load(query_path, process=False)
    query_metrics = compute_metrics(query_mesh, query_mesh)
    query_vec = metrics_to_vec(query_metrics)

    print("Stage 1: Scanning all objects and computing metrics...")
    obj_paths, metrics_vecs = [], []
    for class_dir in os.listdir(models_dir):
        class_path = os.path.join(models_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for obj_file in os.listdir(class_path):
            if not obj_file.lower().endswith('.obj'):
                continue
            obj_path = os.path.join(class_path, obj_file)
            try:
                mesh = trimesh.load(obj_path, process=False)
                metrics = compute_metrics(mesh, mesh)
                vec = metrics_to_vec(metrics)
                if len(vec) == len(query_vec):
                    obj_paths.append(obj_path)
                    metrics_vecs.append(vec)
            except Exception:
                continue
    metrics_vecs = np.array(metrics_vecs)
    dists = np.linalg.norm(metrics_vecs - query_vec, axis=1)
    top_idx = np.argsort(dists)[:top_n_metrics]
    top_candidates = [obj_paths[i] for i in top_idx]

    print("Stage 2: Computing descriptors for top candidates...")
    def descriptor_vector(mesh):
        v = mesh.vertices
        d1 = d1_descriptor(v, n_samples)
        d2 = d2_descriptor(v, n_samples)
        a3 = a3_descriptor(v, n_samples)
        d3 = d3_descriptor(v, n_samples)
        d4 = d4_descriptor(v, n_samples)
        def hist(x, bins): return np.histogram(x, bins=bins, range=(np.nanmin(x), np.nanmax(x)))[0]
        return np.concatenate([
            hist(d1, 20), hist(d2, 15), hist(a3, 15), hist(d3, 20), hist(d4, 20)
        ]).astype(float)

    query_desc = descriptor_vector(query_mesh)
    desc_dists = []
    for i, obj_path in enumerate(top_candidates):
        print(f"Processing descriptor {i+1}/{len(top_candidates)}: {obj_path}")
        try:
            mesh = trimesh.load(obj_path, process=False)
            desc = descriptor_vector(mesh)
            if len(desc) == len(query_desc):
                dist = np.linalg.norm(query_desc - desc)
                desc_dists.append((dist, obj_path))
        except Exception:
            continue
    desc_dists.sort()
    print("Done! Returning most similar objects.")
    return [p for _, p in desc_dists[:top_n_final]]

#similar_objs = find_most_similar_objects(
#     'ShapeDatabase_INFOMR-master/Normalised_models/ClassicPiano/D00009.obj',
#     'ShapeDatabase_INFOMR-master/Normalised_models'
#)

#print("\nMost similar objects:")
#for i, obj_path in enumerate(similar_objs, 1):
#    print(f"{i}: {obj_path}")
if __name__ == "__main__":
    main()

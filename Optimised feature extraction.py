import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

NORMALISED_MODELS_DIR = 'ShapeDatabase_INFOMR-master/Normalised_models'
PROCESSED_FEATURES_DIR = 'ShapeDatabase_INFOMR-master/Processed features'
N_SAMPLES = 10000

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_metrics(metrics, out_path):
    with open(out_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

def plot_and_save_histogram(data, bins, color, title, xlabel, ylabel, out_path):
    # Remove NaN and inf values
    data = data[np.isfinite(data)]
    if data.size == 0:
        print(f"Warning: No valid data for {title}, skipping histogram.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def calculate_descriptors(vertices, n_samples):
    n_vertices = len(vertices)
    rng = np.random.default_rng()
    idxs = rng.integers(0, n_vertices, size=(4, n_samples))

    # D1
    barycenter = np.mean(vertices, axis=0)
    d1 = np.linalg.norm(vertices[idxs[0]] - barycenter, axis=1)

    # D2
    d2 = np.linalg.norm(vertices[idxs[0]] - vertices[idxs[1]], axis=1)

    # A3
    v1, v2, v3 = vertices[idxs[0]], vertices[idxs[1]], vertices[idxs[2]]
    vec1, vec2 = v1 - v2, v3 - v2
    norms1, norms2 = np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1)
    dot_product = np.sum(vec1 * vec2, axis=1)
    epsilon = 1e-8
    cos_angle = np.clip(dot_product / (norms1 * norms2 + epsilon), -1.0, 1.0)
    a3 = np.degrees(np.arccos(cos_angle))

    # D3
    a = np.linalg.norm(v2 - v1, axis=1)
    b = np.linalg.norm(v3 - v2, axis=1)
    c = np.linalg.norm(v1 - v3, axis=1)
    s = (a + b + c) / 2
    area = np.sqrt(np.abs(s * (s - a) * (s - b) * (s - c)))
    d3 = np.sqrt(area)

    # D4
    v4 = vertices[idxs[3]]
    vec_a = v2 - v1
    vec_b = v3 - v1
    vec_c = v4 - v1
    volumes = np.abs(np.einsum('ij,ij->i', vec_a, np.cross(vec_b, vec_c))) / 6.0
    d4 = np.cbrt(volumes)

    return d1, d2, a3, d3, d4

def process_obj(obj_path, out_dir):
    mesh = trimesh.load(obj_path)
    if mesh.vertices.shape[0] < 4:
        print(f"Skipping {obj_path}: not enough vertices.")
        return
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh, multibody=False)
    trimesh.repair.fix_winding(mesh)

    # Metrics
    volume = mesh.volume
    area = mesh.area
    diameter = np.linalg.norm(mesh.extents)
    try:
        hull_volume = mesh.convex_hull.volume
        convexity = volume / hull_volume if hull_volume > 0 else np.nan
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

    metrics = {
        "Mesh volume": volume,
        "Surface area": area,
        "Diameter": diameter,
        "Compactness": compactness,
        "Rectangularity": rectangularity,
        "Convexity": convexity,
        "Eccentricity": eccentricity
    }
    save_metrics(metrics, os.path.join(out_dir, 'metrics.txt'))

    # Descriptors (vectorized)
    vertices = mesh.vertices
    d1, d2, a3, d3, d4 = calculate_descriptors(vertices, N_SAMPLES)

    # Save histograms in parallel
    with ThreadPoolExecutor() as executor:
        executor.submit(plot_and_save_histogram, d1, 20, 'orange', 'Histogram of D1', 'Distance', 'Frequency', os.path.join(out_dir, 'D1_hist.png'))
        executor.submit(plot_and_save_histogram, d2, 15, 'skyblue', 'Histogram of D2', 'Distance', 'Frequency', os.path.join(out_dir, 'D2_hist.png'))
        executor.submit(plot_and_save_histogram, a3, 15, 'lightgreen', 'Histogram of A3', 'Angle (Degrees)', 'Frequency', os.path.join(out_dir, 'A3_hist.png'))
        executor.submit(plot_and_save_histogram, d3, 20, 'violet', 'Histogram of D3', 'Sqrt(Area)', 'Frequency', os.path.join(out_dir, 'D3_hist.png'))
        executor.submit(plot_and_save_histogram, d4, 20, 'teal', 'Histogram of D4', 'Cube Root of Volume', 'Frequency', os.path.join(out_dir, 'D4_hist.png'))

def main():
    for class_dir in os.listdir(NORMALISED_MODELS_DIR):
        class_path = os.path.join(NORMALISED_MODELS_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue
        for obj_file in os.listdir(class_path):
            if obj_file.lower().endswith('.obj'):
                obj_path = os.path.join(class_path, obj_file)
                obj_name = os.path.splitext(obj_file)[0]
                out_dir = os.path.join(PROCESSED_FEATURES_DIR, obj_name)
                ensure_dir(out_dir)
                print(f"Processing {obj_path} -> {out_dir}")
                process_obj(obj_path, out_dir)

if __name__ == "__main__":
    main()

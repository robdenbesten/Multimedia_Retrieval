import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Keep original mesh for convexity metric
original_mesh = trimesh.load('ShapeDatabase_INFOMR-master/Original Database/Fish/D00012.obj')

# Create a working copy for modifications
mesh = original_mesh.copy()

# --- Pre-processing ---, kijken of het ook zonder kan want ik ben hier geen fan  van.

# Fill holes if not watertight
if not mesh.is_watertight:
    print("Mesh is not watertight, attempting to fill holes.")
    mesh.fill_holes()

# If still not watertight, use convex hull as a fallback
if not mesh.is_watertight:
    print("Filling holes failed. Using the convex hull as a fallback.")
    mesh = mesh.convex_hull

# --- Metric Calculations ---

# Calculate properties from the consistent mesh
volume = mesh.volume
area = mesh.area

# 1. Diameter (AABB diagonal)
diameter = np.linalg.norm(mesh.extents)

# 2. Convexity (original volume / convex hull volume)
try:
    original_volume = original_mesh.volume
    hull_volume = original_mesh.convex_hull.volume
    if hull_volume > 0:
        convexity = original_volume / hull_volume
    else:
        convexity = np.nan
except Exception:
    convexity = np.nan

# 3. Eccentricity (ratio of inertia tensor eigenvalues) --- > klopt nog niet denk ik
moments = mesh.principal_inertia_components
if moments[2] > 1e-6:  # Avoid division by zero for planar/linear shapes
    eccentricity = moments[0] / moments[2]
else:
    eccentricity = np.nan

# Compactness (wrt sphere)
if area > 0:
    compactness = (6 * np.sqrt(np.pi) * volume) / (area ** 1.5)
else:
    compactness = np.nan

# Rectangularity
try:
    obb_volume = mesh.bounding_box_oriented.volume
    if obb_volume > 0:
        rectangularity = volume / obb_volume
    else:
        rectangularity = np.nan
except ValueError:
    print("Could not compute OBB, likely because the mesh is planar.")
    rectangularity = np.nan


# --- Output Results ---

print(f"\n--- Mesh Properties ---")
print(f"Mesh volume: {volume}")
print(f"Surface area: {area}")
print(f"Diameter: {diameter}")
print(f"Compactness: {compactness}")
print(f"Rectangularity: {rectangularity}")
print(f"Convexity: {convexity}")
print(f"Eccentricity: {eccentricity}")



# --- D2 Descriptor Histogram ---

vertices = mesh.vertices
n_vertices = len(vertices)
n_samples = 10000  # Number of random pairs

# Generate random indices for vertex pairs
rand_indices1 = np.random.randint(0, n_vertices, n_samples)
rand_indices2 = np.random.randint(0, n_vertices, n_samples)

# Get vertex pairs
p1 = vertices[rand_indices1]
p2 = vertices[rand_indices2]

# Compute D2 distances
d2_distances = np.linalg.norm(p1 - p2, axis=1)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(d2_distances, bins=25, color='skyblue', edgecolor='black')
plt.title('Histogram of D2 (Distance Between 2 Random Vertices)')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

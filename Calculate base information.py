import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the mesh
# It's good practice to keep the original mesh for metrics like convexity
original_mesh = trimesh.load('ShapeDatabase_INFOMR-master/Original Database/Fish/D00012.obj')

# Create a working copy of the mesh for modifications
mesh = original_mesh.copy()

# --- Pre-processing and Fallback Logic ---

# If the mesh is not watertight, try to fill holes.
if not mesh.is_watertight:
    print("Mesh is not watertight, attempting to fill holes.")
    mesh.fill_holes()

# If it's still not watertight, fall back to using its convex hull for subsequent calculations.
if not mesh.is_watertight:
    print("Filling holes failed. Using the convex hull as a fallback.")
    mesh = mesh.convex_hull

# --- Metric Calculations ---

# Now we have a single, consistent, watertight mesh (`mesh`) to calculate most properties from.
volume = mesh.volume
area = mesh.area

# 1. Diameter: The length of the diagonal of the mesh's axis-aligned bounding box.
diameter = np.linalg.norm(mesh.extents)

# 2. Convexity: Calculated using the original mesh volume vs. its convex hull volume.
try:
    original_volume = original_mesh.volume
    hull_volume = original_mesh.convex_hull.volume
    if hull_volume > 0:
        convexity = original_volume / hull_volume
    else:
        convexity = np.nan
except Exception:
    convexity = np.nan

# 3. Eccentricity: Ratio of largest to smallest eigenvalues of the inertia tensor.
# These components eigenvalues of the tensorensor.
moments = mesh.principal_inertia_components
if moments[2] > 1e-6:  # Avoid division by zero for planar/linear shapes
    eccentricity = moments[0] / moments[2]
else:
    eccentricity = np.nan

# Calculate compactness with respect to a sphere
if area > 0:
    compactness = (6 * np.sqrt(np.pi) * volume) / (area ** 1.5)
else:
    compactness = np.nan

# Calculate 3D rectangularity
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
print(f"Diameter (AABB diagonal): {diameter}")
print(f"Compactness (wrt sphere): {compactness}")
print(f"Rectangularity: {rectangularity}")
print(f"Convexity: {convexity}")
print(f"Eccentricity: {eccentricity}")



# --- D2 Descriptor Histogram ---

# Get vertex coordinates
vertices = mesh.vertices
n_vertices = len(vertices)
n_samples = 10000  # Number of random pairs to sample

# Generate random indices for pairs of vertices
rand_indices1 = np.random.randint(0, n_vertices, n_samples)
rand_indices2 = np.random.randint(0, n_vertices, n_samples)

# Get the vertex pairs
p1 = vertices[rand_indices1]
p2 = vertices[rand_indices2]

# Compute the D2 distances
d2_distances = np.linalg.norm(p1 - p2, axis=1)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(d2_distances, bins=25, color='skyblue', edgecolor='black')
plt.title('Histogram of D2 (Distance Between 2 Random Vertices)')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
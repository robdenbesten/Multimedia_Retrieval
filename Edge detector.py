import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the mesh from an OBJ file
mesh = trimesh.load('ShapeDatabase_INFOMR-master/Original Database/AircraftBuoyant/m1337.obj')

print(f"Mesh volume: {mesh.volume}")

# Get unique edges as pairs of vertex indices
edges = mesh.edges_unique

# Get the vertex coordinates
vertices = mesh.vertices

# Compute edge lengths
edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(edge_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Edge Lengths')
plt.xlabel('Edge Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

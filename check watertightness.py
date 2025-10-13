import trimesh
import os
import alphashape
import numpy as np

base_dir = r'C:\Users\thies\PycharmProjects\Multimedia_Retrieval\ShapeDatabase_INFOMR-master\after_remeshing_normalise\Gun'

mesh = None
for file in os.listdir(base_dir):
    if file.lower().endswith('.obj'):
        mesh_path = os.path.join(base_dir, file)
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            print(f"Loaded mesh: {mesh_path}")
            break
        except Exception as e:
            print(f"Error loading {file}: {e}")

if mesh is None:
    print("No .obj mesh found in the folder.")
    exit()

# Compute the concave hull (alpha shape)
alpha = 0.15  # Adjust as needed
alpha_shape = alphashape.alphashape(mesh.vertices, alpha)

if hasattr(alpha_shape, 'triangles'):
    concave_hull = trimesh.Trimesh(vertices=np.array(alpha_shape.vertices), faces=np.array(alpha_shape.faces))
    print("Showing concave hull...")
    concave_hull.show()
else:
    print("Could not compute a valid concave hull for this mesh.")

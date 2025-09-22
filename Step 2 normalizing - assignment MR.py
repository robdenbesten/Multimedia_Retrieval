import trimesh
import math

def normalising_mesh(mesh):
    mesh = trimesh.load_mesh(mesh)
    print(math.dist([0,0,0], mesh.centroid))
    print(mesh.bounds[1] - mesh.bounds[0])
    centralized = mesh.centroid
    mesh.apply_translation(-centralized)
    afbaken_box = mesh.bounds
    grootte = afbaken_box[1] - afbaken_box[0]
    langste = grootte.max()
    scale = 1/langste
    mesh.apply_scale(scale)
    print(math.dist([0, 0, 0], mesh.centroid))
    print(mesh.bounds[1] - mesh.bounds[0])
    mesh.show()

normalising_mesh("bottle.obj")
import numpy as np
#import pymeshlab
#ms = pymeshlab.MeshSet()
#ms.load_new_mesh('bottle.obj')

import vedo
mesh = vedo.load('D00921.obj')

#add the visible cells
mesh.force_opaque().linewidth(1)

vedo.show(mesh)




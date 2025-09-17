import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

import FEA

os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

inp = FEA.FEA_INP()

from matplotlib import pyplot as plt

inp_file = os.path.join(current_path, '_forshellgenerate.inp')
inp.Read_INP(inp_file)

fe = FEA.from_inp(inp)
fe.initialize()
a = fe.surface_sets.get_elements('surface_1_All')[0]
normal = a.get_gaussian_normal(fe.nodes)[0]
r = a.gaussian_points_position(fe.nodes)[0]

coo = a._elems.cpu().numpy()

from mayavi import mlab

mlab.figure()
mlab.quiver3d(r[:, 0], r[:, 1], r[:, 2], normal[:, 0], normal[:, 1], normal[:, 2])
mlab.triangular_mesh(fe.nodes[:, 0], fe.nodes[:, 1], fe.nodes[:, 2], coo, opacity=0.3)
mlab.show()
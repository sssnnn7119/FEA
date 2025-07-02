from ..Main import FEA_Main
from mayavi import mlab



class Visualizator:
    def __init__(self, mesh, results):
        self.mesh = mesh
        self.results = results

    
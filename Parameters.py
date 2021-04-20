import numpy as np

class Parameters:     
    def __init__(self):
        # Energy differences
        self.eps_N = -6
        self.eps_G = -7
        self.eps_S = -2
        self.eps_NS = -2
        #self.eps_S = -3
        #self.eps_Sb = -4

        # Decay rates
        self.gamma_N = 100
        self.gamma_G = 100
        self.gamma_S = 100

        # Reproduction rates
        self.r_N = 10
        self.r_G = 10
        self.r_S = 10

        # Time relevant parameters
        self.T = 24
        self.nofSteps = 3000
        self.dt = self.T/self.nofSteps

        # Signal parameters
        self.range = 0.5
        self.D = 20 / 25
        #self.production = 20
        #self.uptake = 1e-3
        self.signal = 'nonlocal'

        # Newton's method parameters (currently no longer relevant)
        self.Maxit = 20
        self.Tol = 1e-5

        # Organoid growth parameters
        self.nofCells_start = 9                    # Number of cells in the beginning
        self.nofCells_end = 400                     # Approximate number of cells in the end
        self.rmax = 1                               # Maximum cell radius
        self.alpha = 3                              # Cell stiffness
        self.sigma = 0.7                            # Ratio of cell radius with balanced force between adhesion/repulsion
        self.F0 = 0.1                               # Adhesion/repulsion scaling
        #self.relSpeed = 10                        # Speed of transcription relative to division/motion

def setParameters():
    self = Parameters()
    self.__init__()

    return self
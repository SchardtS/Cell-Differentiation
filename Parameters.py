class Parameters:     
    def __init__(self):
        # Energy differences
        self.eps_N = -3
        self.eps_G = -4
        self.eps_A = -1
        self.eps_NA = -1
        self.eps_S = -3.4
        self.eps_Sb = -4

        # Decay rates
        self.gamma_N = 1
        self.gamma_G = 1
        self.gamma_S = 1

        # Signal parameters
        self.intensity = 1
        self.signal = 'local'

        # Time relevant parameters
        self.T = 24
        self.nofSteps = 3000
        self.dt = self.T/self.nofSteps

        # Newton's method parameters
        self.Maxit = 20
        self.Tol = 1e-5

        # Organoid growth parameters
        self.nofCells_start = 10                    # Number of cells in the beginning
        self.nofCells_end = 50                      # Approximate number of cells in the end
        self.rmax = 1                               # Maximum cell radius
        self.alpha = 5                              # Cell stiffness
        self.sigma = 0.7                            # Ratio of cell radius with balanced force between adhesion/repulsion
        self.F0 = 1                                # Adhesion/repulsion scaling
        self.relSpeed = 20                          # Speed of transcription relative to division/motion

def setParameters():
    self = Parameters()
    self.__init__()

    return self
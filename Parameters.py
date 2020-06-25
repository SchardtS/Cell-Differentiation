class Parameters:     
    def __init__(self):
        # Energy differences
        self.eps_N = -3
        self.eps_G = -3.5
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
        self.nofSteps = 100
        self.dt = self.T/self.nofSteps

        # Newton's method parameters
        self.Maxit = 20
        self.Tol = 1e-5

        # Organoid growth parameters
        self.nofCells_start = 10
        self.nofCells_end = 200
        self.rmax = 1
        self.sigma = 10
        self.alpha = 2
        self.F0 = 0.5
        self.relSpeed = 20

def setParameters():
    self = Parameters()
    self.__init__()

    return self
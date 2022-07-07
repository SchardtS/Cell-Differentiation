class Parameters:
    def __init__(self):
        #### time parameters ####
        self.T = 24                               # Time
        self.nofSteps = 5000                      # Number of timesteps
              
        #### geometry parameters ####
        self.r_max = 1                             # Maximum radius
        self.k = 0.083                               # Cell growth rate
        self.F0 = 0.01                             # Force scaling
        self.alpha = 3                             # Cell stiffness
        self.sigma = 0.7                           # Don't know how to call that one
        
        #### transcription parameters ####
        # energy differences
        self.eps_N = -7.32
        self.eps_G = -6
        self.eps_S = -2
        self.eps_NS = -2
        self.eps_GS = -2

        # decay rates
        self.gamma_N = 10
        self.gamma_G = 10
        self.gamma_S = 10

        # reproduction rates
        self.r_N = 1
        self.r_G = 1
        self.r_S = 1
        
        # signal parameters
        self.q = 0.7
        self.signal = 'neighbor'                 # 'neighbor' or 'dispersion'
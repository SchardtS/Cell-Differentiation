class Parameters:
    def __init__(self):
        #### time parameters ####
        self.T = 24                                # Time
        self.N = 3000                              # Number of timesteps
              
        #### geometry parameters ####
        self.dt = self.T/(self.N - 1)              # timestep
        self.r_max = 1                             # Maximum radius
        self.k = 0.5                               # Cell growth rate
        self.F0 = 0.1                              # Force scaling
        self.alpha = 3                             # Cell stiffness
        self.sigma = 0.7                           # Don't know how to call that one
        
        #### transcription parameters ####
        # energy differences
        self.eps_N = -6
        self.eps_G = -7
        self.eps_S = -2
        self.eps_NS = -2

        # decay rates
        self.gamma_N = 100
        self.gamma_G = 100
        self.gamma_S = 100

        # reproduction rates
        self.r_N = 10
        self.r_G = 10
        self.r_S = 10
        
        # signal range
        self.q = 0.3
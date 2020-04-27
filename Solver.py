import numpy as np
from Functions import *
from time import time

def L2prodfull(u, v, FVmesh):
    nofVar = int(len(u)/FVmesh.nofCells)
    prod = 0
    for i in range(nofVar):
        prod += L2prod(u[i*FVmesh.nofCells:(i+1)*FVmesh.nofCells],
                       v[i*FVmesh.nofCells:(i+1)*FVmesh.nofCells], FVmesh)

    return prod

# Newtons's method globalized with line search to find a root of a function.
# INPUT: f - Function depending on one variable (this variable can also be a list)
#        df - Derivative of above function
def Newton(f, df, x0, Prm, FVmesh):
    k = 0
    res = 1
    fnorm = L2prodfull(f(x0),f(x0),FVmesh)**(1/2)

    print('k =', k, 'res =', res, ', |f(x)| =', fnorm)
    xk = x0
    while k < Prm.Maxit and res > Prm.Tol and fnorm > 1e-3*Prm.Tol:

        dx0 = -np.linalg.solve(df(x0), f(x0))
        res = L2prodfull(dx0,dx0,FVmesh)**(1/2)/L2prodfull(x0,x0,FVmesh)**(1/2)

        xk = x0 + dx0

        sigma = 1
        k_armijo = 0
        while 0.5*L2prodfull(f(xk),f(xk),FVmesh) - 0.5*L2prodfull(f(x0),f(x0),FVmesh) > -1e-4*sigma*fnorm**2 and k_armijo < 50:
            sigma = 0.5*sigma
            xk = x0 + sigma*dx0
            k_armijo += 1

        x0 = xk
        k += 1
        fnorm = L2prodfull(f(x0),f(x0),FVmesh)**(1/2)
        print('k =', k, 'res =', res, ', |f(x)| =', fnorm, ', sigma =', sigma)

    print('\n')
    return xk, fnorm

class Solver:

    def __init__(self, nofVar, Prm):
        self.u = np.empty([nofVar, Prm.nofSteps+1])
        self.CompTime = []
        self.FuncNorm = np.empty(Prm.nofSteps)

    def implicit_Euler(self, f, df, u0, Prm, FVmesh):
        F = lambda u: (u-u0)/Prm.dt - f(u)
        dF = lambda u: np.diag(u)/Prm.dt - df(u)
        
        self.u[:,0] = u0
        for i in range(Prm.nofSteps):
            u0 = self.u[:,i]
            F = lambda u: (u-u0)/Prm.dt - f(u)
            self.u[:,i+1], self.FuncNorm[i] = Newton(F, dF, self.u[:,i], Prm, FVmesh)

    
    def Crank_Nicolson(self, f, df, u0, Prm, FVmesh):
        F = lambda u: (u-u0)/Prm.dt - 1/2*(f(u) + f(u0))
        dF = lambda u: np.diag(u)/Prm.dt - 1/2*df(u)
        
        self.u[:,0] = u0
        for i in range(Prm.nofSteps):
            u0 = self.u[:,i]
            F = lambda u: (u-u0)/Prm.dt - 1/2*(f(u) + f(u0))
            self.u[:,i+1], self.FuncNorm[i] = Newton(F, dF, self.u[:,i], Prm, FVmesh)


def solveEquation(f, df, u0, FVmesh, Prm, *Method):
    tic = time()
    nofVar = len(u0)
    self = Solver(nofVar, Prm)
    self.__init__(nofVar, Prm)

    if Method[0] == 'CN':
        self.Crank_Nicolson(f, df, u0, Prm, FVmesh)
    if Method[0] == 'Euler':
        self.implicit_Euler(f, df, u0, Prm, FVmesh)

    toc = time()
    self.CompTime = toc-tic
    
    return self
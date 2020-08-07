import numpy as np
import matplotlib.pyplot as plt
from random import gauss
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Functions import coverPlot, saveData
from Model import rhs_activation
from Parameters import setParameters
from scipy.integrate import solve_ivp
import pandas as pd

def rho(N,G,FVmesh):
    cover_N = np.empty(FVmesh.nofCells)
    cover_G = np.empty(FVmesh.nofCells)
    radius = max(FVmesh.Dist[FVmesh.Dist < np.inf])/2

    for i in range(FVmesh.nofCells):
            Vi = 0
            N_cells = 0
            G_cells = 0
            for j in range(FVmesh.nofCells):
                if (np.linalg.norm(FVmesh.Pos[j] - FVmesh.Pos[i])) <= radius:
                    Vi += 1#FVmesh.Vol[j]
                    if N[j] > G[j]:
                        N_cells += 1#*FVmesh.Vol[j]
                    else:
                        G_cells += 1#*FVmesh.Vol[j]
            
            cover_N[i] = N_cells/Vi
            cover_G[i] = G_cells/Vi

    rho_N = np.mean(cover_N)
    rho_G = np.mean(cover_G)

    return rho_N, rho_G

Prm = setParameters()
Pos = np.array(pd.read_csv('testOrganoid.csv'))
Radius = np.ones(len(Pos))*1.1
FVmesh = initializeFVmesh(Pos, Radius=Radius)

t = np.linspace(0,Prm.T,Prm.nofSteps)
xInit = np.array([gauss(0.03,0.001) if i < FVmesh.nofCells else 
                  gauss(0.03,0.001) for i in range(2*FVmesh.nofCells)])

rangeprm = []
for i in range(-1,1):
    for j in range(1,10):
        rangeprm.append(j*10**i) 
#rangeprm = np.linspace(0.1,30,100)

rho_N = np.empty(len(rangeprm))
rho_G = np.empty(len(rangeprm))
nofN = np.empty(len(rangeprm))
nofG = np.empty(len(rangeprm))
for i in range(len(rangeprm)):
    Prm.range = rangeprm[i]
    rhs = lambda t,x: rhs_activation(0, x, Prm, FVmesh)
    sol = solve_ivp(rhs, [0,Prm.T], xInit, t_eval = t, method = 'Radau')

    N = sol.y[:FVmesh.nofCells,-1]
    G = sol.y[FVmesh.nofCells:,-1]

    rho_N[i], rho_G[i] = rho(N,G,FVmesh)
    nofN[i] = len(N[N>G])/FVmesh.nofCells
    nofG[i] = len(G[G>N])/FVmesh.nofCells

plt.figure()
plt.plot(rangeprm, rho_N, lw = 2)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\rho$')
#plt.plot(rangeprm, rho_G)

plt.figure()
plt.plot(rangeprm, nofN*100, lw = 2)
#plt.plot(rangeprm, nofG)
plt.xlabel(r'$\alpha$')
plt.ylabel('% NANOG Zellen')
plt.show()

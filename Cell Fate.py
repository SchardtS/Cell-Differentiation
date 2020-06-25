import numpy as np
import matplotlib.pyplot as plt
from random import gauss
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Functions import coverPlot
from Model import rhs_activation
from Parameters import setParameters
from scipy.integrate import solve_ivp
import pandas as pd
import os

Prm = setParameters()
#Organoid = initializeOrganoid(Prm)
Pos = np.array(pd.read_csv('testOrganoid.csv'))
FVmesh = initializeFVmesh(Pos)

t = np.linspace(0,Prm.T,Prm.nofSteps)
xInit = np.array([gauss(0.2,0.01) if i < FVmesh.nofCells else 
                  gauss(0.2,0.01) for i in range(2*FVmesh.nofCells)])
rhs = lambda t,x: rhs_activation(0, x, Prm, FVmesh)
sol = solve_ivp(rhs, [0,Prm.T], xInit, t_eval = t, method = 'Radau')

N = sol.y[:FVmesh.nofCells,-1]
G = sol.y[FVmesh.nofCells:,-1]

plt.figure()
for i in range(FVmesh.nofCells):
     plt.plot(t, sol.y[i,:])

plt.title('NANOG')
plt.xlabel('Time')
plt.ylabel('Concentrations')


plt.figure()
for i in range(FVmesh.nofCells):
    plt.plot(t, sol.y[i+FVmesh.nofCells,:])

plt.title('GATA6')
plt.xlabel('Time')
plt.ylabel('Concentrations')

print('Number of Cells =', FVmesh.nofCells)
print('Number of NANOG Cells =', len(N[N>G]))
print('Number of GATA6 Cells =', len(G[G>N]))

plt.figure()
FVmesh.plot(N)

""" coverPlot(N, G, 100, FVmesh)
plt.show() """
plt.show()
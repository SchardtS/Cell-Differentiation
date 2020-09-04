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
import os

Prm = setParameters()
#Organoid = initializeOrganoid(Prm)
x = np.linspace(-0.1,0.1,8)
Pos = np.empty([len(x)**2,2])
for i in range(len(x)):
    for j in range(len(x)):
        Pos[j+i*len(x),0] = x[j]
        Pos[j+i*len(x),1] = x[i]

Pos = np.array(pd.read_csv('testOrganoid.csv'))
Radius = np.ones(len(Pos))*1.1
FVmesh = initializeFVmesh(Pos, Radius=Radius)

t = np.linspace(0,Prm.T,Prm.nofSteps)

x0 = [Prm.r_N/Prm.gamma_N*3/4, Prm.r_G/Prm.gamma_G*3/4]
xInit = np.array([gauss(x0[0],x0[0]*0.01) if i < FVmesh.nofCells else 
                  gauss(x0[1],x0[1]*0.01) for i in range(2*FVmesh.nofCells)])
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
D = Prm.D*25/3600
print('Diffusivity =', D)
print(Prm.dt*Prm.D)

plt.figure()
FVmesh.plot(N)
#coverPlot(N, G, 100, FVmesh, 'Cell Fate')
plt.show()
saveData(FVmesh, Prm, N, G, 'Cell Fate')
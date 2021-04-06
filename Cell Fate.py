import numpy as np
import matplotlib.pyplot as plt
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Functions import coverPlot, saveData, paircorrelation
from Model import rhs_activation
from Parameters import setParameters
from scipy.integrate import solve_ivp
import pandas as pd

Prm = setParameters()
#Organoid = initializeOrganoid(Prm, Transcription=False)
#Pos = Organoid.Pos
Pos = np.array(pd.read_csv('testOrganoid_small.csv'))
Radius = np.ones(len(Pos))*1.1
FVmesh = initializeFVmesh(Pos, Radius=Radius)

t = np.linspace(0,Prm.T,Prm.nofSteps)

x0 = [Prm.r_N/Prm.gamma_N*3/4, Prm.r_G/Prm.gamma_G*3/4]
xInit = np.append(np.random.normal(x0[0], x0[0]*0.01, FVmesh.nofCells),
                  np.random.normal(x0[1], x0[1]*0.01, FVmesh.nofCells))
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
plt.figure()
paircorrelation(N, G, FVmesh)
plt.show()
saveData(FVmesh, Prm, N, G, 'Cell Fate')
print(10*(np.exp(Prm.eps_G-Prm.eps_N)-1)/np.exp(Prm.eps_NS + Prm.eps_NS), len(N[N>G])/len(N), len(G[G>N])/len(N))
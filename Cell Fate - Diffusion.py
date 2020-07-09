import numpy as np
import matplotlib.pyplot as plt
from random import gauss
from FVmesh import initializeFVmesh
from Model import rhs_diffusion
from Solver import solveEquation
from Parameters import setParameters
from scipy.integrate import solve_ivp
from Organoid2D import initializeOrganoid
import pandas as pd
import os

############# INITIALIZE GEOMETRY #############
Prm = setParameters()
Org = initializeOrganoid(Prm, Transcription = False)
Pos = Org.Pos
#Pos = np.array(pd.read_csv('testOrganoid.csv'))
FVmesh = initializeFVmesh(Pos)
FVmesh.P = Org.Radius**2*np.pi/FVmesh.Vol

xInit = np.array([gauss(0.01,0.001) for i in range(3*FVmesh.nofCells)])
f = lambda t,x: rhs_diffusion(0, x, Prm, FVmesh)
t = np.linspace(0,Prm.T,Prm.nofSteps)
sol = solve_ivp(f, [0,Prm.T], xInit, t_eval = t, method = 'Radau')

N = sol.y[:FVmesh.nofCells,-1]
G = sol.y[FVmesh.nofCells:2*FVmesh.nofCells, -1]
S = sol.y[2*FVmesh.nofCells:, -1]


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


plt.figure()
for i in range(FVmesh.nofCells):
    plt.plot(t, sol.y[i+2*FVmesh.nofCells,:])

plt.title('Signal (Fgf4)')
plt.ylabel('Concentrations')
plt.xlabel('Time')

plt.figure()
FVmesh.plot(N)
plt.figure()
FVmesh.plot(FVmesh.P)
plt.show()
saveData(FVmesh, N, G, 'Cell Fate - Diffusion')
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from FVmesh import initializeFVmesh
from Functions import *
from Model import *
from Solver import solveEquation
from Parameters import setParameters
import os

############# INITIALIZE GEOMETRY #############
pos = CellGeometry(10,10)
Prm = setParameters()
FVmesh = initializeFVmesh(pos)

#k = 2
#a = 0.001
#b = 0.01
#c = 0.01
#d = 0.1
#D = 0
#n = 2

eps_N = -4
eps_G = -3
eps_A = -1
D = 0.3

uInit = np.array([0.5 + rd.gauss(0,0.01) for i in range(3*FVmesh.nofCells)])
#uInit[2*FVmesh.nofCells:] = np.array(Cdistance(FVmesh.Pos))
f = lambda t,x: rhs_diffusion_test(0, x, eps_N, eps_G, eps_A, D, FVmesh)
df = lambda u: drhs_diffusion_test(0, u, eps_N, eps_G, eps_A, D, FVmesh)
#sol = solveEquation(f, df, uInit, FVmesh, Prm, 'CN')
t = np.linspace(0,Prm.T,Prm.nofSteps)
from scipy.integrate import solve_ivp
sol = solve_ivp(f, [0,Prm.T], uInit, t_eval = t, method = 'Radau')

N = sol.y[:FVmesh.nofCells,-1]
G = sol.y[FVmesh.nofCells:2*FVmesh.nofCells, -1]
S = sol.y[2*FVmesh.nofCells:, -1]

#t = [i*Prm.dt for i in range(Prm.nofSteps+1)]

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
FVmesh.plot(S)
plt.show()
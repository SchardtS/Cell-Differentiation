################################################################################################################
# Author: Simon Schardt
# Last edit: 13.08.2019
################################################################################################################

import numpy as np
from numpy.linalg import solve, norm
import random as rd
import networkx as nx
from math import log
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Functions import dxx, dxx_test
from numba import jit

def neighbor_mean(x, FVmesh):
    nofCells = len(x)
    suma = np.empty(nofCells)

    for i in range(nofCells):
        mean = 0
        for j in FVmesh.Neigh[i]:
            mean += x[j]
        
        suma[i] = mean/len(FVmesh.Neigh[i])
        
    return suma

def convolute(x, Prm, FVmesh):
    
    np.fill_diagonal(FVmesh.Dist, np.inf)
    Phi = x/FVmesh.Dist**Prm.intensity
    
    return np.sum(Phi, axis=1)/len(x)

def rhs_activation(t, x, Prm, FVmesh):
    nofCells = FVmesh.nofCells
    rhs = np.empty(len(x))
    N = x[:nofCells]
    G = x[nofCells:]
    
    a = np.exp(-Prm.eps_N)
    b = np.exp(-Prm.eps_G)
    c = np.exp(-Prm.eps_A)

    if Prm.signal == 'local':
        Gb = neighbor_mean(G,FVmesh)
    elif Prm.signal == 'nonlocal':
        Gb = convolute(G,Prm,FVmesh)
    else:
        print('Error: Mode not supported, choose local or nonlocal instead')

    pN = ((a*N)*(1+c*Gb))/(1 + a*N*(1+c*Gb) + b*G)
    pG =      (b*G)      /(1 + a*N*(1+c*Gb) + b*G)

    rhs[:nofCells] = pN - Prm.gamma_N*N
    rhs[nofCells:] = pG - Prm.gamma_G*G
    
    return rhs

def rhs_diffusion(t, x, Prm, FVmesh):
    nofCells = FVmesh.nofCells
    rhs = np.empty(len(x))
    N = x[:nofCells]
    G = x[nofCells:2*nofCells]
    S = x[2*nofCells:]  
    Dxx = dxx_test(S,FVmesh)
    Sb = neighbor_mean(S, FVmesh)

    a = np.exp(-eps_N)
    b = np.exp(-eps_G)
    c = np.exp(-eps_A)

    P = -np.exp(-(FVmesh.Pos[:,0]**2+FVmesh.Pos[:,1]**2)/1)

    pN =       a*N     /(1 + a*N + b*G*(1+c*Sb))
    pG = (b*G)*(1+c*Sb)/(1 + a*N + b*G*(1+c*Sb))
    pS = 1 - pG#       1      /(1 + b*G)
    
    rhs[:nofCells] = pN - N
    rhs[nofCells:2*nofCells] = pG - G
    rhs[2*nofCells:] = np.dot(Dxx,P) + pS - S

    return rhs
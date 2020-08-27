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
from Functions import Eq2Mat
from scipy.special import kn

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
    
    #np.fill_diagonal(FVmesh.Dist, np.inf)
    #Phi = x/FVmesh.Dist**Prm.range
    
    D = Prm.D*Prm.T/Prm.nofSteps
    Phi = x/(4*np.pi*D)*np.exp(-(FVmesh.Dist)**2/(4*D))
    #Phi = x*np.exp(-FVmesh.Dist**2/Prm.range)


    #np.fill_diagonal(FVmesh.Dist, 1)
    #Phi = Prm.production*x/(2*np.pi)*kn(0,Prm.uptake**(1/2)*FVmesh.Dist)
    #Phi = Prm.production*x*np.exp(-Prm.uptake**(1/2)*FVmesh.Dist)/(4*np.pi*FVmesh.Dist)
    np.fill_diagonal(Phi, 0)

    return np.sum(Phi, axis=1)#/len(x)

def diffusion(x, Prm, FVmesh):
    Dxx = dxx(FVmesh)
    uptake = Prm.gamma_G/(1+np.exp(-Prm.eps_G)*x)

    LinOp = 100*Dxx - np.diag((uptake+Prm.gamma_G))
    s = np.linalg.solve(-LinOp,x)
    return 400*uptake/Prm.gamma_G*s


def rhs_activation(t, x, Prm, FVmesh):
    nofCells = FVmesh.nofCells
    rhs = np.empty(len(x))
    N = x[:nofCells]
    G = x[nofCells:]
    
    a = np.exp(-Prm.eps_N)
    b = np.exp(-Prm.eps_G)
    c = np.exp(-Prm.eps_S)
    d = np.exp(-Prm.eps_NS)

    if Prm.signal == 'local':
        S = neighbor_mean(G,FVmesh)
    elif Prm.signal == 'nonlocal':
        S = convolute(G,Prm,FVmesh)
    elif Prm.signal == 'diffusion':
        S = diffusion(G,Prm,FVmesh)
    else:
        print('Error: Mode not supported, choose local or nonlocal instead')

    pN = (a*N)*(1+d*c*S)/(1 + a*N*(1+d*c*S) + b*G + c*S)
    pG =      (b*G)      /(1 + a*N*(1+d*c*S) + b*G + c*S)

    rhs[:nofCells] = Prm.r_N*pN - Prm.gamma_N*N
    rhs[nofCells:] = Prm.r_G*pG - Prm.gamma_G*G
    
    return rhs#*Prm.relSpeed

def rhs_diffusion(t, x, Prm, FVmesh):
    nofCells = FVmesh.nofCells
    rhs = np.empty(len(x))
    N = x[:nofCells]
    G = x[nofCells:2*nofCells]
    S = x[2*nofCells:]
    #Dxx = dxx(FVmesh)
    Dxx = dxx_test(S,FVmesh)
    Sb = neighbor_mean(S, FVmesh)

    a = np.exp(-Prm.eps_N)
    b = np.exp(-Prm.eps_G)
    c = np.exp(-Prm.eps_Sb)
    d = np.exp(-Prm.eps_S)

    pN = a*N / (1 + a*N + b*G + c*Sb + b*c*G*Sb)
    pG = b*G*(1 + c*Sb) / (1 + a*N + b*G + c*Sb + b*c*G*Sb)
    pS = d*S / (1 + d*S + b*G)
    
    rhs[:nofCells] = pN - Prm.gamma_N*N
    rhs[nofCells:2*nofCells] = pG - Prm.gamma_G*G
    rhs[2*nofCells:] = pS - Prm.gamma_S*S + 0.1*np.dot(Dxx,FVmesh.P)

    return rhs*Prm.relSpeed



############################################################################################

def region_mean(x, FVmesh):
    nofCells = len(x)
    suma = np.empty(nofCells)

    for i in range(nofCells):
        mean = 0
        for j in FVmesh.Neigh[i]:
            mean += x[j]
        
        suma[i] = (mean + x[i])/(len(FVmesh.Neigh[i])+1)
        
    return suma


def rhs_test(t, x, Prm, FVmesh):
    nofCells = FVmesh.nofCells
    rhs = np.empty(len(x))
    N = x[:nofCells]
    G = x[nofCells:2*nofCells]
    S = x[2*nofCells:]
    Sb = region_mean(S, FVmesh)

    a = np.exp(-Prm.eps_N)
    b = np.exp(-Prm.eps_G)
    c = np.exp(-Prm.eps_S)
    eps_act = 2
    
    nu = 1

    pN =         nu*a*N         / (nu**2 + a*N + b*G*(nu+eps_act*c*Sb) + c*Sb)
    pG =  b*G*(nu+eps_act*c*Sb) / (nu**2 + a*N + b*G*(nu+eps_act*c*Sb) + c*Sb)
    pS =  c*Sb*(nu+eps_act*b*G) / (nu**2 + a*N + b*G*(nu+eps_act*c*Sb) + c*Sb)
    
    rhs[:nofCells] = pN - Prm.gamma_N*N
    rhs[nofCells:2*nofCells] = pG - Prm.gamma_G*G
    rhs[2*nofCells:] = pS - Prm.gamma_S*S

    return rhs*Prm.relSpeed
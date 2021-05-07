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

def neighbor_signal(x, FVmesh):
    val = np.empty(FVmesh.nofCells)

    for i in range(FVmesh.nofCells):
        mean = 0
        for j in FVmesh.Neigh[i]:
            mean += x[j]
        
        val[i] = mean/len(FVmesh.Neigh[i])
        
    return val

def graph_signal(x, Prm, FVmesh):
    q = Prm.range
    scaling = q**(FVmesh.GraphDist)
    val = x*scaling*(1-q)/q
    #scaling = np.array(FVmesh.GraphDist)
    #scaling[scaling > 1] = 0
    #val = x*scaling
    
    np.fill_diagonal(val, 0)

    return np.sum(val, axis=1)#/scaling.sum(0)

def relative_distance(v, FVmesh):
    from scipy.spatial.distance import cdist
    
    rel_Pos = FVmesh.Pos - v

    return cdist(FVmesh.Pos, rel_Pos)

def convolute(x, Prm, FVmesh):
    
    #np.fill_diagonal(FVmesh.Dist, np.inf)
    #Phi = x/FVmesh.Dist**Prm.range
    
    D = Prm.D
    dt = Prm.dt
    #Dist = relative_distance([0,-300000*dt], FVmesh)
    Phi = x/(4*np.pi*D*dt)*np.exp(-(FVmesh.Dist)**2/(4*D*dt))
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
        S = neighbor_signal(G,FVmesh)
    elif Prm.signal == 'nonlocal':
        S = graph_signal(G,Prm,FVmesh)
        #S = convolute(G,Prm,FVmesh)
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
    A = FVmesh.GraphDist
    #A[A <= 1] = 1
    #A[A > 1] = 0
    #q = Prm.range
    #A = (1-q)/q*q**(FVmesh.GraphDist)
    #np.fill_diagonal(A, 0)
    FVmesh.Vol[:] = 1
    A = dxx(FVmesh)
    #np.fill_diagonal(A, 0)
    #Dxx = dxx_test(S,FVmesh)
    #P = np.exp(-FVmesh.Pos[:,0]**2-FVmesh.Pos[:,1]**2)

    a = np.exp(-Prm.eps_N)
    b = np.exp(-Prm.eps_G)
    c = np.exp(-Prm.eps_S)

    Sb = S*a*N/(1+a*N)#np.dot(A,S)/6

    pN = a*N / (1 + a*N + b*G*(1 + c*Sb) + c*Sb)
    pG = b*G*(1 + c*Sb) / (1 + a*N + b*G*(1 + c*Sb) + c*Sb)
    pS = (1 - pG)
    
    rhs[:nofCells] = Prm.r_N*pN - Prm.gamma_N*N
    rhs[nofCells:2*nofCells] = Prm.r_G*pG - Prm.gamma_G*G
    rhs[2*nofCells:] = Prm.r_S*pS - Prm.gamma_S*S + 0.1*np.dot(A, S)
    return rhs



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
################################################################################################################
# Author: Simon Schardt
# Last edit: 13.08.2019
################################################################################################################

import numpy as np
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from numpy.linalg import solve, norm
import random as rd
import networkx as nx
from math import log
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint, Point
from scipy.spatial.distance import cdist
import pandas as pd
from FVmesh import initializeFVmesh
from matplotlib.animation import FuncAnimation


# Fate assignment function (only valid until a better deciding criterion has been found)
# INPUT: N - NANOG levels of all cells
#        G - GATA6 levels of all cells
def fate(N,G):
    return [1 if N[i]/max(N) >= G[i]/max(G) else 0 for i in range(len(N))]

def energyfate(N,G):
    dN = np.diff(N)
    dG = np.diff(G)
    
    x = np.empty(dN.shape[0])
    y = np.empty(dN.shape[0])
    for i in range(dN.shape[0]):
        for j in range(dN.shape[1]):
            if dN[i,j] < 0 and dG[i,j] < 0:
                x[i] = N[i,j+1]/max(N[:,j+1])
                y[i] = G[i,j+1]/max(G[:,j+1])
                break
                
    return fate(x,y)

def voronoi_volumes(pos):
    vor = Voronoi(pos)
    vol = np.empty(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(vor.vertices[indices]).volume
    return vol

# Finds the ID of any cell from the Delaunay Cell Graph
# INPUT: i - Index of the position
#        tri - Delaunay triangulation of your position data
def find_neighbors(i, tri):
    return tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]

# Distances of every point x_i to each point x_j.
# INPUT: pos - Positional Data [x, y, z], with x = [x_1,...,x_n] etc.
#        i - Index i of first position (only if single distances are needed)
#        j - Index j of second position (only if single distances are needed)
def distance(pos, *index):
    if index == ():                
        dist = cdist(pos, pos)
    else:
        dist = np.linalg.norm(pos[index[0]]-pos[index[1]])

    return dist

# Distances of every point x_i to the centre of mass.
# INPUT: pos - Positional Data [x, y, z], with x = [x_1,...,x_n] etc.
#        index - Index of wanted position (only if single distances are needed)
def Cdistance(pos, *index):
    if pos[0,:].size == 2:
        Centre = np.array([sum(pos[:,0])/len(pos), sum(pos[:,1])/len(pos)])
    if pos[0,:].size == 3:
        Centre = np.array([sum(pos[:,0])/len(pos), sum(pos[:,1])/len(pos), sum(pos[:,2])/len(pos)])
    if index == ():    
        dist = [np.linalg.norm(p-Centre) for p in pos]
    else:
        dist = np.linalg.norm(pos[index]-Centre)

    return dist

# Newtons's method globalized with line search to find a root of a function.
# INPUT: f - Function depending on one variable (this variable can also be a list)
#        df - Derivative of above function
#        tol - Prefered tolerance for the residual
#        maxit - Maximum number of Iterations
def Newton(f, df, x0, tol, maxit):
    k = 0
    res = 1
    fnorm = norm(f(x0))
    print('k =', k, 'res =', res, ', |f(x)| =', fnorm)
    while k < maxit and res > tol and fnorm > 1e-3*tol:
        if type(x0) == int or type(x0) == float:
            dx0 = -f(x0)/df(x0)
            res = abs(dx0/x0)
        else:
            dx0 = -solve(df(x0), f(x0))
            res = norm(dx0)/norm(x0)
            
        xk = x0 + dx0
        
        sigma = 1
        k_armijo = 0
        while 0.5*norm(f(xk))**2 - 0.5*norm(f(x0))**2 > -1e-4*sigma*fnorm**2 and k_armijo < 50:
            sigma = 0.5*sigma
            xk = x0 + sigma*dx0
            k_armijo += 1
        
        x0 = xk
        k += 1
        fnorm = norm(f(x0))
        print('k =', k, 'res =', res, ', |f(x)| =', fnorm, ', sigma =', sigma)
        
    return xk

# Initializes a spherical cell geometry based on a rectangular grid with randomly disturbed positions (currently only supports equal amounts of numbers in every direction)
# INPUT: NX - Number of cells in x-direction
#        NY - Number of cells in y-direction
#        NZ - Number of cells in z-direction (optional: decides wether 2D or 3D geometry should be created)
def CellGeometry(NX, NY, *NZ):
    if NZ == ():
     
        perturbation = 0.5/NX
        
        XPos = np.linspace(-1, 1, NX)
        YPos = np.linspace(-1, 1, NY)

        CellPos = [0]*NX*NY
        for i in range(NX):
            for j in range(NY):
                X = XPos[i] + rd.gauss(0,perturbation)
                Y = YPos[j] + rd.gauss(0,perturbation)
                CellPos[j*NX+i] = [max([abs(X),abs(Y)])/(X**2+Y**2)**(1/2)*X,
                                       max([abs(X),abs(Y)])/(X**2+Y**2)**(1/2)*Y]
                #CellPos[j*NX+i] = [X,Y]
    else:
        
        perturbation = 0.5/NX
        
        NZ = int(NZ[0])
        XPos = np.linspace(-1, 1, NX)
        YPos = np.linspace(-1, 1, NY)
        ZPos = np.linspace(-1, 1, NZ)

        CellPos = [0]*NX*NY*NZ
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    X = XPos[i] + rd.gauss(0,perturbation)
                    Y = YPos[j] + rd.gauss(0,perturbation)
                    Z = ZPos[k] + rd.gauss(0,perturbation)
                    CellPos[i*NX**2+j*NY+k] = [max([abs(X),abs(Y),abs(Z)])/(X**2+Y**2+Z**2)**(1/2)*X,
                                               max([abs(X),abs(Y),abs(Z)])/(X**2+Y**2+Z**2)**(1/2)*Y,
                                               max([abs(X),abs(Y),abs(Z)])/(X**2+Y**2+Z**2)**(1/2)*Z]
                
    return np.array(CellPos)


# L2 inner product on the Voronoi mesh from the FVmesh class.
# INPUT: u - Discretized L2 function (list of function values at grid points of the FVmesh)
#        v - Discretized L2 function (list of function values at grid points of the FVmesh)
#        FVmesh - Finite Volume mesh class
# OUTPUT: dxx_mat - Discretization matrix
def L2prod(u, v, FVmesh):
    return np.sum(FVmesh.Vol*u*v)
    

# Finite Volume discretization for laplace operators on the Voronoi mesh from the FVmesh class.
# INPUT: FVmesh - Finite Volume mesh class
# OUTPUT: dxx_mat - Discretization matrix
def dxx(FVmesh):   
    np.fill_diagonal(FVmesh.Dist, 1)
    offdiag = FVmesh.Edge/np.reshape(FVmesh.Vol, [FVmesh.nofCells, 1])/FVmesh.Dist
    diag = np.diag(np.sum(offdiag, axis=1))

    dxx_mat = offdiag - diag

    return dxx_mat

def dxx_test(a,FVmesh):   
    np.fill_diagonal(FVmesh.Dist, 10)
    aT = np.reshape(a,[len(a),1])
    A = (a+aT)/2
    A[FVmesh.Edge == 0] = 0
    offdiag = FVmesh.Edge/np.reshape(FVmesh.Vol, [FVmesh.nofCells, 1])/FVmesh.Dist*A
    diag = np.diag(np.sum(offdiag, axis=1))

    dxx_mat = offdiag - diag

    return dxx_mat

def Eq2Mat(eq, N):
    E = np.eye(N)
    Mat = np.empty([N,N])
    
    for i in range(N):
        Mat[:,i] = eq(E[:,i])

    return Mat


def coverPlot(N, G, nofCalc, FVmesh):
    nofCells = len(FVmesh.Pos)
    cover_N = np.empty(nofCells)
    cover_G = np.empty(nofCells)
    radius = np.linspace(0,max(FVmesh.Dist[FVmesh.Dist < np.inf]),nofCalc)
    f_N = np.empty(nofCalc) 
    f_G = np.empty(nofCalc)

    for k,r in enumerate(radius):
        for i in range(nofCells):
            Vi = 0
            N_cells = 0
            G_cells = 0
            for j in range(nofCells):
                if (np.linalg.norm(FVmesh.Pos[j] - FVmesh.Pos[i])) <= r:
                    Vi += FVmesh.Vol[j]
                    if N[j] > G[j]:
                        N_cells += 1*FVmesh.Vol[j]
                    else:
                        G_cells += 1*FVmesh.Vol[j]
            
            cover_N[i] = N_cells/Vi
            cover_G[i] = G_cells/Vi

        f_N[k] = np.mean(cover_N)
        f_G[k] = np.mean(cover_G)
        
    plt.figure()
    plt.rc('font', size=14)
    plt.plot(radius/radius[-1],f_N, lw = 3, color = 'm')
    plt.xlabel('Radius')
    plt.ylabel('$\\rho$')
        
    plt.axhline(sum(FVmesh.Vol[N>G])/sum(FVmesh.Vol), color = 'k', linestyle = '--', lw = 2)
    plt.axhline(len(N[N>G])/len(N), color = 'k', linestyle = '--', lw = 2)

    plt.figure()
    plt.rc('font', size=14)
    plt.plot(radius/radius[-1],f_G, lw = 3, color = 'c')
    plt.xlabel('Radius')
    plt.ylabel('$\\rho$')
        
    plt.axhline(sum(FVmesh.Vol[G>N])/sum(FVmesh.Vol), color = 'k', linestyle = '--', lw = 2)
    plt.axhline(len(G[G>N])/len(N), color = 'k', linestyle = '--', lw = 2)

    return

def saveData(FVmesh, N, G, folder):

    dic = {'x-Position': FVmesh.Pos[:,0], 'y-Position': FVmesh.Pos[:,1],
         'Radius': FVmesh.Radius, 'NANOG': N, 'GATA6': G}
    df = pd.DataFrame(dic)
    df.to_csv('Results/'+folder+'/Data.csv', index = False)

    plt.figure()
    FVmesh.plot(N)
    plt.savefig('Results/'+folder+'/NANOG.png')
    plt.savefig('Results/'+folder+'/NANOG.pdf')

    plt.figure()
    FVmesh.plot(G)
    plt.savefig('Results/'+folder+'/GATA6.png')
    plt.savefig('Results/'+folder+'/GATA6.pdf')

    return

def saveOrg(n, Organoid, Prm, folder):

    indices = np.linspace(0,Prm.nofSteps,n+1)
    for j, i in enumerate(indices):
        index = int(i)
        plt.figure()
        N = Organoid.Data[index][3]
        G = Organoid.Data[index][4]
        Pos = Organoid.Data[index][1]
        Rad = Organoid.Data[index][2]
        FVmesh = initializeFVmesh(Pos, Radius = Rad)
        FVmesh.plot(N, size=1000/len(Organoid.IDs), bounds=[min(Organoid.NANOG),max(Organoid.NANOG)])
        bmin = min(min(Organoid.Pos[:,0])*1.3,min(Organoid.Pos[:,1])*1.3)
        bmax = max(max(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,1])*1.3)
        plt.xlim(bmin, bmax)
        plt.ylim(bmin, bmax)

        k = str(j)+'of'+str(n)
        plt.savefig('Results/'+folder+'/NANOG_'+k+'.png')
        plt.savefig('Results/'+folder+'/NANOG_'+k+'.pdf')

        dic = {'x-Position': FVmesh.Pos[:,0], 'y-Position': FVmesh.Pos[:,1],
         'Radius': FVmesh.Radius, 'NANOG': N, 'GATA6': G}
        df = pd.DataFrame(dic)
        df.to_csv('Results/'+folder+'/Data_'+k+'.csv', index = False)

    return

def saveAnim(Organoid, Prm, folder):
    fig, ax = plt.subplots()

    def update(i):
        Pos = Organoid.Data[i][1]
        NANOG = Organoid.Data[i][3]
        FVmesh = initializeFVmesh(Pos)
        plt.cla()
        FVmesh.plot(NANOG, size=1000/len(Organoid.IDs), bounds=[min(Organoid.NANOG),max(Organoid.NANOG)])
        bmin = min(min(Organoid.Pos[:,0])*1.3,min(Organoid.Pos[:,1])*1.3)
        bmax = max(max(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,1])*1.3)
        plt.xlim(bmin, bmax)
        plt.ylim(bmin, bmax)
        return

    ani = FuncAnimation(fig, update, frames=Prm.nofSteps, interval=1, blit=False)
    ani.save('Results/'+folder+'/NANOG.mp4', fps=70, dpi=400)

    return




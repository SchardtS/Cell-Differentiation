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
import networkx as nx
import itertools


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


def graph_distance(FVmesh):
    G = nx.Graph()
    for path in FVmesh.Tri.simplices:
        nx.add_path(G, path)
        
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(G))
    GraphDist = np.empty([FVmesh.nofCells, FVmesh.nofCells])
    for i in range(FVmesh.nofCells):
        for j in range(FVmesh.nofCells):
            GraphDist[i,j] = dist_dict[i][j]

    return GraphDist

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

def coverPlot(N, G, nofCalc, FVmesh, folder):
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
                    Vi += 1#FVmesh.Vol[j]
                    if N[j] > G[j]:
                        N_cells += 1#*FVmesh.Vol[j]
                    else:
                        G_cells += 1#*FVmesh.Vol[j]
            
            cover_N[i] = N_cells/Vi
            cover_G[i] = G_cells/Vi

        f_N[k] = np.mean(cover_N)/len(N[N>G])*len(N)
        f_G[k] = np.mean(cover_G)/len(G[G>N])*len(N)
        
    ylimiter = max(max(abs(f_N)-1),max(abs(f_G-1)))
    ylimiter = np.ceil(ylimiter*100)/100
    plt.figure()
    plt.rc('font', size=14)
    plt.plot(radius/radius[-1],f_N, lw = 3, color = 'm')
    plt.plot(radius/radius[-1],f_G, lw = 3, color = 'c')
    plt.xlabel('Radius')
    plt.ylabel('$\\rho$')
    plt.ylim([1-ylimiter-ylimiter/10,1+ylimiter+ylimiter/10])
    plt.locator_params(axis='y', nbins=3)
    
    """ from matplotlib.ticker import FormatStrFormatter
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) """
        
    plt.axhline(1, color = 'k', linestyle = '--', lw = 2)

    plt.savefig('Results/'+folder+'/PairCorr.png')
    plt.savefig('Results/'+folder+'/PairCorr.pdf')

    return

def paircorrelation(N, G, FVmesh, ls = 'solid'):
    Gr = nx.Graph()
    for path in FVmesh.Tri.simplices:
    
        path1 = [path[0], path[1]]
        path2 = [path[1], path[2]]
        path3 = [path[2], path[0]]

        if FVmesh.Dist[path1[0],path1[1]] < 2.2:
            nx.add_path(Gr, path1)
        if FVmesh.Dist[path2[0],path2[1]] < 2.2:    
            nx.add_path(Gr, path2)
        if FVmesh.Dist[path3[0],path3[1]] < 2.2:
            nx.add_path(Gr, path3)
        
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(Gr))
    GraphDist = np.empty([FVmesh.nofCells, FVmesh.nofCells])
    for i in range(FVmesh.nofCells):
        for j in range(FVmesh.nofCells):
            GraphDist[i,j] = dist_dict[i][j]


    x = np.array(fate(N, G))
    maxdist = int(np.max(GraphDist))
    ind = np.where(x==1)[0]
    dist = GraphDist[ind].T[ind].T
    rho0 = sum(x)/len(x)
    rho1 = (sum(x)-1)/(len(x)-1)
    
    Px = np.empty(maxdist)
    for i in range(1,maxdist+1):
        Px[i-1] = len(dist[dist==i])/len(GraphDist[GraphDist==i])/rho0/rho1

    y = np.array(fate(G, N))
    ind = np.where(y==1)[0]
    dist = GraphDist[ind].T[ind].T
    rho0 = sum(y)/len(y)
    rho1 = (sum(y)-1)/(len(y)-1)
    
    Py = np.empty(maxdist)
    for i in range(1,maxdist+1):
        Py[i-1] = len(dist[dist==i])/len(GraphDist[GraphDist==i])/rho0/rho1
        
    #plt.figure()
    plt.rc('font', size=14)
    distances = [i for i in range(1,int(np.max(GraphDist))+1)]
    plt.plot(distances, Px, lw=2, color='m', linestyle = ls)#, label = 'NANOG')
    plt.plot(distances, Py, lw=2, color='c', linestyle = ls)#, label = 'GATA6')

    plt.xlabel('Distance')
    plt.ylabel('$\\rho$')
    #plt.legend()

    return

def saveData(FVmesh, Prm, N, G, folder):

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

    with open('Results/'+folder+'/Parameters.txt', 'w') as f:
        print('Energy differences---------------------', file=f)
        print('eps_N =', Prm.eps_N, file=f)
        print('eps_G =', Prm.eps_G, file=f)
        print('eps_S =', Prm.eps_S, file=f)
        print('eps_NS =', Prm.eps_NS, file=f)
        print('', file=f)

        print('Reproduction rates---------------------', file=f)
        print('r_N =', Prm.r_N, file=f)
        print('r_G =', Prm.r_G, file=f)
        print('', file=f)

        print('Decay rates----------------------------', file=f)
        print('gamma_N =', Prm.gamma_N, file=f)
        print('gamma_G =', Prm.gamma_G, file=f)
        print('', file=f)

        print('Signal parameters----------------------', file=f)
        print('signal =', Prm.signal, file=f)
        if Prm.signal == 'nonlocal':
            print('D =', Prm.D, file=f)

        print('', file=f)

        print('Numerical parameters-------------------', file=f)
        print('T', Prm.T, file=f)
        print('nofSteps =', Prm.nofSteps, file=f)
        print('dt =', Prm.dt, file=f)
        print('', file=f)
 
        print('Cell numbers---------------------------', file=f)
        print('Number of Cells =', FVmesh.nofCells, file=f)
        print('Number of NANOG Cells =', len(N[N>G]), file=f)
        print('Number of GATA6 Cells =', len(G[G>N]), file=f)
    return

def loadData(file):
    Data = pd.read_csv(file)
    x = Data['x-Position']
    y = Data['y-Position']
    Pos = np.empty([len(x), 2])
    Pos[:,0] = x
    Pos[:,1] = y

    Radius = Data['Radius']
    N = Data['NANOG']
    G = Data['GATA6']

    return Pos, Radius, N, G

def saveOrg(n, Organoid, Prm, folder):

    indices = np.linspace(0,Prm.nofSteps,n+1)
    for j, i in enumerate(indices):
        index = int(i)
        plt.figure()
        N = Organoid.Data[index][3]
        G = Organoid.Data[index][4]
        Pos = Organoid.Data[index][1]
        Rad = Organoid.Data[index][2]
        FVmesh = initializeFVmesh(Pos, Radius = np.ones(len(Rad))*.8)
        FVmesh.plot(N, size=1000/len(Organoid.IDs), bounds=[min(Organoid.NANOG),max(Organoid.NANOG)])
        bmin = min(min(Organoid.Pos[:,0])*1.3,min(Organoid.Pos[:,1])*1.3)
        bmax = max(max(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,1])*1.3)
        plt.xlim(bmin, bmax)
        plt.ylim(bmin, bmax)

        k = str(j)+'of'+str(n)
        plt.savefig('Results/'+folder+'/NANOG_'+k+'.png', transparent = True)
        plt.savefig('Results/'+folder+'/NANOG_'+k+'.pdf')

        dic = {'x-Position': FVmesh.Pos[:,0], 'y-Position': FVmesh.Pos[:,1],
         'Radius': FVmesh.Radius, 'NANOG': N, 'GATA6': G}
        df = pd.DataFrame(dic)
        df.to_csv('Results/'+folder+'/Data_'+k+'.csv', index = False)

        with open('Results/'+folder+'/Parameters.txt', 'w') as f:
            print('Energy differences---------------------', file=f)
            print('eps_N =', Prm.eps_N, file=f)
            print('eps_G =', Prm.eps_G, file=f)
            print('eps_S =', Prm.eps_S, file=f)
            print('eps_NS =', Prm.eps_NS, file=f)
            print('', file=f)

            print('Reproduction rates---------------------', file=f)
            print('r_N =', Prm.r_N, file=f)
            print('r_G =', Prm.r_G, file=f)
            print('', file=f)

            print('Decay rates----------------------------', file=f)
            print('gamma_N =', Prm.gamma_N, file=f)
            print('gamma_G =', Prm.gamma_G, file=f)
            print('', file=f)

            print('Signal parameters----------------------', file=f)
            print('signal =', Prm.signal, file=f)
            if Prm.signal == 'nonlocal':
                print('D =', Prm.D, file=f)

            print('', file=f)

            print('Numerical parameters-------------------', file=f)
            print('T', Prm.T, file=f)
            print('nofSteps =', Prm.nofSteps, file=f)
            print('dt =', Prm.dt, file=f)
            print('', file=f)

            print('Growth parameters----------------------', file=f)
            print('nofCells_start =', Prm.nofCells_start, file=f)
            print('nofCells_end =', Prm.nofCells_end, file=f)
            print('r_max =', Prm.rmax, file=f)
            print('alpha =', Prm.alpha, file=f)
            print('sigma =', Prm.sigma, file=f)
            print('F0 =', Prm.F0, file=f)
            print('', file=f)

            print('Cell numbers---------------------------', file=f)
            print('Number of Cells =', FVmesh.nofCells, file=f)
            print('Number of NANOG Cells =', len(N[N>G]), file=f)
            print('Number of GATA6 Cells =', len(G[G>N]), file=f)

    return

def saveAnim(Organoid, Prm, folder):
    fig, ax = plt.subplots()

    def update(i):
        Pos = Organoid.Data[i][1]
        Radius = Organoid.Data[i][2]
        NANOG = Organoid.Data[i][3]
        FVmesh = initializeFVmesh(Pos, Radius = np.ones(len(Radius))*.8)
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

# Cutoff distance in experimental data should be fixed to 91
def graphdistance3D(Pos, cutoff = 91):
    
    Gr = nx.Graph()
    Dist = cdist(Pos, Pos)
    tri = Delaunay(Pos)
    n = len(Pos)

    for simp in tri.simplices:
        for path in list(itertools.combinations(simp, 2)):
            if Dist[path[0],path[1]] < cutoff:
                nx.add_path(Gr, path)

    dist_dict = dict(nx.all_pairs_dijkstra_path_length(Gr))
    dist = np.empty([n, n])
    for i in range(n):
        for j in range(n):
            dist[i,j] = dist_dict[i][j]
    
    """tri = Delaunay(Pos)
    Dist = cdist(Pos, Pos)
    Gr = nx.Graph()
    n = len(Pos)

    simplices = tri.simplices[(Dist[tri.simplices[:,0],tri.simplices[:,1]] < cutoff) & 
                              (Dist[tri.simplices[:,0],tri.simplices[:,2]] < cutoff) &
                              (Dist[tri.simplices[:,0],tri.simplices[:,3]] < cutoff) &
                              (Dist[tri.simplices[:,1],tri.simplices[:,2]] < cutoff) & 
                              (Dist[tri.simplices[:,1],tri.simplices[:,3]] < cutoff) &
                              (Dist[tri.simplices[:,2],tri.simplices[:,3]] < cutoff)]

    for path in simplices:
        nx.add_path(Gr, path)

    dist_dict = dict(nx.all_pairs_dijkstra_path_length(Gr))
    dist = np.empty([n, n])
    for i in range(n):
        for j in range(n):
            dist[i,j] = dist_dict[i][j]"""


    return dist


def loadExpData(ID):

    Data = pd.read_csv('Data/includingSurfaceDistance/extendedRawDataICMOrganoids.csv')

    Organoids = max(Data['OrganoidID'])
    Dataindex = []
    Cells = []

    for i in range(len(Data)):
        if Data['OrganoidID'][i] == ID:
            Dataindex.append(i)

    x = np.array(Data.loc[Dataindex,'CentroidX'])
    y = np.array(Data.loc[Dataindex,'CentroidY'])
    z = np.array(Data.loc[Dataindex,'CentroidZ'])
    N = np.array(Data.loc[Dataindex,'Nanog-Avg'])
    G = np.array(Data.loc[Dataindex,'Gata6-Avg'])
    Stage = np.array(Data.loc[Dataindex,'stage'])
    Population = np.array(Data.loc[Dataindex,'Population'])
    Nmax = max(Data.loc[Dataindex,'Nanog-Avg'])
    Gmax = max(Data.loc[Dataindex,'Gata6-Avg'])
    Nmin = min(Data.loc[Dataindex,'Nanog-Avg'])
    Gmin = min(Data.loc[Dataindex,'Gata6-Avg'])
    print('Organoid', ID, 'is', Stage[0], 'old')
    print('Organoid', ID, 'consists of', len(Dataindex), 'cells')
    print('Organoid', ID, 'consists of', len(x[Population=='N+G-']), 'NANOG cells')
    print('Organoid', ID, 'consists of', len(x[Population=='N-G+']), 'GATA6 cells')
    print('Organoid', ID, 'has a NANOG:GATA6 ratio of', len(x[Population=='N+G-'])/len(x[Population=='N-G+']))

    Pos = np.empty([len(x), 3])
    Pos[:,0] = x
    Pos[:,1] = y
    Pos[:,2] = z
    
    return N, G, Population, Pos


def pc_bounds(Pop, GraphDist, N, portion_x = 1):
    maxdist = int(np.max(GraphDist))
    x = np.zeros([len(Pop), N])   
    y = np.zeros([len(Pop), N])
    x[(Pop == 'N+G-')] = 1
    x[(Pop == 'N+G+') | (Pop == 'N-G-')] = np.random.random(x[(Pop == 'N+G+') | (Pop == 'N-G-')].shape)    
    y[(Pop == 'N-G+')] = 1
    y[(Pop == 'N+G+') | (Pop == 'N-G-')] = np.random.random(x[(Pop == 'N+G+') | (Pop == 'N-G-')].shape)
    
    if portion_x == 1:
        portion_x = len(x[(Pop == 'N-G+')])/len(x[(Pop == 'N-G+') | (Pop == 'N+G-')])
    portion_y = 1 - portion_x
        
    x[x > portion_x] = 1
    x[x <= portion_x] = 0
    y[y > portion_y] = 1
    y[y <= portion_y] = 0
    
    cells = x.shape[0]
    cells_x = np.sum(x, axis=0)
    cells_y = np.sum(y, axis=0)
    rho_x = cells_x*(cells_x - 1)/(cells*(cells - 1))
    rho_y = cells_y*(cells_y - 1)/(cells*(cells - 1))

    Px = np.zeros([maxdist,N])
    Py = np.zeros([maxdist,N])
    for j in range(N):
        ind_x = np.where(x[:,j]==1)[0]
        pairs_x = GraphDist[ind_x].T[ind_x].T
        
        ind_y = np.where(y[:,j]==1)[0]
        pairs_y = GraphDist[ind_y].T[ind_y].T

        for i in range(1,maxdist+1):
            Px[i-1,j] = len(pairs_x[pairs_x==i])/len(GraphDist[GraphDist==i])/rho_x[j]
            Py[i-1,j] = len(pairs_y[pairs_y==i])/len(GraphDist[GraphDist==i])/rho_y[j]

    Px_max = np.max(Px, axis=1)
    Px_min = np.min(Px, axis=1)    
    Py_max = np.max(Py, axis=1)
    Py_min = np.min(Py, axis=1)
    
    return Px_min, Px_max, Py_min, Py_max


def pc_mean(Pop, GraphDist, N, portion_x = 1):
    maxdist = int(np.max(GraphDist))
    x = np.zeros([len(Pop), N])   
    y = np.zeros([len(Pop), N])
    x[(Pop == 'N+G-')] = 1
    x[(Pop == 'N+G+') | (Pop == 'N-G-')] = np.random.random(x[(Pop == 'N+G+') | (Pop == 'N-G-')].shape)    
    y[(Pop == 'N-G+')] = 1
    y[(Pop == 'N+G+') | (Pop == 'N-G-')] = np.random.random(x[(Pop == 'N+G+') | (Pop == 'N-G-')].shape)
    
    if portion_x == 1:
        portion_x = len(x[(Pop == 'N-G+')])/len(x[(Pop == 'N-G+') | (Pop == 'N+G-')])
    portion_y = 1 - portion_x
        
    x[x > portion_x] = 1
    x[x <= portion_x] = 0
    y[y > portion_y] = 1
    y[y <= portion_y] = 0
    
    cells = x.shape[0]
    cells_x = np.sum(x, axis=0)
    cells_y = np.sum(y, axis=0)
    rho_x = cells_x*(cells_x - 1)/(cells*(cells - 1))
    rho_y = cells_y*(cells_y - 1)/(cells*(cells - 1))

    Px = np.zeros([maxdist,N])
    Py = np.zeros([maxdist,N])
    for j in range(N):
        ind_x = np.where(x[:,j]==1)[0]
        pairs_x = GraphDist[ind_x].T[ind_x].T
        
        ind_y = np.where(y[:,j]==1)[0]
        pairs_y = GraphDist[ind_y].T[ind_y].T

        for i in range(1,maxdist+1):
            Px[i-1,j] = len(pairs_x[pairs_x==i])/len(GraphDist[GraphDist==i])/rho_x[j]
            Py[i-1,j] = len(pairs_y[pairs_y==i])/len(GraphDist[GraphDist==i])/rho_y[j]

    Px_mean = np.mean(Px, axis=1)
    Py_mean = np.mean(Py, axis=1)
    
    return Px_mean, Py_mean


def moran_bounds(Pop, GraphDist, N, portion_x = 1):
    x = np.zeros([len(Pop), N])
    x[(Pop == 'N+G-')] = 1
    x[(Pop == 'N+G+') | (Pop == 'N-G-')] = np.random.random(x[(Pop == 'N+G+') | (Pop == 'N-G-')].shape)
    
    if portion_x == 1:
        portion_x = len(x[(Pop == 'N-G+')])/len(x[(Pop == 'N-G+') | (Pop == 'N+G-')])
    portion_y = 1 - portion_x
        
    x[x > portion_x] = 1
    x[x <= portion_x] = 0
    
    I = np.empty(N)
    for j in range(N):
        W = np.copy(GraphDist)
        W[W > 1] = 0
        y = x[:,j] - x[:,j].mean()

        numerator = np.dot(y, np.dot(W, y))
        denominator = np.sum(y**2)

        I[j] = len(y)/np.sum(W)*numerator/denominator

    I_max = np.max(I)
    I_min = np.min(I)
    
    return I_min, I_max
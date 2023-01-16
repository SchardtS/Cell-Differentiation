import numpy as np
import matplotlib.pyplot as plt
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Functions import loadData, fate, paircorrelation
import networkx as nx

""" N = np.empty([9,177])
G = np.empty([9,177])

Pos, Radius, N[0,:], G[0,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=1_10/Data.csv')
Pos, Radius, N[1,:], G[1,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=2_10/Data.csv')
Pos, Radius, N[2,:], G[2,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=3_10/Data.csv')
Pos, Radius, N[3,:], G[3,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=4_10/Data.csv')
Pos, Radius, N[4,:], G[4,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=5_10/Data.csv')
Pos, Radius, N[5,:], G[5,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=6_10/Data.csv')
Pos, Radius, N[6,:], G[6,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=7_10/Data.csv')
Pos, Radius, N[7,:], G[7,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=8_10/Data.csv')
Pos, Radius, N[8,:], G[8,:] = loadData('Results/Publications/Pattern Formation/Cell Fate - q=9_10/Data.csv')

FVmesh = initializeFVmesh(Pos, Radius=Radius)
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

plt.figure()
for i in range(len(N)):
    x = np.array(fate(N[i,:], G[i,:]))
    y = np.array(fate(G[i,:], N[i,:]))
    maxdist = int(np.max(GraphDist))
    ind = np.where(x==1)[0]
    dist = GraphDist[ind].T[ind].T
    rho0 = sum(x)/len(x)
    rho1 = (sum(x)-1)/(len(x)-1)

    Px = np.empty(maxdist)
    for k in range(1,maxdist+1):
        Px[k-1] = len(dist[dist==k])/len(GraphDist[GraphDist==k])/rho0/rho1
        
    plt.rc('font', size=14)
    distances = [j for j in range(1,int(np.max(GraphDist))+1)]
    plt.plot(distances, Px, lw=2, label='$q = '+str((i+1)/10)+'$')
    plt.xlabel('Distance')
    plt.ylabel('$\\rho_n$')

plt.axhline(1, color='k', lw=2, linestyle='--')
plt.legend(ncol=2)

plt.figure()
for i in range(len(N)):
    x = np.array(fate(G[i,:], N[i,:]))
    maxdist = int(np.max(GraphDist))
    ind = np.where(x==1)[0]
    dist = GraphDist[ind].T[ind].T
    rho0 = sum(x)/len(x)
    rho1 = (sum(x)-1)/(len(x)-1)

    Px = np.empty(maxdist)
    for k in range(1,maxdist+1):
        Px[k-1] = len(dist[dist==k])/len(GraphDist[GraphDist==k])/rho0/rho1
        
    plt.rc('font', size=14)
    distances = [j for j in range(1,int(np.max(GraphDist))+1)]
    plt.plot(distances, Px, lw=2, label='$q = '+str((i+1)/10)+'$')
    plt.xlabel('Distance')
    plt.ylabel('$\\rho_g$')

plt.axhline(1, color='k', lw=2, linestyle='--')
plt.legend(ncol=2)
plt.show() """


""" Pos_small, Radius, N_small, G_small = loadData('Results/Publications/Pattern Formation/Cell Fate - 23 to 70 q=9_10/Data.csv')
Pos      , Radius, N,       G       = loadData('Results/Publications/Pattern Formation/Cell Fate - 88 to 89 q=9_10/Data.csv')
Pos_large, Radius, N_large, G_large = loadData('Results/Publications/Pattern Formation/Cell Fate - 207 to 117 q=9_10/Data.csv')

plt.figure()
Radius = np.ones(len(Pos_small))*1.1
FVmesh1 = initializeFVmesh(Pos_small, Radius=Radius)
FVmesh1.plot(N_small, size=1000/len(Pos_large))
bmin = min(min(Pos_large[:,0])*1.3,min(Pos_large[:,1])*1.3)
bmax = max(max(Pos_large[:,0])*1.3,max(Pos_large[:,1])*1.3)
plt.xlim(bmin, bmax)
plt.ylim(bmin, bmax)
#plt.savefig('NANOG_1_10_small.pdf')
#plt.savefig('NANOG_1_10_small.png')

plt.figure()
Radius = np.ones(len(Pos))*1.1
FVmesh2 = initializeFVmesh(Pos, Radius=Radius)
FVmesh2.plot(N, size=1000/len(Pos_large))
bmin = min(min(Pos_large[:,0])*1.3,min(Pos_large[:,1])*1.3)
bmax = max(max(Pos_large[:,0])*1.3,max(Pos_large[:,1])*1.3)
plt.xlim(bmin, bmax)
plt.ylim(bmin, bmax)
#plt.savefig('NANOG_1_10_mid.pdf')
#plt.savefig('NANOG_1_10_mid.png')

plt.figure()
Radius = np.ones(len(Pos_large))*1.1
FVmesh3 = initializeFVmesh(Pos_large, Radius=Radius)
FVmesh3.plot(N_large, size=1000/len(Pos_large))
bmin = min(min(Pos_large[:,0])*1.3,min(Pos_large[:,1])*1.3)
bmax = max(max(Pos_large[:,0])*1.3,max(Pos_large[:,1])*1.3)
plt.xlim(bmin, bmax)
plt.ylim(bmin, bmax)
#plt.savefig('NANOG_1_10_large.pdf')
#plt.savefig('NANOG_1_10_large.png')

plt.figure()
paircorrelation(N_large,G_large,FVmesh3)
paircorrelation(N_small,G_small,FVmesh1,ls='dotted')
paircorrelation(N,G,FVmesh2,ls='dashed')
plt.axhline(1, color = 'k', linestyle='dashed', lw=2)
plt.legend(['NANOG', 'GATA6'])
plt.savefig('pair_correlation_9_10_size.pdf')
plt.savefig('pair_correlation_9_10_size.png')


plt.show() """

import numpy as np
import matplotlib.pyplot as plt
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Functions import coverPlot, saveData, paircorrelation
from Model import rhs_diffusion
from Parameters import setParameters
from scipy.integrate import solve_ivp
import pandas as pd

Prm = setParameters()
Prm.eps_N = -6.25
Prm.eps_G = -6
Prm.eps_S = -2
Prm.range = 0.9
Pos = np.array(pd.read_csv('testOrganoid.csv'))
Radius = np.ones(len(Pos))*1.1
FVmesh = initializeFVmesh(Pos, Radius=Radius)
plt.hist(FVmesh.Vol, 20, edgecolor='black')
plt.show()

t = np.linspace(0,Prm.T,Prm.nofSteps)

x0 = [Prm.r_N/Prm.gamma_N*3/4, Prm.r_G/Prm.gamma_G*3/4, Prm.r_S/Prm.gamma_S*3/4]
xInit = np.random.normal(x0[0], x0[0]*0.01, 3*FVmesh.nofCells)

rhs = lambda t,x: rhs_diffusion(0, x, Prm, FVmesh)
sol = solve_ivp(rhs, [0,Prm.T], xInit, t_eval = t, method = 'Radau')

N = sol.y[:FVmesh.nofCells,-1]
G = sol.y[FVmesh.nofCells:2*FVmesh.nofCells,-1]
S = sol.y[2*FVmesh.nofCells:,-1]

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

plt.title('SIGNAL')
plt.xlabel('Time')
plt.ylabel('Concentrations')

print('Number of Cells =', FVmesh.nofCells)
print('Number of NANOG Cells =', len(N[N>G]))
print('Number of GATA6 Cells =', len(G[G>N]))
print(Prm.dt*Prm.D)

plt.figure()
FVmesh.plot(N)
plt.show()


Prm.eps_N = -8
Prm.eps_G = -6
Prm.eps_S = -2
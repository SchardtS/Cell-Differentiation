import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import matplotlib as mpl
import matplotlib.cm as cm
from Parameters import Parameters
from matplotlib.animation import FuncAnimation
import itertools
import networkx as nx
from scipy.spatial import Delaunay


class Organoid(Parameters):
    def __init__(self):
        Parameters.__init__(self)
        
    def initialConditions(self, file = None):
        if file == None:
            self.xyz = np.array([[-0.5,-0.5,-0.3], [0.5,-0.5,-0.3], [0,0.5,-0.3], [0,0.25,0.3]])
            self.r = np.array([0.8, 0.9, 0.75, 0.83])
            
        else:
            Data = pd.read_csv(file)
            if 'z-Position' in Data:
                self.xyz = Data[['x-Position','y-Position','z-Position']].to_numpy()
            else:
                self.xyz = Data[['x-Position','y-Position']].to_numpy()
            self.r = Data['Radius'].to_numpy()

        self.nofCells = len(self.r)
        self.t = 0
        self.r0 = self.r
        self.t0 = np.zeros(self.nofCells)
        N0 = self.r_N/self.gamma_N*3/4
        G0 = self.r_N/self.gamma_N*3/4
        self.u = np.append(np.random.normal(N0, N0*0.01, self.nofCells),
                            np.random.normal(G0, G0*0.01, self.nofCells))
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:]
        self.Data = []
        self.dist = cdist(self.xyz, self.xyz)
        
    def radiusGrowth(self):
        self.r_old = self.r
        self.r = self.r_max - np.exp(-self.k*(self.t-self.t0))*(self.r_max - self.r0)
        
    def divisionProbability(self, r):
        c = 100/self.r_max
        y = 0.95*self.r_max
        r_min = 0.9*self.r_max
        b = (1 + np.exp(c*(self.r_max-y)))*(1 + np.exp(c*(r_min-y)))/(np.exp(c*(r_min-y)) - np.exp(c*(self.r_max-y)))
        a = -b/(1 + np.exp(c*(r_min-y)))
        P = a+b/(1 + np.exp(c*(r-y)))

        return np.maximum(0,P)
    
    def cellDivision(self):
        P0 = self.divisionProbability(self.r_old)       
        P = self.divisionProbability(self.r)

        Prob = (P-P0)/(1-P0)
        Prob[self.r_old == self.r0] = P[self.r_old == self.r0]

        # Choose where division will randomly occur
        random_numbers = np.random.rand(self.nofCells)
        indices = np.where(random_numbers < Prob)

        # New radius based on the area of the mother cell being two times that of the daughter cells
        # Use volume in 3D instead
        r_new = self.r[indices]/2**(1/3)
        N_new = self.N[indices]/2
        G_new = self.G[indices]/2

        # Distance between the two daughter cells
        dist = np.random.normal((self.r[indices] - r_new)/2, 0.1*(self.r[indices] - r_new)/2)

        # Angles of cell division
        angle1 = np.random.rand(len(indices))*np.pi
        angle2 = np.random.rand(len(indices))*2*np.pi

        # Displacement vectors
        dx = dist*np.sin(angle1)*np.cos(angle2)
        dy = dist*np.sin(angle1)*np.sin(angle2)
        dz = dist*np.cos(angle1)

        # Displacement
        dxyz = np.array([dx,dy,dz]).T
        xyz1 = self.xyz[indices] + dxyz
        xyz2 = self.xyz[indices] - dxyz

        # Change x-y-position to new value and add new cell to array
        self.xyz[indices] = xyz1
        self.xyz = np.append(self.xyz, xyz2, axis=0)

        # Update radii and include radii for new cells
        self.r[indices] = r_new
        self.r = np.append(self.r, r_new)

        # Change initial radius to value directly after division and include new cells to array
        self.r0[indices] = r_new
        self.r0 = np.append(self.r0, r_new)

        # Change initial radius to value directly after division and include new cells to array
        self.t0[indices] = self.t
        self.t0 = np.append(self.t0, self.t0[indices])
                               
        # Include new cells to array
        self.N[indices] = N_new
        self.N = np.append(self.N, N_new)
        self.G[indices] = G_new
        self.G = np.append(self.G, G_new)
        self.u = np.append(self.N, self.G)
    
        # Change new number of cells
        self.nofCells = len(self.r)

    def displacement(self):
        
        self.dist = cdist(self.xyz, self.xyz)

        x = self.xyz[:,0]
        y = self.xyz[:,1]
        z = self.xyz[:,2]

        # Pairwise sum of radii and difference of coordinates
        r_pairwise = self.r + self.r[:,None]
        x_pairwise = x - x[:,None]
        y_pairwise = y - y[:,None]
        z_pairwise = z - z[:,None]

        # Absolute values of forces according to Morse potential
        F = self.F0*2*self.alpha*(np.exp(-self.alpha*(self.dist-r_pairwise*self.sigma)) - np.exp(-2*self.alpha*(self.dist-r_pairwise*self.sigma)))
        F[self.dist > r_pairwise] = 0

        # Fill distance matrix with inf on diagonal
        dist = self.dist*1
        np.fill_diagonal(dist, np.inf)

        # x- and y-direction of forces
        Fx = F*(x_pairwise)/dist
        Fy = F*(y_pairwise)/dist
        Fz = F*(z_pairwise)/dist

        # Sum of all forces acting on each cell as a vector
        Force = np.array([np.sum(Fx, axis=1), np.sum(Fy, axis=1), np.sum(Fz, axis=1)]).T
        
        self.xyz = self.xyz + self.dt*Force# + 0.1*np.random.normal(0, self.dt**(1/2), self.xyz.shape)

    def graphdistance(self):
        Gr = nx.Graph()
        self.dist = cdist(self.xyz, self.xyz)
        rr = self.r + self.r[:,None]
        tri = Delaunay(self.xyz)
                               
        for nodes in tri.simplices:
            for path in list(itertools.combinations(nodes, 2)):
                if self.dist[path[0],path[1]] < rr[path[0],path[1]]:
                    nx.add_path(Gr, path)

        dist_dict = dict(nx.all_pairs_dijkstra_path_length(Gr))
        self.GraphDist = np.empty([self.nofCells, self.nofCells])
        for i in range(self.nofCells):
            for j in range(self.nofCells):
                self.GraphDist[i,j] = dist_dict[i][j]

        #self.GraphDist = np.floor(self.dist/np.mean(2*self.r))
  
    def transcription(self):
        rhs = np.empty(self.nofCells*2)

        b = np.exp(-self.eps_N)
        a = np.exp(-self.eps_G)
        c = np.exp(-self.eps_S)
        d = np.exp(-self.eps_GS)

        d_ij = np.maximum(self.GraphDist-1, 0)         
        scaling = self.q**(d_ij)
        np.fill_diagonal(scaling, 0)
        val = self.N*scaling#*(1-self.q)/self.q
        np.fill_diagonal(val, 0)
        self.S = val.sum(1)/max(scaling.sum(1))

        pN =        (b*self.N)        /(1 + a*self.G*(1+d*c*self.S) + b*self.N + c*self.S)
        pG = (a*self.G)*(1+d*c*self.S)/(1 + a*self.G*(1+d*c*self.S) + b*self.N + c*self.S)

        #pN = (a*self.N)*(1+d*c*self.S)/(1 + a*self.N*(1+d*c*self.S) + b*self.G + c*self.S)
        #pG =        (b*self.G)        /(1 + a*self.N*(1+d*c*self.S) + b*self.G + c*self.S)

        rhs[:self.nofCells] = self.r_N*pN - self.gamma_N*self.N
        rhs[self.nofCells:] = self.r_G*pG - self.gamma_G*self.G
                               
        self.u = self.u + self.dt*rhs
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:]
                                      
    def cellPlot(self, *Val, size = None, bounds = None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        size = 30000/len(self.xyz)*self.r
        if Val == ():
            ax.scatter(self.xyz[:,0], self.xyz[:,1], self.xyz[:,2], c = 'k', s=size)

        else: 
            ax.scatter(self.xyz[:,0], self.xyz[:,1], self.xyz[:,2], c=Val, cmap = 'cool', s=size)
        
        ax.axis('off')

    def timePlot(self):

        plt.figure()
        for i in range(self.nofCells):
            t = []
            NANOG = []
            for k in range(int(self.T/self.dt)):
                if len(self.Data[k][2])-1 >= i:
                    NANOG.append(self.Data[k][2][i])
                    t.append(k*self.dt)
            plt.plot(t, NANOG)

        plt.title('NANOG')
        plt.xlabel('Time')
        plt.ylabel('Concentrations')

    def pcf(self, ls = 'solid', lw = 2, plot = True):
        x = np.zeros(self.nofCells)
        x[self.N > self.G] = 1
        maxdist = int(np.max(self.GraphDist))
        ind_N = np.where(x==1)[0]
        ind_G = np.where(x==0)[0]
        if ind_N.size == 0:
            self.pcf_N = np.empty(maxdist)
            for i in range(1,maxdist+1):
                self.pcf_N[i-1] = 0
        if ind_G.size == 0:
            self.pcf_G = np.empty(maxdist)
            for i in range(1,maxdist+1):
                self.pcf_G[i-1] = 0
                
        else:
            dist_N = self.GraphDist[ind_N].T[ind_N].T
            rho_N = sum(x)/len(x)*(sum(x)-1)/(len(x)-1)

            dist_G = self.GraphDist[ind_G].T[ind_G].T
            rho_G = (len(x)-sum(x))/len(x)*((len(x)-sum(x))-1)/(len(x)-1)

            self.pcf_N = np.empty(maxdist)
            self.pcf_G = np.empty(maxdist)
            for i in range(1,maxdist+1):
                self.pcf_N[i-1] = len(dist_N[dist_N==i])/len(self.GraphDist[self.GraphDist==i])/rho_N
                self.pcf_G[i-1] = len(dist_G[dist_G==i])/len(self.GraphDist[self.GraphDist==i])/rho_G

        if plot == True:
            plt.rc('font', size=14)
            plt.plot(range(1,maxdist+1), self.pcf_N, color='m', lw = lw, ls = ls, label = 'N+G-')
            plt.plot(range(1,maxdist+1), self.pcf_G, color='c', lw = lw, ls = ls, label = 'N-G+')
            plt.axhline(1, ls='dashed', color='k')

    def moran(self):
        x = np.zeros(self.N.shape)
        x[self.N > self.G] = 1
        
        W = np.copy(self.GraphDist)
        W[W > 1] = 0
        y = x - x.mean()

        numerator = np.dot(y, np.dot(W, y))
        denominator = np.sum(y**2)

        self.Morans_I = self.nofCells/np.sum(W)*numerator/denominator

    def collectData(self):
        self.Data.append([self.xyz,self.r,self.N,self.G])
        return

    def saveData(self, directory = ''):
        
        # Create directory if not existent
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Save plot of geometry
        plt.figure(figsize=[6.4, 4.8])
        self.cellPlot()
        plt.savefig(directory + 'tissue.png', transparent = True) 
        plt.savefig(directory + 'tissue.pdf', transparent = True)   

        # Save plot of NANOG
        plt.figure(figsize=[6.4, 4.8])
        self.cellPlot(self.N)
        plt.savefig(directory + 'NANOG.png', transparent = True) 
        plt.savefig(directory + 'NANOG.pdf', transparent = True)   

        # Save plot of GATA6
        plt.figure(figsize=[6.4, 4.8])
        self.cellPlot(self.G)
        plt.savefig(directory + 'GATA6.png', transparent = True) 
        plt.savefig(directory + 'GATA6.pdf', transparent = True)

        # Save organoid Data
        df = pd.DataFrame()
        df['x-Position'] = self.xyz[:,0]
        df['y-Position'] = self.xyz[:,1]
        df['z-Position'] = self.xyz[:,2]
        df['Radius'] = self.r
        df['NANOG'] = self.N
        df['GATA6'] = self.G
        df.to_csv(directory + 'Data.csv', index = False)
        
        # Save all parameters in .txt file
        with open(directory + 'Parameters.txt', 'w') as f:
            f.write(''.join(["%s = %s\n" % (k,v) for k,v in self.__dict__.items() if not hasattr(v, '__iter__')]))

    def saveAnim(self, directory = '', frames = None, fps = 60):

        fig = plt.figure()
        bmin = min(min(self.xyz[:,0]),min(self.xyz[:,1])) - 1.5*self.r_max
        bmax = max(max(self.xyz[:,0]),max(self.xyz[:,1])) + 1.5*self.r_max


        def update(i):
            plt.cla()

            org = Organoid()
            org.nofCells = len(self.Data[i][0])
            org.xyz = self.Data[i][0]
            org.r = self.Data[i][1]
            org.N = self.Data[i][2]
            org.dist = cdist(org.xyz, org.xyz)

            org.cellPlot(org.N, size=1000/self.nofCells, bounds=[min(self.N),max(self.N)])
            plt.xlim(bmin, bmax)
            plt.ylim(bmin, bmax)
            plt.gca().set_adjustable("box")
            return

        if frames == None:
            frames = len(self.Data)
        else:
            frames = np.unique(np.linspace(0, len(self.Data)-1, frames, dtype=int))

        ani = FuncAnimation(fig, update, frames=frames)
        ani.save(directory + '/NANOG.mp4', fps=fps, dpi=200, savefig_kwargs={'transparent': True, 'facecolor': 'none'})

        return

    def evolution(self, T = 0, file = None, mode = 'transcription + geometry'):
        if T != 0:
            self.T = T

        self.dt = self.T/(self.nofSteps - 1)

        if not hasattr(self, 'xyz'):
            self.initialConditions(file = file)
        
        if mode == 'geometry':
            for i in range(self.nofSteps):
                self.t += self.dt
                self.radiusGrowth()
                self.cellDivision()
                self.displacement()
                self.collectData()
                
        if mode == 'transcription':
            self.graphdistance()
            for i in range(self.nofSteps):
                self.t += self.dt
                self.transcription()
                #self.collectData()
                
        if mode == 'transcription + geometry':
            for i in range(self.nofSteps):
                self.t += self.dt
                self.radiusGrowth()
                self.cellDivision()
                self.displacement()
                self.graphdistance()
                self.transcription()
                self.collectData()
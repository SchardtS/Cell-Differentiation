from math import gamma
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import matplotlib as mpl
import matplotlib.cm as cm
from Parameters import Parameters
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
from scipy.spatial import Delaunay
import itertools


class Organoid(Parameters):
    def __init__(self):
        Parameters.__init__(self)
        
    def initialConditions(self, file = None):
        if file == None:
            self.xy = np.array([[-0.5,-0.5], [0.5,-0.5], [0,0.5]])
            self.r = np.array([0.8, 0.9, 0.75])
            
        else:
            Data = pd.read_csv(file)
            if 'z-Position' in Data:
                self.xy = Data[['x-Position','y-Position','z-Position']].to_numpy()
            else:
                self.xy = Data[['x-Position','y-Position']].to_numpy()
            self.r = Data['Radius'].to_numpy()

        self.nofCells = len(self.r)
        self.t = 0
        self.r0 = self.r
        self.t0 = np.zeros(self.nofCells)
        N0 = self.r_N/self.gamma_N*3/4
        G0 = self.r_G/self.gamma_G*3/4
        self.u = np.append(np.random.normal(N0, N0*0.01, self.nofCells),
                           np.random.normal(G0, G0*0.01, self.nofCells))
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:]

        self.Data = []
        self.dist = cdist(self.xy, self.xy)
        
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
        r_new = self.r[indices]/2**(1/2)
        N_new = self.N[indices]/2
        G_new = self.G[indices]/2

        # Distance between the two daughter cells
        dist = np.random.normal((self.r[indices] - r_new)/2, 0.1*(self.r[indices] - r_new)/2)

        # Angles of cell division
        angle = np.random.rand(len(indices))*2*np.pi

        # Displacement vectors
        dx = dist*np.cos(angle)
        dy = dist*np.sin(angle)

        # Displacement
        dxy = np.array([dx,dy]).T
        xy1 = self.xy[indices] + dxy
        xy2 = self.xy[indices] - dxy

        # Change x-y-position to new value and add new cell to array
        self.xy[indices] = xy1
        self.xy = np.append(self.xy, xy2, axis=0)

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
        self.dist = cdist(self.xy, self.xy)

        x = self.xy[:,0]
        y = self.xy[:,1]

        # Pairwise sum of radii and difference of coordinates
        r_pairwise = self.r + self.r[:,None]
        x_pairwise = x - x[:,None]
        y_pairwise = y - y[:,None]

        # Absolute values of forces according to Morse potential
        F = self.F0*2*self.alpha*(np.exp(-self.alpha*(self.dist-r_pairwise*self.sigma)) - np.exp(-2*self.alpha*(self.dist-r_pairwise*self.sigma)))
        F[self.dist > r_pairwise] = 0

        # Fill distance matrix with inf on diagonal
        dist = self.dist*1
        np.fill_diagonal(dist, np.inf)

        # x- and y-direction of forces
        Fx = F*(x_pairwise)/dist
        Fy = F*(y_pairwise)/dist

        # Sum of all forces acting on each cell as a vector
        Force = np.array([np.sum(Fx, axis=1), np.sum(Fy, axis=1)]).T
        
        self.xy = self.xy + self.dt*Force# + 0.1*np.random.normal(0, self.dt**(1/2), self.xy.shape)
        
    def graphdistance(self):
        Gr = nx.Graph()
        self.dist = cdist(self.xy, self.xy)
        rr = self.r + self.r[:,None]
        tri = Delaunay(self.xy)
                               
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


    def communication(self):
        d_ij = np.maximum(self.GraphDist-1, 0)
        #scaling = self.q**(d_ij)
        scaling = self.GraphDist[:]
        scaling[scaling > 1] = 0
        scaling[scaling < 1] = -2
        self.A = (scaling.T/max(scaling.sum(0))).T
        #self.A = (scaling.T/scaling.sum(0)).T
        #np.fill_diagonal(self.A, 0)

    def signal(self):
        s_crit = (self.r_N*self.gamma_G*np.exp(-self.eps_N) - self.r_G*self.gamma_N*np.exp(-self.eps_G)) / \
                 (self.r_G*self.gamma_N*np.exp(-self.eps_G)*np.exp(-self.eps_S)*np.exp(-self.eps_GS))

        d_ij = np.maximum(self.GraphDist-1, 0)
        scaling = self.q**(d_ij)
        np.fill_diagonal(scaling, 0)

        s_crit = 0.1
        val = (s_crit - (self.gamma_G/self.r_G)*self.G*s_crit)*scaling
        #val = (s_crit + (self.gamma_N/self.r_N*self.N - self.gamma_G/self.r_G*self.G)*s_crit)*scaling
        np.fill_diagonal(val, 0)
        #self.S = val.sum(1)/max(scaling.sum(1))
        self.S = val.sum(1)/scaling.sum(1)

    def transcription(self):
        rhs = np.empty(self.nofCells*2)

        a = np.exp(-self.eps_G)
        b = np.exp(-self.eps_N)
        c = np.exp(-self.eps_S)
        d = np.exp(-self.eps_GS)

        #self.signal()
        self.communication()
        self.S = .5*np.dot(self.A, self.N)

        pN =        (b*self.N)        /(1 + a*self.G*(1+d*c*self.S) + b*self.N + c*self.S)
        pG = (a*self.G)*(1+d*c*self.S)/(1 + a*self.G*(1+d*c*self.S) + b*self.N + c*self.S)

        rhs[:self.nofCells] = self.r_N*pN - self.gamma_N*self.N
        rhs[self.nofCells:] = self.r_G*pG - self.gamma_G*self.G
                               
        self.u = self.u + self.dt*rhs
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:]

    """ def transcription(self):
        gamma = 10
        r = 1

        x = np.exp(-1)
        eta_x = np.exp(1)

        p = lambda f: eta_x*x/(f + eta_x*x)

        n0 = p(1)*r/gamma
        g0 = p(1)*r/gamma
        s0 = p(1)*r/gamma

        eta_n = np.exp(6)
        eta_g = np.exp(6)
        eta_s = np.exp(6)

        eta_nx = np.exp(4)
        eta_gx = np.exp(2)
        eta_sx = np.exp(2)

        d_ij = np.maximum(self.GraphDist-1, 0)
        scaling = self.q**(d_ij)
        np.fill_diagonal(scaling, 0)
        s_crit = .036*3
        val = np.empty(self.nofCells)
        val[self.G > 0.05] = np.exp(-3.3)
        val[self.G <= 0.05] = np.exp(-10)
        #val = (s_crit - (gamma/r)*self.G*s_crit)*scaling
        np.fill_diagonal(val, 0)
        s = val.sum(1)/max(scaling.sum(1))
        self.s = s

        f_n = lambda n, g: (1 + eta_n*n)*(1 + eta_g*g)*(1 + eta_s*s)/(1 + eta_n*eta_nx*n)
        f_g = lambda n, g: (1 + eta_n*n)*(1 + eta_g*g)*(1 + eta_s*s)/(1 + eta_g*eta_gx*g)/(1 + eta_s*eta_sx*s)
        p = lambda f: eta_x*x/(f + eta_x*x)

        self.u[:self.nofCells] = self.N + self.dt*(r*p(f_n(self.N,self.G)) - gamma*self.N)
        self.u[self.nofCells:] = self.G + self.dt*(r*p(f_g(self.N,self.G)) - gamma*self.G)
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:] """

    """ def transcription(self):
        rhs = np.empty(self.nofCells*3)

        a_n = 2.5
        a_ns = 0.5
        a_g = 3
        a_s = 3

        d_ij = np.maximum(self.GraphDist-1, 0)
        #distance = 50
        #d_ij[d_ij <= distance] = 1
        #d_ij[d_ij > distance] = 0

        #S = np.empty(self.nofCells)
        #for i in range(self.nofCells):
        #    S[i] = sum(self.S*d_ij[:,i])/(sum(d_ij[:,i]))

        scaling = self.q**(d_ij)
        np.fill_diagonal(scaling, 0)
        val = self.S*scaling#*(1-self.q)/self.q
        np.fill_diagonal(val, 0)
        S = val.sum(1)/max(scaling.sum(1))

        rhs[:self.nofCells] = a_n/(1 + self.G**2) + a_ns/(1 + S**2) - self.N
        rhs[self.nofCells:2*self.nofCells] = a_g/(1 + self.N**2) - self.G
        rhs[2*self.nofCells:] = a_s/(1 + self.G**2) - self.S

        self.u = self.u + 10*self.dt*rhs
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:2*self.nofCells]
        self.S = self.u[2*self.nofCells:] """

    """ def transcription(self):
        rhs = np.empty(self.nofCells*2)

        a = np.exp(-self.eps_N)
        b = np.exp(-self.eps_G)
        c = np.exp(-self.eps_S)
        d = np.exp(-self.eps_NS)

        d_ij = np.maximum(self.GraphDist-1, 0)         
        scaling = self.q**(d_ij)
        np.fill_diagonal(scaling, 0)
        val = self.G*scaling#*(1-self.q)/self.q
        np.fill_diagonal(val, 0)
        self.S = val.sum(1)/max(scaling.sum(1))

        pN = (a*self.N)*(1+d*c*self.S)/(1 + a*self.N*(1+d*c*self.S) + b*self.G + c*self.S)
        pG =        (b*self.G)        /(1 + a*self.N*(1+d*c*self.S) + b*self.G + c*self.S)

        rhs[:self.nofCells] = self.r_N*pN - self.gamma_N*self.N
        rhs[self.nofCells:] = self.r_G*pG - self.gamma_G*self.G
                               
        self.u = self.u + self.dt*rhs
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:] """
                                   
    def cellPlot(self, *Val, size = None, bounds = None, radius = 'individual'):
        if size == None:
            size = 1000/len(self.xy)
        
        if radius == 'individual':
            r = self.r
        if radius == 'mean':
            r = self.r.mean()*np.ones(self.nofCells)

        #### polygon construction ####
        polygons = []
        cells = [Point(self.xy[i,:]).buffer(r[i]) for i in range(self.nofCells)]
        for i in range(self.nofCells):
            indices = np.where((self.dist[i,:] < r[i] + r[:]) & (self.dist[i,:] != 0))
            cell1 = cells[i]

            d = self.dist[i,indices[0]]
            r_neigh = r[indices] 
            a = (r[i]**2 - r_neigh**2 + d**2)/(2*d)
            d12 = self.xy[indices[0],:] - self.xy[i,:]
            d12_orth = np.array([d12[:,1],-d12[:,0]]).T

            rect1 = self.xy[i,:] + d12/d[:,None]*a[:,None] + d12_orth/d[:,None]*r[i]
            rect2 = self.xy[i,:] - d12/d[:,None]*r[i] + d12_orth/d[:,None]*r[i]
            rect3 = self.xy[i,:] - d12/d[:,None]*r[i] - d12_orth/d[:,None]*r[i]
            rect4 = self.xy[i,:] + d12/d[:,None]*a[:,None] - d12_orth/d[:,None]*r[i]

            for j in range(len(indices[0])):

                rectangle = np.array([rect1[j,:],rect2[j,:],rect3[j,:],rect4[j,:]])
                rectangle = Polygon(rectangle)

                cell1 = cell1.intersection(rectangle)

            polygons.append(cell1)    
        
        #### plot polygons ####
        if Val == ():
            for i in range(self.nofCells):
                x, y = polygons[i].exterior.xy
                plt.plot(x, y, 'k')
        
        #### color filling ####
        else:
            Val = Val[0]

            if bounds == None:
                bounds = [min(Val), max(Val)]
            
            norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1], clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='cool')

            for i in range(self.nofCells):
                x,y = polygons[i].exterior.xy
                plt.fill(x,y, facecolor=mapper.to_rgba(float(Val[i])), edgecolor='k', linewidth=1, zorder=1)
        
        #### plot cell nuclei ####
        plt.scatter(self.xy[:,0],self.xy[:,1], color='k', s=size, zorder=2)
        plt.axis('equal')
        plt.axis('off')

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

    def pcf(self, ls = 'solid', lw = 2, plot = True, font_size=14):
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
            plt.rc('font', size=font_size)
            plt.plot(range(1,maxdist+1), self.pcf_N, color='m', lw = lw, ls = ls)
            plt.plot(range(1,maxdist+1), self.pcf_G, color='c', lw = lw, ls = ls)
            plt.axhline(1, ls='dashed', color='k')
            plt.xlabel('Distance')
            plt.ylabel('$\\rho_n, \\rho_g$')

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
        self.Data.append([self.xy,self.r,self.N,self.G])
        return

    def saveData(self, directory = ''):
        
        # Create directory if not existent
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Save plot of geometry
        plt.figure(figsize=[6.4, 4.8])
        self.cellPlot(radius = 'mean')
        plt.savefig(directory + 'tissue.png', transparent = True) 
        plt.savefig(directory + 'tissue.pdf', transparent = True)   

        # Save plot of NANOG
        plt.figure(figsize=[6.4, 4.8])
        self.cellPlot(self.N, radius = 'mean')
        plt.savefig(directory + 'NANOG.png', transparent = True) 
        plt.savefig(directory + 'NANOG.pdf', transparent = True)   

        # Save plot of GATA6
        plt.figure(figsize=[6.4, 4.8])
        self.cellPlot(self.G, radius = 'mean')
        plt.savefig(directory + 'GATA6.png', transparent = True) 
        plt.savefig(directory + 'GATA6.pdf', transparent = True)

        # Save organoid Data
        df = pd.DataFrame()
        df['x-Position'] = self.xy[:,0]
        df['y-Position'] = self.xy[:,1]
        df['Radius'] = self.r
        df['NANOG'] = self.N
        df['GATA6'] = self.G
        df.to_csv(directory + 'Data.csv', index = False)
        
        # Save all parameters in .txt file
        with open(directory + 'Parameters.txt', 'w') as f:
            f.write(''.join(["%s = %s\n" % (k,v) for k,v in self.__dict__.items() if not hasattr(v, '__iter__')]))

    def saveAnim(self, directory = '', frames = None, fps = 60):

        fig = plt.figure()
        bmin = min(min(self.xy[:,0]),min(self.xy[:,1])) - 1.5*self.r_max
        bmax = max(max(self.xy[:,0]),max(self.xy[:,1])) + 1.5*self.r_max


        def update(i):
            plt.cla()

            org = Organoid()
            org.nofCells = len(self.Data[i][0])
            org.xy = self.Data[i][0]
            org.r = self.Data[i][1]
            org.N = self.Data[i][2]
            org.dist = cdist(org.xy, org.xy)

            org.cellPlot(org.N, size=1000/self.nofCells, bounds=[min(self.N),max(self.N)],radius='mean')
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

    def saveGIF(self, directory = '', frames = None, fps = 50, mode = None):

        fig = plt.figure()
        bmin = min(min(self.xy[:,0]),min(self.xy[:,1])) - 1.5*self.r_max
        bmax = max(max(self.xy[:,0]),max(self.xy[:,1])) + 1.5*self.r_max


        def update(i):
            plt.cla()

            org = Organoid()
            org.nofCells = len(self.Data[i][0])
            org.xy = self.Data[i][0]
            org.r = self.Data[i][1]
            org.N = self.Data[i][2]
            org.G = self.Data[i][3]
            org.dist = cdist(org.xy, org.xy)

            if mode == 'NANOG':
                org.cellPlot(org.N, size=1000/self.nofCells, bounds=[min(self.N),max(self.N)], radius='mean')
            if mode == 'GATA6':
                org.cellPlot(org.G, size=1000/self.nofCells, bounds=[min(self.G),max(self.G)], radius='mean')
            else:
                org.cellPlot(size=1000/self.nofCells,radius='mean')
            plt.xlim(bmin, bmax)
            plt.ylim(bmin, bmax)
            plt.gca().set_adjustable("box")
            return

        if frames == None:
            frames = len(self.Data)
        else:
            frames = np.unique(np.linspace(0, len(self.Data)-1, frames, dtype=int))

        ani = FuncAnimation(fig, update, frames=frames)
        writer = PillowWriter(fps=fps)
        ani.save(directory + '/NANOG.gif', writer=writer)

        return

    def evolution(self, T = 0, file = None, mode = 'transcription + geometry'):
        if T != 0:
            self.T = T       
        N = int(self.T/self.dt)

        if not hasattr(self, 'xy'):
            self.initialConditions(file = file)
        
        if mode == 'geometry':
            for i in range(N):
                self.t += self.dt
                self.radiusGrowth()
                self.cellDivision()
                self.displacement()
                self.collectData()
                
        if mode == 'transcription':
            self.graphdistance()
            for i in range(N):
                self.t += self.dt
                self.transcription()
                self.collectData()
                
        if mode == 'transcription + geometry':
            for i in range(N):
                self.t += self.dt
                self.radiusGrowth()
                self.cellDivision()
                self.displacement()
                self.graphdistance()
                self.transcription()
                self.collectData()
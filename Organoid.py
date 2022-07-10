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
from scipy.optimize import fsolve
from scipy.special import erf

def norm(x):
    return sum(x**2)**(1/2)

def truncated_normal_cdf(x, mu, sigma, interval):
    cdf = lambda y: (1 + erf(y/2**(1/2)))
    
    a = (interval[0] - mu)/sigma           
    b = (interval[1] - mu)/sigma
    z = (x - mu)/sigma
    
    return (cdf(z) - cdf(a)) / (cdf(b) - cdf(a))


class Organoid(Parameters):
    def __init__(self):
        Parameters.__init__(self)
        
    def initialConditions(self, dim, file = None):
        if file == None:
            if dim == 2:
                self.pos = np.array([[0,0]], dtype=float)
                self.dim = 2
            
            if dim == 3:
                self.pos = np.array([[0,0,0]], dtype=float)
                self.dim = 3

            self.r = np.array([0.75], dtype=float)
            
        else:
            Data = pd.read_csv(file)
            if 'z-Position' in Data:
                self.pos = Data[['x-Position','y-Position','z-Position']].to_numpy()
                self.dim = 3
            else:
                self.pos = Data[['x-Position','y-Position']].to_numpy()
                self.dim = 2
            self.r = Data['Radius'].to_numpy()

        self.nofCells = len(self.r)
        self.t = 0
        self.r0 = self.r
        self.t0 = np.zeros(self.nofCells, dtype=float)
        N0 = self.r_N/self.gamma_N*3/4
        G0 = self.r_G/self.gamma_G*3/4
        self.u = np.append(np.random.normal(N0, N0*0.01, self.nofCells),
                           np.random.normal(G0, G0*0.01, self.nofCells))
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:]

        self.Data = []

        if self.nofCells >= 2:
            self.dist = cdist(self.pos, self.pos)
        else:
            self.dist = np.array([0])
        
    def radiusGrowth(self):
        if 'division' in self.ignore:
            return
        else:
            self.r_old = self.r
            #self.r = self.r_max - np.exp(-self.k*(self.t-self.t0))*(self.r_max - self.r0)
            self.r = self.r_max/(1 + ((self.r_max - self.r0)/self.r0)*np.exp(-self.k*self.r_max*(self.t - self.t0)))
        
    def divisionProbability(self, r):
        if 'division' in self.ignore:
            return
        else:
            #c = 100/self.r_max
            #y = 0.95*self.r_max
            #r_min = 0.9*self.r_max
            #b = (1 + np.exp(c*(self.r_max-y)))*(1 + np.exp(c*(r_min-y)))/(np.exp(c*(r_min-y)) - np.exp(c*(self.r_max-y)))
            #a = -b/(1 + np.exp(c*(r_min-y)))
            #P = a+b/(1 + np.exp(c*(r-y)))

            P = truncated_normal_cdf(r, 0.95, 0.015, [0.9, 1])
        return np.maximum(0,P)
    
    def cellDivisionDistance(self):
        if 'division' in self.ignore:
            return
        else:
            r_exp = 0.95*self.r_max/2**(1/self.dim)                 # expected radius after cell division

            f = lambda h: 2*self.alpha*self.F0*self.dt \
                        + np.exp(self.alpha*(h-2*r_exp*self.sigma)) + np.log(1 - np.exp(self.alpha*(h-2*r_exp*self.sigma))) \
                        - np.exp(self.alpha*(-2*r_exp*self.sigma)) - np.log(1 - np.exp(self.alpha*(-2*r_exp*self.sigma)))

            df = lambda h: self.alpha*np.exp(self.alpha*(h-2*r_exp*self.sigma))/(np.exp(-self.alpha*(h-2*r_exp*self.sigma))-1)
            self.divDist = fsolve(f, 1.5*self.sigma*r_exp, fprime=df)[0]

    def cellDivision(self):
        if 'division' in self.ignore:
            return
        else:
            P0 = self.divisionProbability(self.r_old)       
            P = self.divisionProbability(self.r)

            Prob = (P-P0)/(1-P0)
            Prob[self.r_old == self.r0] = P[self.r_old == self.r0]

            # Choose where division will randomly occur
            random_numbers = np.random.rand(self.nofCells)
            indices = np.where(random_numbers < Prob)[0]
            if self.nofCells + len(indices) >= self.maxCells:
                indices = indices[:self.maxCells-self.nofCells]


            if len(indices) > 0:
                # New radius based on the area of the mother cell being two times that of the daughter cells
                # Use volume in 3D instead
                r_new = self.r[indices]/2**(1/self.dim)
                N_new = self.N[indices]/2
                G_new = self.G[indices]/2

                # Distance between the two daughter cells
                dist = np.random.normal(self.divDist/2, self.divDist/2*0.1, len(indices))

                if self.dim == 2:
                    # Angles of cell division
                    angle = np.random.rand(len(indices))*2*np.pi

                    # Displacement vectors
                    dx = dist*np.cos(angle)
                    dy = dist*np.sin(angle)

                    # Displacement
                    dxy = np.array([dx,dy]).T
                    pos1 = self.pos[indices] + dxy
                    pos2 = self.pos[indices] - dxy
                if self.dim == 3:
                    # Angles of cell division  
                    angle1 = np.random.rand(len(indices))*np.pi
                    angle2 = np.random.rand(len(indices))*2*np.pi

                    # Displacement vectors
                    dx = dist*np.sin(angle1)*np.cos(angle2)
                    dy = dist*np.sin(angle1)*np.sin(angle2)
                    dz = dist*np.cos(angle1)

                    # Displacement
                    dxyz = np.array([dx,dy,dz]).T
                    pos1 = self.pos[indices] + dxyz
                    pos2 = self.pos[indices] - dxyz

                # Change x-y-position to new value and add new cell to array
                self.pos[indices] = pos1
                self.pos = np.append(self.pos, pos2, axis=0)

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
        if 'displacement' in self.ignore:
            return
        else:
            self.dist = cdist(self.pos, self.pos)

            x = self.pos[:,0]
            y = self.pos[:,1]
            if self.dim == 3:
                z = self.pos[:,2]

            # Pairwise sum of radii and difference of coordinates
            r_pairwise = self.r + self.r[:,None]
            x_pairwise = x - x[:,None]
            y_pairwise = y - y[:,None]
            if self.dim == 3:
                z_pairwise = z - z[:,None]

            # Absolute values of forces according to Morse potential
            F = self.F0*2*self.alpha*(np.exp(-self.alpha*(self.dist-r_pairwise*self.sigma)) \
                                    - np.exp(-2*self.alpha*(self.dist-r_pairwise*self.sigma)))
            F[self.dist > r_pairwise] = 0

            # Fill distance matrix with inf on diagonal
            dist = self.dist*1
            np.fill_diagonal(dist, np.inf)

            # x- and y-direction of forces
            Fx = F*(x_pairwise)/dist
            Fy = F*(y_pairwise)/dist
            if self.dim == 3:
                Fz = F*(z_pairwise)/dist

            # Sum of all forces acting on each cell as a vector
            if self.dim == 2:
                Force = np.array([np.sum(Fx, axis=1), np.sum(Fy, axis=1)]).T
            if self.dim == 3:
                Force = np.array([np.sum(Fx, axis=1), np.sum(Fy, axis=1), np.sum(Fz, axis=1)]).T

            self.pos = self.pos + self.dt*Force# + 0.1*np.random.normal(0, self.dt**(1/2), self.pos.shape)
      
    def graphdistance(self):
        if 'transcription' in self.ignore:
            return
        else:
            Gr = nx.Graph()
            self.dist = cdist(self.pos, self.pos)
            rr = self.r + self.r[:,None]
            tri = Delaunay(self.pos)
                                
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
        if 'transcription' in self.ignore:
            return
        else:
            if self.signal == 'dispersion':
                d_ij = np.maximum(self.GraphDist-1, 0)
                scaling = self.q**(d_ij)
                np.fill_diagonal(scaling, 0)
                self.A = (scaling.T/max(scaling.sum(0))).T
            elif self.signal == 'neighbor':
                scaling = self.GraphDist.copy()
                #scaling[scaling <= 1] = 1
                scaling[scaling != 1] = 0
                self.A = (scaling.T/scaling.sum(0)).T
            else:
                print('ERROR: signal parameter must be either \'neighbor\' or \'dispersion\'')

    def transcription(self):
        if 'transcription' in self.ignore:
            return
        else:
            rhs = np.empty(self.nofCells*2)

            a = np.exp(-self.eps_G)
            b = np.exp(-self.eps_N)
            c = np.exp(-self.eps_S)
            d = np.exp(-self.eps_GS)

            self.S = np.dot(self.A, self.N)

            pN =        (b*self.N)        /(1 + a*self.G*(1+d*c*self.S) + b*self.N + c*self.S)
            pG = (a*self.G)*(1+d*c*self.S)/(1 + a*self.G*(1+d*c*self.S) + b*self.N + c*self.S)

            rhs[:self.nofCells] = self.r_N*pN - self.gamma_N*self.N
            rhs[self.nofCells:] = self.r_G*pG - self.gamma_G*self.G
                                
            self.u = self.u + self.dt*rhs
            self.N = self.u[:self.nofCells]
            self.G = self.u[self.nofCells:]

    def angle_sorted_neighbors(self, i):
        indices = np.where((self.dist[i,:] < self.r[i] + self.r) & (self.dist[i,:] != 0))[0]
        diff = self.pos[i,:] - self.pos[indices,:]
    
        angles = np.arctan2(diff[:,1],diff[:,0])
        sorted_indices = np.array([ind for _, ind in sorted(zip(angles, indices))])

        return sorted_indices

    def circle_intersections(self, i):
        indices = self.angle_sorted_neighbors(i)
        intersections = []
        for j in indices:
            d = self.dist[i,j]
            a = (self.r[i]**2 - self.r[j]**2 + d**2)/(2*d)
            b = (self.r[i]**2 - a**2)**(1/2)
            d12 = self.pos[j,:] - self.pos[i,:]
            d12_orth = np.array([d12[1],-d12[0]]).T
            xy1 = self.pos[i,:] + a*d12/d - b*d12_orth/d
            xy2 = self.pos[i,:] + a*d12/d + b*d12_orth/d
            
            intersections.append(xy1)
            intersections.append(xy2)

        return np.array(intersections)

    def polygon_corners(intersections):
        corners = []
        edges = {}
        for i in range(0,len(intersections),2):
            j = (i+2) % len(intersections)
            x1,y1 = intersections[i,:]
            x2,y2 = intersections[i+1,:]
            x3,y3 = intersections[j,:]
            x4,y4 = intersections[j+1,:]
            
            beta = ((x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)) / \
                   ((y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1))
        
            alpha = (x3 - x1)/(x2 - x1) +  (x4 - x3)/(x2 - x1)*beta
            
            if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1:
                corners.append(np.array([x1,y1]) + alpha*np.array([x2-x1,y2-y1]))
                edges
            else:
                corners.append(np.array([x1,y1])) 
                corners.append(np.array([x4,y4]))
                
        return np.array(corners)

    def cellPlot(self, *Val, size = None, bounds = None, radius = 'individual', cmap = 'cool'):
        if self.dim == 2:
            if size == None:
                size = 1000/len(self.pos)
            
            if radius == 'individual':
                r = self.r
            if radius == 'mean':
                r = self.r.mean()*np.ones(self.nofCells)

            #### polygon construction ####
            polygons = []

            cells = [Point(self.pos[i,:]).buffer(r[i]) for i in range(self.nofCells)]
            if self.nofCells == 1:
                polygons.append(cells[0])
            else:
                self.dist = cdist(self.pos, self.pos)
                for i in range(self.nofCells):
                    indices = np.where((self.dist[i,:] < r[i] + r[:]) & (self.dist[i,:] != 0))
                    cell1 = cells[i]

                    d = self.dist[i,indices[0]]
                    r_neigh = r[indices] 
                    a = (r[i]**2 - r_neigh**2 + d**2)/(2*d)
                    d12 = self.pos[indices[0],:] - self.pos[i,:]
                    d12_orth = np.array([d12[:,1],-d12[:,0]]).T

                    rect1 = self.pos[i,:] + d12/d[:,None]*a[:,None] + d12_orth/d[:,None]*r[i]
                    rect2 = self.pos[i,:] - d12/d[:,None]*r[i] + d12_orth/d[:,None]*r[i]
                    rect3 = self.pos[i,:] - d12/d[:,None]*r[i] - d12_orth/d[:,None]*r[i]
                    rect4 = self.pos[i,:] + d12/d[:,None]*a[:,None] - d12_orth/d[:,None]*r[i]

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
                mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

                for i in range(self.nofCells):
                    x,y = polygons[i].exterior.xy
                    plt.fill(x,y, facecolor=mapper.to_rgba(float(Val[i])), edgecolor='k', linewidth=1, zorder=1)
            
            #### plot cell nuclei ####
            plt.scatter(self.pos[:,0],self.pos[:,1], color='k', s=size, zorder=2)
            plt.axis('equal')
            plt.axis('off')

        if self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            size = 30000/len(self.pos)*self.r

            c = np.full((self.nofCells,) + (4,), [0,1,1,1])
            c[self.pos[:,0] < 0, -1] = 0

            #pos = self.pos[(self.pos[:,0] < 0) | (self.pos[:,1] < 0) | (self.pos[:,2] < 0)]
            #size = size[(self.pos[:,0] < 0) | (self.pos[:,1] < 0) | (self.pos[:,2] < 0)]
            if Val == ():
                #ax.scatter(pos[:,0], pos[:,1], pos[:,2], c = 'k', s=size, depthshade=False)
                ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], c = c, s=size)

            else: 
                ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], c=Val, cmap = 'cool', s=size)
            
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
            plt.plot(range(1,maxdist+1), self.pcf_G, color='m', lw = lw, ls = ls)
            plt.plot(range(1,maxdist+1), self.pcf_N, color='c', lw = lw, ls = ls)
            plt.axhline(1, ls='dashed', color='k')
            plt.xlabel('Distance')
            plt.ylabel('$\\rho_u, \\rho_v$')

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
        self.Data.append([self.pos,self.r,self.N,self.G])
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
        df['x-Position'] = self.pos[:,0]
        df['y-Position'] = self.pos[:,1]
        if self.dim == 3:
            df['z-Position'] = self.pos[:,2]

        df['Radius'] = self.r
        df['NANOG'] = self.N
        df['GATA6'] = self.G
        df.to_csv(directory + 'Data.csv', index = False)
        
        # Save all parameters in .txt file
        with open(directory + 'Parameters.txt', 'w') as f:
            f.write(''.join(["%s = %s\n" % (k,v) for k,v in self.__dict__.items() if not hasattr(v, '__iter__')]))

    def saveAnim(self, directory = '', frames = None, fps = 60):

        fig = plt.figure()
        bmin = min(min(self.pos[:,0]),min(self.pos[:,1])) - 1.5*self.r_max
        bmax = max(max(self.pos[:,0]),max(self.pos[:,1])) + 1.5*self.r_max


        def update(i):
            plt.cla()

            org = Organoid()
            org.dim = self.dim
            org.nofCells = len(self.Data[i][0])
            org.pos = self.Data[i][0]
            org.r = self.Data[i][1]
            org.N = self.Data[i][2]
            #org.dist = cdist(org.pos, org.pos)

            org.cellPlot(size=1000/self.nofCells, bounds=[min(self.N),max(self.N)])
            plt.xlim(bmin, bmax)
            plt.ylim(bmin, bmax)
            if self.dim == 3:
                plt.zlim(bmin, bmax)
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
        bmin = min(min(self.pos[:,0]),min(self.pos[:,1])) - 1.5*self.r_max
        bmax = max(max(self.pos[:,0]),max(self.pos[:,1])) + 1.5*self.r_max


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
            if self.dim == 3:
                plt.zlim(bmin, bmax)
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

    def evolution(self, T = None, file = None, dim = 2, ignore = [], maxCells = 100000):
        self.maxCells = maxCells

        if T != None:
            self.T = T
  
        self.dt = self.T/(self.nofSteps - 1)

        self.ignore = ignore

        if not hasattr(self, 'pos'):
            self.initialConditions(dim, file = file)
            self.dt = self.T/(self.nofSteps - 1)

        # Full model with possibility to switch of different parts
        self.cellDivisionDistance()
        if 'displacement' in self.ignore and 'division' in self.ignore:
            if (self.dim == 2 and self.nofCells >= 3) or (self.dim == 3 and self.nofCells >= 4):
                self.graphdistance()
                self.communication()
        for i in range(self.nofSteps):
            if self.nofCells == self.maxCells:
                break
            self.t += self.dt
            self.radiusGrowth()
            self.cellDivision()
            if self.nofCells >= 2:
                self.displacement()
            if (self.dim == 2 and self.nofCells >= 3) or (self.dim == 3 and self.nofCells >= 4):
                if 'displacement' not in self.ignore or 'division' not in self.ignore:
                    self.graphdistance()
                    self.communication()
                
                self.transcription()

            self.collectData()


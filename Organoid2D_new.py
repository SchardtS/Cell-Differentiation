import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import matplotlib as mpl
import matplotlib.cm as cm
from Parameters_new import Parameters
from matplotlib.animation import FuncAnimation

class Organoid(Parameters):
    def __init__(self):
        Parameters.__init__(self)
        
    def initialConditions(self, file = None):
        if file == None:
            self.xy = np.array([[-0.5,-0.5], [0.5,-0.5], [0,0.5]])
            self.r = np.array([0.8, 0.9, 0.75])
            
            
        else:
            Data = pd.read_csv(file)
            self.xy = Data[['x-Position','y-Position']].to_numpy()
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
        F = self.F0*2*self.alpha*(np.exp(-2*self.alpha*(self.dist-r_pairwise*self.sigma)) - np.exp(-self.alpha*(self.dist-r_pairwise*self.sigma)))
        F[self.dist > r_pairwise] = 0

        # Fill distance matrix with inf on diagonal
        dist = self.dist*1
        np.fill_diagonal(dist, np.inf)

        # x- and y-direction of forces
        Fx = F*(x_pairwise)/dist
        Fy = F*(y_pairwise)/dist

        # Sum of all forces acting on each cell as a vector
        Force = np.array([np.sum(Fx, axis=1), np.sum(Fy, axis=1)]).T
        
        self.xy = self.xy - self.dt*Force
        
    def graphdistance(self):
        #Gr = nx.Graph()
        #tri = Delaunay(self.xy)
                               
        #simplex_distance1 = self.dist[tri.simplices[:,0], tri.simplices[:,1]]
        #simplex_distance2 = self.dist[tri.simplices[:,1], tri.simplices[:,2]]
        #simplex_distance3 = self.dist[tri.simplices[:,0], tri.simplices[:,2]]

        #simplex_radii1 = self.r[tri.simplices[:,0]]+self.r[tri.simplices[:,1]]
        #simplex_radii2 = self.r[tri.simplices[:,1]]+self.r[tri.simplices[:,2]]
        #simplex_radii3 = self.r[tri.simplices[:,0]]+self.r[tri.simplices[:,2]]

        #simplices = tri.simplices[(simplex_distance1 < simplex_radii1*1.1) & 
        #                          (simplex_distance2 < simplex_radii2*1.1) &
        #                          (simplex_distance3 < simplex_radii3*1.1)]

        #for path in simplices:
        #    nx.add_path(Gr, path)

        #dist_dict = dict(nx.all_pairs_dijkstra_path_length(Gr))
        #self.GraphDist = np.empty([self.nofCells, self.nofCells])
        #for i in range(self.nofCells):
        #    for j in range(self.nofCells):
        #        self.GraphDist[i,j] = dist_dict[i][j]
        self.GraphDist = np.floor(self.dist/np.mean(2*self.r))
  
    def transcription(self):
        rhs = np.empty(self.nofCells*2)

        a = np.exp(-self.eps_N)
        b = np.exp(-self.eps_G)
        c = np.exp(-self.eps_S)
        d = np.exp(-self.eps_NS)
                               
        scaling = self.q**(self.GraphDist-1)
        np.fill_diagonal(scaling, 0)
        val = self.G*scaling#*(1-self.q)/self.q
        np.fill_diagonal(val, 0)
        self.S = np.sum(val, axis=1)/max(scaling.sum(1))

        pN = (a*self.N)*(1+d*c*self.S)/(1 + a*self.N*(1+d*c*self.S) + b*self.G + c*self.S)
        pG =        (b*self.G)        /(1 + a*self.N*(1+d*c*self.S) + b*self.G + c*self.S)

        rhs[:self.nofCells] = self.r_N*pN - self.gamma_N*self.N
        rhs[self.nofCells:] = self.r_G*pG - self.gamma_G*self.G
                               
        self.u = self.u + self.dt*rhs
        self.N = self.u[:self.nofCells]
        self.G = self.u[self.nofCells:]
                                      
    def cellPlot(self, *Val, size = None, bounds = None):        
        if size == None:
            size = 1000/len(self.xy)
        
        #### polygon construction ####
        polygons = []
        cells = [Point(self.xy[i,:]).buffer(self.r[i]) for i in range(self.nofCells)]
        for i in range(self.nofCells):
            indices = np.where((self.dist[i,:] < self.r[i] + self.r[:]) & (self.dist[i,:] != 0))
            cell1 = cells[i]

            d = self.dist[i,indices[0]]
            r_neigh = self.r[indices] 
            a = (self.r[i]**2 - r_neigh**2 + d**2)/(2*d)
            d12 = self.xy[indices[0],:] - self.xy[i,:]
            d12_orth = np.array([d12[:,1],-d12[:,0]]).T

            rect1 = self.xy[i,:] + d12/d[:,None]*a[:,None] + d12_orth/d[:,None]*self.r[i]
            rect2 = self.xy[i,:] - d12/d[:,None]*self.r[i] + d12_orth/d[:,None]*self.r[i]
            rect3 = self.xy[i,:] - d12/d[:,None]*self.r[i] - d12_orth/d[:,None]*self.r[i]
            rect4 = self.xy[i,:] + d12/d[:,None]*a[:,None] - d12_orth/d[:,None]*self.r[i]

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

    def collectData(self):
        self.Data.append([self.xy,self.r,self.N,self.G])
        return

    def saveData(self, directory = ''):
        # Save all parameters in .txt file
        with open(directory + 'Parameters.txt', 'w') as f:
            f.write(''.join(["%s = %s\n" % (k,v) for k,v in self.__dict__.items() if not hasattr(v, '__iter__')]))

        # Save plot of geometry
        plt.figure()
        self.cellPlot()
        plt.savefig(directory + 'tissue.png', transparent = True) 
        plt.savefig(directory + 'tissue.pdf', transparent = True)   

        # Save plot of NANOG
        plt.figure()
        self.cellPlot(self.N)
        plt.savefig(directory + 'NANOG.png', transparent = True) 
        plt.savefig(directory + 'NANOG.pdf', transparent = True)   

        # Save plot of GATA6
        plt.figure()
        self.cellPlot(self.G)
        plt.savefig(directory + 'GATA6.png', transparent = True) 
        plt.savefig(directory + 'GATA6.pdf', transparent = True)

        df = pd.DataFrame()
        df['x-Position'] = self.xy[:,0]
        df['y-Position'] = self.xy[:,1]
        df['Radius'] = self.r
        df['NANOG'] = self.N
        df['GATA6'] = self.G
        df.to_csv(directory + 'Data.csv', index = False)

    def saveAnim(self, directory = ''):
        fig, ax = plt.subplots()

        def update(i):
            org = Organoid()
            org.nofCells = len(self.Data[i][0])
            org.xy = self.Data[i][0]
            org.r = self.Data[i][1]
            org.N = self.Data[i][2]
            org.dist = cdist(org.xy, org.xy)
            plt.cla()
            org.cellPlot(org.N, size=1000/self.nofCells, bounds=[min(org.N),max(org.N)])
            bmin = min(min(self.xy[:,0]),min(self.xy[:,1])) - self.r_max
            bmax = max(max(self.xy[:,0]),max(self.xy[:,1])) + self.r_max
            plt.xlim(bmin, bmax)
            plt.ylim(bmin, bmax)
            return

        ani = FuncAnimation(fig, update, frames=len(self.Data), interval=1, blit=False)
        ani.save(directory + '/NANOG.mp4', fps=70, dpi=400)

        return

    def evolution(self, T = 0, file = None, mode = 'transcription + geometry'):
        if T == 0:
            T = self.T       
        N = int(T/self.dt)

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
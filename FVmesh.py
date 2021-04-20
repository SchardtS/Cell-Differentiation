import numpy as np
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from shapely.geometry import Polygon, Point
from numpy.linalg import solve, norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import networkx as nx

class FVmesh:
    '''
    Attributes: - self.Vol          Volumes of voronoi cells
                - self.Edge         Edge lengths of neighboring voronoi cells i and j
                - self.Dist         Distance of points i and j
                - self.Tri          Delaunay triangulation
                - self.Vor          Voronoi tesselation
                - self.Neigh        Neighboring cells according to Delaunay
                - self.D_mean       Mean distance between neighboring cells
                - self.Hull         Convex hull
                - self.Poly         Polygons given by voronoi cells cut off at self.D_mean distance
                                    away from convex hull
                - self.nofCells     Number of cells
                - self.Pos          Position of cells
    '''
    def __init__(self):
        self.nofCells = []
        self.Pos = []
        self.Poly = []
        self.Vol = []
        self.Edge = []
        self.Dist = []
        self.D_mean = []
        self.Vor = []
        self.Tri = []
        self.Neigh = []
        self.Hull = []
        self.GraphDist = []

    def distances(self):             
        self.Dist = cdist(self.Pos, self.Pos)
        
    def mean_distance(self):
        d_mean = 0
        for i in range(self.nofCells):
            d_mean += sum([self.Dist[i,n]/len(self.Neigh[i])/self.nofCells for n in self.Neigh[i]])
            
        self.D_mean = d_mean

    def remove_edges(self):
        simplex_distance1 = self.Dist[self.Tri.simplices[:,0],self.Tri.simplices[:,1]]
        simplex_distance2 = self.Dist[self.Tri.simplices[:,1],self.Tri.simplices[:,2]]
        simplex_distance3 = self.Dist[self.Tri.simplices[:,0],self.Tri.simplices[:,2]]

        simplex_radii1 = self.Radius[self.Tri.simplices[:,0]]+self.Radius[self.Tri.simplices[:,1]]
        simplex_radii2 = self.Radius[self.Tri.simplices[:,1]]+self.Radius[self.Tri.simplices[:,2]]
        simplex_radii3 = self.Radius[self.Tri.simplices[:,0]]+self.Radius[self.Tri.simplices[:,2]]

        self.Tri.simplices = self.Tri.simplices[(simplex_distance1 < simplex_radii1) & 
                                                (simplex_distance2 < simplex_radii2) &
                                                (simplex_distance3 < simplex_radii3)]
                                                
    def graph_distance(self):
        G = nx.Graph()
        for path in self.Tri.simplices:
            nx.add_path(G, path)
            
        dist_dict = dict(nx.all_pairs_dijkstra_path_length(G))
        self.GraphDist = np.empty([self.nofCells, self.nofCells])
        for i in range(self.nofCells):
            for j in range(self.nofCells):
                self.GraphDist[i,j] = dist_dict[i][j]

    def polygons(self):
        xmin, xmax = min(self.Pos[:,0]), max(self.Pos[:,0])
        ymin, ymax = min(self.Pos[:,1]), max(self.Pos[:,1])
        bbox = 10*np.array([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
        vor = Voronoi(np.append(self.Pos, bbox, axis=0))
        #bounding_poly = Polygon(self.Hull.points[self.Hull.vertices]).buffer(self.D_mean/2)
        if type(self.TE) != type(None):
            TE_poly = Polygon(self.TE)
        
        polylist = []
        for i, reg_num in enumerate(vor.point_region):
            indices = vor.regions[reg_num]
            if -1 not in indices:
                if type(self.Radius) != type(None):
                    radius = self.Radius[i]
                else:
                    radius = self.D_mean/2*10/9

                poly = Polygon(vor.vertices[indices])
                bounding_poly = Point(self.Pos[i,0], self.Pos[i,1]).buffer(radius)#self.D_mean/2)
                poly = bounding_poly.intersection(poly)
                
                if type(self.TE) != type(None):
                    poly = TE_poly.intersection(poly)

                polylist.append(poly)
        
        self.Poly = polylist
        
    def polygons_new(self):
        polylist = []
        for i in range(self.nofCells):
            cell1 = Point(self.Pos[i,:]).buffer(self.Radius[i])
            for j in self.Neigh[i]:
                cell2 = Point(self.Pos[j,:]).buffer(self.Radius[j])
                poly = cell1.intersection(cell2)

                d = np.linalg.norm(self.Pos[i,:]-self.Pos[j,:])
                a = (self.Radius[i]**2 - self.Radius[j]**2 + d**2)/(2*d)
                d12 = self.Pos[j,:] - self.Pos[i,:]
                d12_orth = np.array([d12[1],-d12[0]])
                sq1 = self.Pos[i,:] + d12/d*a + d12_orth/d*self.Radius[i]
                sq2 = self.Pos[i,:] - d12/d*self.Radius[i] + d12_orth/d*self.Radius[i]
                sq3 = self.Pos[i,:] - d12/d*self.Radius[i] - d12_orth/d*self.Radius[i]
                sq4 = self.Pos[i,:] + d12/d*a - d12_orth/d*self.Radius[i]

                square = np.array([sq1,sq2,sq3,sq4])
                square = Polygon(square)

                cell1 = cell1.intersection(square)

            polylist.append(cell1)

        self.Poly = polylist
        
    def volumes(self):
        vol = np.empty(self.nofCells)
        for i in range(self.nofCells):
            vol[i] = self.Poly[i].area
            
        self.Vol = vol
        
    def neighbors(self):
        neigh = [0]*self.nofCells
        for i in range(self.nofCells):
            all_neigh = self.Tri.vertex_neighbor_vertices[1][self.Tri.vertex_neighbor_vertices[0][i]:self.Tri.vertex_neighbor_vertices[0][i+1]]
            neigh[i] = [j for j in all_neigh if self.Dist[i,j] < self.Radius[i] + self.Radius[j]]

        self.Neigh = neigh
        
    def edges(self):
        length = np.empty([self.nofCells, self.nofCells])
        for i in range(self.nofCells):
            for j in range(self.nofCells):  
                if i not in self.Neigh[j]:
                    length[i,j] = 0
                else:
                    edge = self.Poly[i].intersection(self.Poly[j])
                    length[i,j] = edge.length

        self.Edge = length
        
    def plot(self, *Val, size = None, bounds = None):
        if size == None:
            size = 1000/self.nofCells

        if Val == ():
            for i in range(self.nofCells):
                if self.Poly[i].is_empty:
                    continue
                x,y = self.Poly[i].exterior.xy
                plt.plot(x,y, 'k', alpha = 1) #0.8

            plt.scatter(self.Pos[:,0],self.Pos[:,1], color = 'k', alpha=1, s=size)
            if type(self.TE) != type(None):
                plt.plot(self.TE[:,0], self.TE[:,1], color = 'r', lw = 2, alpha=0.8)
        
        else:
            Val = Val[0]

            if bounds == None:
                bounds = [min(Val), max(Val)]
            
            norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1], clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='cool')

            for i in range(self.nofCells):
                if self.Poly[i].is_empty:
                    continue
                x,y = self.Poly[i].exterior.xy
                plt.fill(x,y, facecolor=mapper.to_rgba(float(Val[i])), edgecolor='k', alpha = 1, linewidth=1)

            plt.scatter(self.Pos[:,0],self.Pos[:,1], color = 'k', alpha=1, s=size)
            if type(self.TE) != type(None):
                plt.plot(self.TE[:,0], self.TE[:,1], color = 'r', lw = 2, alpha=1)

        #plt.axes().set_aspect('equal')
        plt.axis('off')
        #plt.show()

    def radius_distance(self):
        self.GraphDist = np.floor(self.Dist/np.mean(self.Radius))
        
    
def initializeFVmesh(pos, Radius=None, TE=None, reduced = False):

    if reduced == False:
        self = FVmesh()
        self.__init__()
        self.TE = TE
        self.Radius = Radius
        self.Pos = pos
        self.nofCells = len(self.Pos)
        self.Tri = Delaunay(self.Pos)
        self.Vor = Voronoi(self.Pos)
        #self.Hull = ConvexHull(self.Pos)
        
        self.distances()
        #self.remove_edges()
        #self.neighbors()
        #self.mean_distance()
        self.graph_distance()
        #self.polygons()
        #self.polygons_new()
        #self.volumes()
        #self.edges()
        
    elif reduced == True:
        self = FVmesh()
        self.__init__()
        self.Pos = pos
        self.Radius = Radius
        self.nofCells = len(self.Pos)
        self.distances()
        self.radius_distance()

        #self.Tri = Delaunay(self.Pos)
        #self.neighbors()
        #self.distances()

    else:
        print('Error: reduced =', str(reduced),'not supported!')

    return self   
            
    
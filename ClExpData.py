import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
from scipy.spatial import Delaunay
import itertools
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class ExpData():
    def __init__(self, file):
        Data = pd.read_csv(file)
        self.id = Data['OrganoidID'].to_numpy()
        self.pos = Data[['CentroidX', 'CentroidY', 'CentroidZ']].to_numpy()
        self.stage = Data['stage'].to_numpy()
        self.pop = Data['Fate_KM_Cluster'].to_numpy()
        self.N = Data['Nanog-Avg'].to_numpy()
        self.G = Data['Gata6-Avg'].to_numpy()
        self.pcf = {}
        self.pcf_mean = {}
        self.moran = {}
        self.GraphDist = {}

    def info(self, ID):
        print('Organoid', ID, 'is', self.stage[self.id == ID][0], 'old')
        print('Organoid', ID, 'consists of', len(self.id[self.id == ID]), 'cells')
        print('Organoid', ID, 'consists of', len(self.id[(self.id == ID) & (self.pop == 'N+G-')]), 'NANOG cells')
        print('Organoid', ID, 'consists of', len(self.id[(self.id == ID) & (self.pop == 'N-G+')]), 'GATA6 cells')

    def pcf_bounds(self, ID):
        if ID in self.pcf:
            PN = self.pcf[ID][0,:]
            PG = self.pcf[ID][1,:]

        else:
            self.graphdistance(ID)
            GraphDist = self.GraphDist[ID]
            pop = self.pop[self.id == ID]

            maxdist = int(np.max(GraphDist))
            N = np.zeros([len(pop)])   
            G = np.zeros([len(pop)])
            N[pop == 1] = 1
            G[pop == 0] = 1
            
            cells = N.shape[0]
            cells_N = np.sum(N)
            cells_G = np.sum(G)
            rho_N = cells_N*(cells_N - 1)/(cells*(cells - 1))
            rho_G = cells_G*(cells_G - 1)/(cells*(cells - 1))

            PN = np.zeros([maxdist])
            PG = np.zeros([maxdist])

            ind_N = np.where(N==1)[0]
            pairs_N = GraphDist[ind_N].T[ind_N].T
            
            ind_G = np.where(G==1)[0]
            pairs_G = GraphDist[ind_G].T[ind_G].T

            for i in range(1,maxdist+1):
                PN[i-1] = len(pairs_N[pairs_N==i])/len(GraphDist[GraphDist==i])/rho_N
                PG[i-1] = len(pairs_G[pairs_G==i])/len(GraphDist[GraphDist==i])/rho_G

            self.pcf[ID] = np.stack((PN, PG))

    def graphdistance(self, ID):
        if not ID in self.GraphDist:   
            pos = self.pos[self.id == ID]

            Gr = nx.Graph()
            Dist = cdist(pos, pos)
            tri = Delaunay(pos)
            n = len(pos)

            all_dist = []
            for i in range(len(pos)):
                neigh = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
                for j in neigh:
                    all_dist.append(Dist[i,j])

            self.cutoff = np.mean(all_dist) + 2*np.std(all_dist)

            for simp in tri.simplices:
                for path in list(itertools.combinations(simp, 2)):
                    if Dist[path[0],path[1]] < self.cutoff:
                        nx.add_path(Gr, path)

            Gdist_dict = dict(nx.all_pairs_dijkstra_path_length(Gr))
            Gdist = np.empty([n, n])
            for i in range(n):
                for j in range(n):
                    Gdist[i,j] = Gdist_dict[i][j]

            self.GraphDist[ID] = Gdist
            
    def moran_bounds(self, ID, sample_size, portion_N = 1):
        if not(ID in self.moran and self.sample_size == sample_size):
            self.sample_size = sample_size
            self.graphdistance(ID)
            GraphDist = self.GraphDist[ID]
            pop = self.pop[self.id == ID]

            N = np.zeros([len(pop), self.sample_size])
            N[(pop == 'N+G-')] = 1
            N[(pop == 'N+G+') | (pop == 'N-G-')] = np.random.random(N[(pop == 'N+G+') | (pop == 'N-G-')].shape)    
            
            if portion_N == 1:
                portion_N = len(N[(pop == 'N-G+')])/len(N[(pop == 'N-G+') | (pop == 'N+G-')])
                
            N[N > portion_N] = 1
            N[N <= portion_N] = 0
            
            I = np.empty(self.sample_size)
            for j in range(self.sample_size):
                W = np.copy(GraphDist)
                W[W > 1] = 0
                y = N[:,j] - N[:,j].mean()

                numerator = np.dot(y, np.dot(W, y))
                denominator = np.sum(y**2)

                I[j] = len(y)/np.sum(W)*numerator/denominator

            self.moran[ID] = [np.min(I), np.mean(I), np.max(I)]

    def cellPlot(self, ID):
        fig = plt.figure()
        xyz = self.pos[self.id == ID]
        pop = self.pop[self.id == ID]
        ax = fig.add_subplot(1,2,1,projection='3d')
        size = 10000/len(xyz)

        i_N = np.where(pop == 'N+G-')
        i_G = np.where(pop == 'N-G+')
        i_D = np.where((pop == 'N+G+') | (pop == 'N-G-'))

        ax.scatter(xyz[i_N,0], xyz[i_N,1], xyz[i_N,2], color = 'm', s=size)
        ax.scatter(xyz[i_G,0], xyz[i_G,1], xyz[i_G,2], color = 'c', s=size)
        ax.scatter(xyz[i_D,0], xyz[i_D,1], xyz[i_D,2], color = 'k', s=size)
        ax.axis('off')

    def fullPlot(self, ID, sample_size):

        self.pcf_bounds(ID, sample_size=sample_size, plot = False)
        #plt.rcParams["font.family"] = "Comic Sans MS"
        fig = plt.figure(figsize=(15,8))
        fig.suptitle('Organoid ID = ' + str(self.id[self.id == ID][0]) + 
                     ', Number of cells = ' + str(len(self.id[self.id == ID])),
                     fontweight='bold', fontsize = 20)
        plt.rc('font', size=14)
        ax1 = fig.add_subplot(2,3,1)
        ranges = list(range(1,len(self.pcf[ID][0])+1))
        ax1.fill_between(ranges, self.pcf[ID][0], self.pcf[ID][2], color='m', alpha=0.2, label='NANOG')
        ax1.fill_between(ranges, self.pcf[ID][3], self.pcf[ID][5], color='c', alpha=0.2, label='GATA6')
        ax1.plot(ranges, self.pcf[ID][1], 'm', lw=2)
        ax1.plot(ranges, self.pcf[ID][4], 'c', lw=2)
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('$\\rho$')
        ax1.legend()

        ax2 = fig.add_subplot(2,3,4)
        pop = self.pop[self.id == ID]
        nofD = len(pop[(pop == 'N-G-') | (pop == 'N+G+')])/len(pop)*100
        nofN = len(pop[pop == 'N+G-'])/len(pop)*100
        nofG = len(pop[pop == 'N-G+'])/len(pop)*100
        ax2.bar(['N+G+ & \n N-G-', 'N+G-', 'N-G+'], [nofD, nofN, nofG], color=['gray', 'm', 'c'], edgecolor='k')
        for i, v in enumerate([nofD, nofN, nofG]):
            ax2.text(i, v+5, str(int(np.round(v)))+'%', color='black', fontweight='bold', ha='center')
        ax2.set_ylim([0,100])
        ax2.set_ylabel('Proportions')
        ticks = [0,20,40,60,80,100]
        ax2.set_yticks(ticks, [str(x)+'%' for x in ticks])

        xyz = self.pos[self.id == ID]
        pop = self.pop[self.id == ID]
        ax = fig.add_subplot(2,3,(2,6),projection='3d')
        size = 20000/len(xyz)

        i_N = np.where(pop == 'N+G-')
        i_G = np.where(pop == 'N-G+')
        i_D = np.where((pop == 'N+G+') | (pop == 'N-G-'))

        ax.scatter(xyz[i_N,0], xyz[i_N,1], xyz[i_N,2], color = 'm', s=size)
        ax.scatter(xyz[i_G,0], xyz[i_G,1], xyz[i_G,2], color = 'c', s=size)
        ax.scatter(xyz[i_D,0], xyz[i_D,1], xyz[i_D,2], color = 'k', s=size)
        ax.axis('off')

    def fullPlot_HTML(self, ID, sample_size, file=None):

        self.pcf_bounds(ID, sample_size, plot = False)
        ranges = list(range(1,len(self.pcf[ID][0])+1))
        pop = self.pop[self.id == ID]
        nofD = len(pop[(pop == 'N-G-') | (pop == 'N+G+')])/len(pop)*100
        nofN = len(pop[pop == 'N+G-'])/len(pop)*100
        nofG = len(pop[pop == 'N-G+'])/len(pop)*100

        xyz = self.pos[self.id == ID]
        pop = self.pop[self.id == ID]

        fig = make_subplots(rows=2, cols=3,specs=[[{}, {"rowspan": 2, "colspan": 2, "type": "scatter3d"}, None],
                                          [{}, None, None]])

        info1 = '<b>PCF minimum (N+G-)</b><br>dist = %{x}<br>PCF = %{y}<extra></extra>'
        fig.add_trace(
            go.Scatter(x=ranges, y=self.pcf[ID][0], fill=None, mode='lines',
            hovertemplate=info1, line_color='rgba(0.75,0,0.75,1)'),
            row=1, col=1
        )
        info2 = '<b>PCF maximum (N+G-)</b><br>dist = %{x}<br>PCF = %{y}<extra></extra>'
        fig.add_trace(
            go.Scatter(x=ranges, y=self.pcf[ID][2], fill='tonexty',mode='lines',
            hovertemplate=info2, line_color='rgba(0.75,0,0.75,1)'),
            row=1, col=1 
        )
        info3 = '<b>PCF minimum (N-G+)</b><br>dist = %{x}<br>PCF = %{y}<extra></extra>'
        fig.add_trace(
            go.Scatter(x=ranges, y=self.pcf[ID][3], fill=None, mode='lines',
            hovertemplate=info3, line_color='rgba(0,0.75,0.75,1)'),
            row=1, col=1
        )
        info4 = '<b>PCF maximum (N-G+)</b><br>dist = %{x}<br>PCF = %{y}<extra></extra>'
        fig.add_trace(
            go.Scatter(x=ranges, y=self.pcf[ID][5], fill='tonexty',mode='lines',
            hovertemplate=info4, line_color='rgba(0,0.75,0.75,1)'),
            row=1, col=1 
        )

        infobar = '<b>%{y}%</b><extra></extra>'
        fig.add_trace(go.Bar(x=['DN & DP'], y = [nofD], name='DN & DP', marker_color='rgba(0.75,0.75,0.75,1)', hovertemplate=infobar), row=2, col=1)
        fig.add_trace(go.Bar(x=['N+G-'], y = [nofN], name='N+G-', marker_color='rgba(0.75,0,0.75,1)', hovertemplate=infobar), row=2, col=1)
        fig.add_trace(go.Bar(x=['N-G+'], y = [nofG], name='N-G+', marker_color='rgba(0,0.75,0.75,1)', hovertemplate=infobar), row=2, col=1)

        colors3d = np.array(['rgba(0.75,0.75,0.75,1)' for _ in range(len(xyz))])
        colors3d[pop == 'N+G-'] = 'rgba(0.75,0,0.75,1)'
        colors3d[pop == 'N-G+'] = 'rgba(0,0.75,0.75,1)'
        text3d = np.array(['<b>DN or DP</b>' for _ in range(len(xyz))])
        text3d[pop == 'N+G-'] = '<b>N+G-</b>'
        text3d[pop == 'N-G+'] = '<b>N-G+</b>'
        info3d = '%{hovertext}<br>x = %{x}<br>y = %{y}<br>z = %{z}<extra></extra>'
        fig.add_trace(
            go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2], mode='markers',
            hovertext=text3d,
            hovertemplate=info3d,
            marker=dict(
            size=50/len(xyz)**(1/3),
            color=colors3d,
            line=dict(
                color='black',
                width=1
            )
            )),
            row=1, col=2
        )

        fig.update_layout(height=600, width=900, title_text="Organoid ID = "+str(ID)+", Number of cells = "+str(len(xyz)), paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False)
        fig.update_layout({'xaxis':{'title_text':'distances','linecolor':'black'},
                        'yaxis':{'title_text':'PCF','linecolor':'black'},
                        'xaxis2':{'title_text':'cell type','linecolor':'black'},
                        'yaxis2':{'range': [0, 100],'dtick': 20, 'title_text':'proportions','linecolor':'black'},}
        )

        fig.update_layout(scene = dict(xaxis = dict(showbackground=False,showticklabels=False,visible=False,range=[np.min(xyz),np.max(xyz)]),
                                    yaxis = dict(showbackground=False,showticklabels=False,visible=False,range=[np.min(xyz),np.max(xyz)]),
                                    zaxis = dict(showbackground=False,showticklabels=False,visible=False,range=[np.min(xyz),np.max(xyz)])))
        
        if file == None:
            fig.show()
        else:
            fig.write_html(file)

    def sliderPlot_HTML(self, file=None):
        fig = make_subplots(rows=2, cols=3,specs=[[{}, {"rowspan": 2, "colspan": 2, "type": "scatter3d"}, None],
                                [{}, None, None]])

        info1 = '<b>PCF (N+G-)</b><br>dist = %{x}<br>PCF = %{y}<extra></extra>'
        info2 = '<b>PCF (N-G+)</b><br>dist = %{x}<br>PCF = %{y}<extra></extra>'
        infobar = '<b>%{y}%</b><extra></extra>'
        info3d = '%{hovertext}<br>x = %{x}<br>y = %{y}<br>z = %{z}<extra></extra>'

        steps = []

        IDs = np.unique(self.id)#[self.stage == '48h'])
        for i, ID in enumerate(IDs):
            print(i+1, 'of', len(IDs))
            if i == 0:
                visibility = True
            else:
                visibility = False

            self.pcf_bounds(ID)
            ranges = list(range(1,len(self.pcf[ID][0])+1))
            pop = self.pop[self.id == ID]
            nofN = len(pop[pop == 1])/len(pop)*100
            nofG = len(pop[pop == 0])/len(pop)*100

            xyz = self.pos[self.id == ID]
            pop = self.pop[self.id == ID]

            fig.add_trace(
                go.Scatter(x=ranges, y=self.pcf[ID][0], fill=None, mode='lines',
                hovertemplate=info1, line_color='rgba(0.75,0,0.75,1)', visible=visibility),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=ranges, y=self.pcf[ID][1], fill=None,mode='lines',
                hovertemplate=info2, line_color='rgba(0,0.75,0.75,1)', visible=visibility),
                row=1, col=1 
            )

            fig.add_trace(go.Bar(x=['N+G-'], y =[nofN], name='N+G-', marker_color='rgba(0.75,0,0.75,1)', hovertemplate=infobar, visible=visibility), row=2, col=1)
            fig.add_trace(go.Bar(x=['N-G+'], y =[nofG], name='N-G+', marker_color='rgba(0,0.75,0.75,1)', hovertemplate=infobar, visible=visibility), row=2, col=1)

            colors3d = np.array(['rgba(0.75,0.75,0.75,1)' for _ in range(len(xyz))])
            colors3d[pop == 1] = 'rgba(0.75,0,0.75,1)'
            colors3d[pop == 0] = 'rgba(0,0.75,0.75,1)'
            text3d = np.array(['<b>DN or DP</b>' for _ in range(len(xyz))])
            text3d[pop == 1] = '<b>N+G-</b>'
            text3d[pop == 0] = '<b>N-G+</b>'
            fig.add_trace(
                go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2], mode='markers',
                hovertext=text3d,
                hovertemplate=info3d,
                visible=visibility,
                marker=dict(
                size=50/len(xyz)**(1/3),
                color=colors3d,
                line=dict(
                    color='black',
                    width=1
                )
                )),
                row=1, col=2
            )

            step = dict(method = 'update',
                        args = [{'visible': [False]*len(IDs)*8}, 
                                {'title': "Organoid ID = "+str(i+1)+", Number of cells = "+str(len(xyz))+", Stage = "+str(self.stage[self.id == i+1][0])}]
                                , label=str(i+1),)
            for j in range(5):
                step['args'][0]['visible'][i*5+j] = True
            steps.append(step)

        fig.update_layout(height=600, width=900, title="Organoid ID = "+str(1)+", Number of cells = "+str(len(self.id[self.id == IDs[0]])), paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False)
        
        fig.update_layout({'xaxis':{'title_text':'distances','linecolor':'black'},
                        'yaxis':{'title_text':'PCF','linecolor':'black'},
                        'xaxis2':{'title_text':'cell type','linecolor':'black'},
                        'yaxis2':{'range': [0, 100],'dtick': 20, 'title_text':'proportions','linecolor':'black'},}
        )

        fig.update_layout(scene = dict(xaxis = dict(showbackground=False,showticklabels=False,visible=False,range=[np.min(xyz),np.max(xyz)]),
                                    yaxis = dict(showbackground=False,showticklabels=False,visible=False,range=[np.min(xyz),np.max(xyz)]),
                                    zaxis = dict(showbackground=False,showticklabels=False,visible=False,range=[np.min(xyz),np.max(xyz)])))

        sliders = [{'steps': steps}]
        fig.layout.sliders = sliders

        if file == None:
            fig.show()
        else:
            fig.write_html(file)

    def propPlot(self, ID):
        #plt.rc('font', size=14)
        pop = self.pop[self.id == ID]
        nofD = len(pop[(pop == 'N-G-') | (pop == 'N+G+')])/len(pop)*100
        nofN = len(pop[pop == 'N+G-'])/len(pop)*100
        nofG = len(pop[pop == 'N-G+'])/len(pop)*100
        plt.bar(['DN & DP', 'N+G-', 'N-G+'], [nofD, nofN, nofG], color=['gray', 'm', 'c'], edgecolor='k')
        for i, v in enumerate([nofD, nofN, nofG]):
            plt.text(i, v+5, str(int(np.round(v)))+'%', color='black', fontweight='bold', ha='center')
        plt.ylim([0,100])
        plt.ylabel('Proportions')
        ticks = [0,20,40,60,80,100]
        plt.yticks(ticks, [str(x)+'%' for x in ticks])
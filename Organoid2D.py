#!/usr/bin/env python
# coding: utf-8

import random
from scipy.spatial import distance
import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.integrate import nquad
from FVmesh import initializeFVmesh
from Model import rhs_activation


#Hilfsfunktionen

def Distance(Pos):              
    dist = cdist(Pos, Pos)
    return dist

def Forces(Pos,r,F0,a,dist,sigma):
    rT = np.reshape(r,[len(r),1])
    x = Pos[:,0]
    y = Pos[:,1]
    xT = np.reshape(x,[len(x),1])
    yT = np.reshape(y,[len(y),1])
    
    r_pairwise = r+rT
    x_pairwise = x-xT
    y_pairwise = y-yT
    
    np.place(x_pairwise, dist/(r_pairwise) > sigma/2, 0)
    np.place(y_pairwise, dist/(r_pairwise) > sigma/2, 0)
    
    F = F0*2*(np.exp(-2*a*(dist-r_pairwise)) - np.exp(-a*(dist-r_pairwise)))
    #F[dist > 2] = 0

    np.fill_diagonal(dist, np.inf)
    Fx = F*(x_pairwise)/dist
    Fy = F*(y_pairwise)/dist
    
    return Fx, Fy

def DivisionProbability(r):
    c = 100
    y = 0.95
    rmin = 0.9
    rmax = 1
    b = (1 + np.exp(c*(rmax-y)))*(1 + np.exp(c*(rmin-y)))/(np.exp(c*(rmin-y)) - np.exp(c*(rmax-y)))
    a = -b/(1 + np.exp(c*(rmin-y)))
    P = a+b/(1 + np.exp(c*(r-y)))

    return np.maximum(0, P)

def ProbabilityDensity(r):
    c = 100
    y = 0.95
    rmin = 0.9
    rmax = 1
    b = (1 + np.exp(c*(rmax-y)))*(1 + np.exp(c*(rmin-y)))/(np.exp(c*(rmin-y)) - np.exp(c*(rmax-y)))
    fval = -b*c/(1 + np.exp(c*(r-y)))/(1 + np.exp(c*(y-r)))

    return fval

def ExpectedDivisionRadius():
    c = 100
    y = 0.95
    rmin = 0.9
    rmax = 1
    b = (1 + np.exp(c*(rmax-y)))*(1 + np.exp(c*(rmin-y)))/(np.exp(c*(rmin-y)) - np.exp(c*(rmax-y)))
    a = -b/(1 + np.exp(c*(rmin-y)))
    E = rmax - a*(rmax-rmin) + b/c*np.log((1+np.exp(-c*(rmax-y)))/(1+np.exp(-c*(rmin-y))))
    return E

""" def GrowthRate(nofCells_start, nofCells_end, endtime, rmax):
    T = endtime
    r_div = ExpectedDivisionRadius()
    k = -1/(np.log(2)*T)*np.log(nofCells_end/nofCells_start)*np.log((rmax - r_div)/(rmax - (1/2)**(1/3)*r_div))
    return k """

def GrowthRate(Prm):

    func = lambda r, r0: np.log((Prm.rmax - r)/(Prm.rmax - r0/2**(1/2))) \
        *ProbabilityDensity(r)*ProbabilityDensity(r0)
    I, err = nquad(func, [[0.9, 1],[0.9, 1]])

    k = - np.log(Prm.nofCells_end/Prm.nofCells_start)/(np.log(2)*Prm.T)*I

    func3 = lambda r, r0: -1/k*func(r,r0)
    T_DIV, err = nquad(func3, [[0.9, 1],[0.9, 1]])
    print('Expected cell division time =', T_DIV)

    func2 = lambda r, r0: 2**(-Prm.T*k/np.log((Prm.rmax - r)/(Prm.rmax - r0/2**(1/2)))) \
        *ProbabilityDensity(r)*(10*2**(1/2)-ProbabilityDensity(r0))*Prm.nofCells_start
    I2, err = nquad(func2, [[0.9, 1],[0.9, 1]])
    print('Expected number of Cells =', I2)
    return k

class Organoid:
  
    def __init__(self):
        self.IDs = []
        self.Pos = []
        self.Radius = []
        self.oldRadius = []
        self.initRadius = []
        self.Data = []
        self.t0 = []
        self.DivRad = []
        self.DivTime = []
        self.Dist = []
        self.NANOG = []
        self.GATA6 = []
        return
    
    def initializeCells(self,Prm):
        #Bereich festlegen
 
        Radius_Bereich = Prm.nofCells_start**(1/3)*Prm.rmax*0.75
        min_XY = Radius_Bereich/2*-1
        max_XY = Radius_Bereich/2

               
        #Erste Zelle auf (0,0)
        #Cell0 = [0,0]
        self.IDs.append(0)
        self.Pos = np.array([[0,0]])
        self.Radius = np.append(self.Radius, random.gauss((1/2)**(1/3)*ExpectedDivisionRadius(),0.1))
        self.t0 = np.append(self.t0, 0)
            
        #restliche Zellen Random im festgelegten Bereich verteilen mit einem Mindestabstand zueinander.
        for i in range(1,Prm.nofCells_start):
            trys = 0
            self.IDs.append(i)
            self.t0 = np.append(self.t0, 0)
            self.Radius = np.append(self.Radius, random.gauss((1/2)**(1/3)*ExpectedDivisionRadius(),0.1))  
            Distance_is_OK = False
            self.Pos = np.append(self.Pos, [[random.uniform(min_XY,max_XY),
                                            random.uniform(min_XY,max_XY)]], axis=0)
            while Distance_is_OK == False:                    
                safer = 1
                if trys >= 50 and trys < 150:
                    safer = 0.95
                elif trys >=150 and trys < 350:
                    safer = 0.9
                elif trys >= 350 and trys < 500:
                    safer = 0.7
                elif trys >= 500 and trys < 750:
                    safer = 0.6
                elif trys >= 750 and trys < 1000:
                    safer = 0.5
                elif trys >= 1000 and trys < 2000:
                    safer = 0.4
                elif trys >= 2000 and trys < 5000:
                    safer = 0.25
                elif trys >= 5000 and trys < 10000:
                    safer = 0.1
                elif trys >= 50000:
                    safer = 0.01
                Distance_i_MP = euclidean(self.Pos[i],[0,0])
                if Distance_i_MP > Radius_Bereich:
                    Distance_is_OK = False
                    self.Pos[i] = [random.uniform(min_XY,max_XY),
                                   random.uniform(min_XY,max_XY)]
                    continue
                for ii in self.IDs:                                     
                    if not ii == i:
                        Distance_i_ii = euclidean(self.Pos[i],self.Pos[ii])
                        if Distance_i_ii < safer*Prm.rmax:
                            Distance_is_OK = False
                            self.Pos[i] = [random.uniform(min_XY,max_XY),
                                           random.uniform(min_XY,max_XY)]
                            trys += 1
                            break 
                        elif Distance_i_ii >= safer*Prm.rmax:
                            Distance_is_OK = True 
                            
        self.oldRadius = self.Radius
        self.initRadius = self.Radius
        self.Dist = Distance(self.Pos)
        self.NANOG = np.array([0.5 + random.gauss(0,0.05) for i in self.IDs])
        self.GATA6 = np.array([0.5 + random.gauss(0,0.05) for i in self.IDs])
        return
                            
    def cellDivision(self,Prm):
        
        P0 = DivisionProbability(self.oldRadius/Prm.rmax)       
        P = DivisionProbability(self.Radius/Prm.rmax)
        IDs = self.IDs

        for i in range(len(IDs)):
            radius = self.Radius[i]
            if radius < 0.9:
                continue
                
            if self.oldRadius[i] == self.initRadius[i]:
                Prob = P[i]
            else:
                Prob = (P[i]-P0[i])/(1-P0[i])
           
            if random.random() < Prob:

                self.DivRad.append(self.Radius[i])
                self.DivTime.append(self.t - self.t0[i])

                NewID = self.IDs[-1]+1
                self.IDs.append(NewID)
                NewRadius = self.Radius[i]/2**(1/2)
                
                Distance_between_new_cells = self.Radius[i]-NewRadius
                Dbnc_low = (Distance_between_new_cells)/2.2
                Dbnc_high = (Distance_between_new_cells)/1.8
                
                rand = random.randint(0,1)
                if rand == 0:
                    RandomDistanceX = -random.uniform(Dbnc_low,Dbnc_high)
                elif rand == 1:
                    RandomDistanceX = random.uniform(Dbnc_low,Dbnc_high)
                rand = random.randint(0,1)
                if rand == 0:
                    RandomDistanceY = -random.uniform(Dbnc_low,Dbnc_high)
                elif rand == 1:
                    RandomDistanceY = random.uniform(Dbnc_low,Dbnc_high)

                RandomDistance = [RandomDistanceX,RandomDistanceY]
                NewPos1 = np.array(self.Pos[i])+np.array(RandomDistance)
                NewPos2 = np.array(self.Pos[i])-np.array(RandomDistance)
                self.Pos = np.append(self.Pos, [NewPos1], axis=0)
                self.Pos[i] = NewPos2
                
                self.Radius[i] = NewRadius
                self.Radius = np.append(self.Radius, NewRadius)
                self.initRadius[i] = NewRadius
                self.initRadius = np.append(self.initRadius, NewRadius)
                self.t0[i] = self.t
                self.t0 = np.append(self.t0, self.t)
                self.NANOG = np.append(self.NANOG, self.NANOG[i])
                self.GATA6 = np.append(self.GATA6, self.GATA6[i])

        return
                 
    def radiusGrowth(self,Prm):
        self.oldRadius = self.Radius
        self.Radius = Prm.rmax - np.exp(-self.k*(self.t-self.t0))*(Prm.rmax - self.initRadius)
        return
     
    def displacement(self,Prm):
        self.Dist = Distance(self.Pos)
        Fx, Fy = Forces(self.Pos,self.Radius,Prm.F0,Prm.alpha,self.Dist,Prm.sigma)
        Fx_sum = np.sum(Fx, axis=1)
        Fy_sum = np.sum(Fy, axis=1)
        
        self.Pos = self.Pos - self.dt*np.array([Fx_sum,Fy_sum]).T
        return

    def transcription(self, Prm):
        FVmesh = initializeFVmesh(self.Pos, reduced=True)
        x = np.append(self.NANOG, self.GATA6)
        rhs = rhs_activation(self.t, x, Prm, FVmesh)

        rhs_N = rhs[:len(self.IDs)]
        rhs_G = rhs[len(self.IDs):]

        self.NANOG = self.NANOG + self.dt*100*rhs_N
        self.GATA6 = self.GATA6 + self.dt*100*rhs_G

    def dataCollecting(self):     
        IDs = self.IDs[:]
        Pos = np.array(self.Pos[:])
        NANOG = np.array(self.NANOG)
        GATA6 = np.array(self.GATA6)
        Radius = self.Radius[:]
        self.Data.append([IDs,Pos,Radius,NANOG,GATA6])
        return

def initializeOrganoid(Prm):

    self = Organoid()
    self.__init__()
    self.t = 0
    self.k = GrowthRate(Prm)
    self.dt = Prm.dt
    self.initializeCells(Prm)
    self.dataCollecting()
    self.cellDivision(Prm)
    for step in range(1, Prm.nofSteps+1):
        self.t = step*Prm.dt
        self.radiusGrowth(Prm)
        self.cellDivision(Prm)
        self.displacement(Prm)
        self.transcription(Prm)
        self.dataCollecting()
    return self   
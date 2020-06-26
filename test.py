import numpy as np
import matplotlib.pyplot as plt
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Parameters import setParameters
import pandas as pd
from matplotlib.animation import FuncAnimation

Prm = setParameters()
#TEfunc = lambda theta: np.array([4*np.cos(theta),9*np.sin(theta)]).T
#TE = TEfunc(np.linspace(0,2*np.pi,100))
TE = 5*np.array([[1,-1],[1,1],[-1,1],[-1,-1]])
Organoid = initializeOrganoid(Prm, TE=None, Transcription=False)
FVmesh = initializeFVmesh(Organoid.Pos, TE=None)

FVmesh.plot()
plt.show()

""" fig, ax = plt.subplots()
def update(i):
    Pos = Organoid.Data[i][1]
    FVmesh = initializeFVmesh(Pos)
    plt.cla()
    FVmesh.plot(size=1000/len(Organoid.IDs), bounds=[min(Organoid.NANOG),max(Organoid.NANOG)])
    plt.xlim(min(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,0])*1.3)
    plt.ylim(min(Organoid.Pos[:,1])*1.3,max(Organoid.Pos[:,1])*1.3)
    return

ani2 = FuncAnimation(fig, update, frames=Prm.nofSteps, interval=1, blit=False)
ani2.save('Results/test.mp4', fps=70, dpi=200) """

#df = pd.DataFrame(FVmesh.Pos)
#df.to_csv('testOrganoid.csv', index=False)
import numpy as np
from Organoid2D import initializeOrganoid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from FVmesh import initializeFVmesh
from Parameters import setParameters
from Functions import coverPlot, saveOrg, saveAnim

Prm = setParameters()
Organoid = initializeOrganoid(Prm)
print('Organoid has', len(Organoid.IDs), 'cells')
print('Minimum NANOG expression level is', min(Organoid.NANOG))
print('Maximum NANOG expression level is', max(Organoid.NANOG))
print('Number of NANOG Cells =', len(Organoid.NANOG[Organoid.NANOG>Organoid.GATA6]))
print('Number of GATA6 Cells =', len(Organoid.GATA6[Organoid.GATA6>Organoid.NANOG]))

folder = 'Cell Fate - Organoid'
saveOrg(20, Organoid, Prm, folder)
#saveAnim(Organoid, folder)

#fig, ax = plt.subplots()
""" def update1(i):
    Pos = Organoid.Data[i][1]
    plt.cla()
    plt.scatter(Pos[:,0],Pos[:,1], s = 1000/len(Organoid.IDs), color = 'k')
    plt.xlim(min(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,0])*1.3)
    plt.ylim(min(Organoid.Pos[:,1])*1.3,max(Organoid.Pos[:,1])*1.3)
    plt.axis('off')
    return

def update2(i):
    Pos = Organoid.Data[i][1]
    FVmesh = initializeFVmesh(Pos)
    plt.cla()
    FVmesh.plot(size=1000/len(Organoid.IDs), bounds=[min(Organoid.NANOG),max(Organoid.NANOG)])
    plt.xlim(min(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,0])*1.3)
    plt.ylim(min(Organoid.Pos[:,1])*1.3,max(Organoid.Pos[:,1])*1.3)
    return

def update3(i):
    Pos = Organoid.Data[i][1]
    NANOG = Organoid.Data[i][3]
    FVmesh = initializeFVmesh(Pos)
    plt.cla()
    FVmesh.plot(NANOG, size=1000/len(Organoid.IDs), bounds=[min(Organoid.NANOG),max(Organoid.NANOG)])
    plt.xlim(min(Organoid.Pos[:,0])*1.3,max(Organoid.Pos[:,0])*1.3)
    plt.ylim(min(Organoid.Pos[:,1])*1.3,max(Organoid.Pos[:,1])*1.3)
    return

#ani1 = FuncAnimation(fig, update1, frames=steps, interval=1, blit=False)
#ani2 = FuncAnimation(fig, update2, frames=steps, interval=1, blit=False)
ani3 = FuncAnimation(fig, update3, frames=3000, interval=1, blit=False)
#plt.show()

#ani1.save('Results/scatter.mp4', fps=70, dpi=200)
#ani2.save('Results/voronoi.mp4', fps=70, dpi=200)
ani3.save('Results/transcription.mp4', fps=70, dpi=400) """
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
#saveAnim(Organoid, Prm, folder)
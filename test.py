import numpy as np
import matplotlib.pyplot as plt
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Parameters import setParameters

Prm = setParameters()
Organoid = initializeOrganoid(Prm)
FVmesh = initializeFVmesh(Organoid.Pos)
FVmesh.plot(Organoid.NANOG)
print(min(Organoid.NANOG), max(Organoid.NANOG))
plt.show()
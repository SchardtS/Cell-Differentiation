import numpy as np
import matplotlib.pyplot as plt
from FVmesh import initializeFVmesh
from Organoid2D import initializeOrganoid
from Parameters import setParameters
import pandas as pd

Prm = setParameters()
Organoid = initializeOrganoid(Prm, Transcription=False)
FVmesh = initializeFVmesh(Organoid.Pos)
FVmesh.plot()
plt.show()

df = pd.DataFrame(FVmesh.Pos)
df.to_csv('testOrganoid.csv', index=False)
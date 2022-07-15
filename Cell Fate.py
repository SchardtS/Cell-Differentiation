from Organoid import Organoid
from ExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np



# Instantiate organoid class
#org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
#org.evolution(T=200., dim=3, ignore=['transcription'])
#org.evolution(T=10, dim=3, ignore=['division', 'displacement'])
#print(org.divDist, org.nofCells)

# Plot the result
#plt.figure()
#org.cellPlot()
#plt.show()

# Save animation
#org.saveAnim(directory='Results', frames= 1000)

dat = ExpData('Data/includingSurfaceDistance/extendedRawDataICMOrganoids.csv')
dat.combinedPlot('48h', 1000, file=None)
plt.show()
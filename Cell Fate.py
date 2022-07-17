from Organoid import Organoid
#from ExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np



# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
org.evolution(dim=2)
print(org.nofCells)

# Plot the result
plt.figure(figsize=(10,10))
org.circularPlot(TF='GATA6', nofPlots=8, bounds=[0,0.1], radius='mean', size=0)
plt.show()

# Save animation
#org.saveAnim(directory='Results', frames= 1000)

#dat = ExpData('Data/includingSurfaceDistance/extendedRawDataICMOrganoids.csv')
#dat.combinedPlot('48h', 1000, file=None)
#plt.show()
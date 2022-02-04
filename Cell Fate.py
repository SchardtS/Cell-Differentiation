from Organoid2D import Organoid
from ExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np

# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
org.nofSteps = 3000
org.dt = org.T/(org.nofSteps - 1)
org.evolution(T=200, file = 'Organoid_mid.csv', mode='transcription')

# Plot the result
org.cellPlot(org.N)
org.timePlot()
plt.show()

#org.saveGIF(directory='Results', frames=1000, mode='NANOG')
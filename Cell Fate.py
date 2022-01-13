#from Organoid2D import Organoid
from Organoid2D import Organoid
from ExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np

# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
org.evolution(T=30)

# Plot the result
org.cellPlot(org.N)
plt.show()

org.saveGIF(directory='Results', frames=1000, mode='NANOG')
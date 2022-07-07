from Organoid import Organoid
from ClExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np



# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
org.evolution(T=234., dim=3, ignore=['transcription'])
print(org.divDist, org.nofCells)

# Plot the result
plt.figure()
org.cellPlot(radius='mean')
plt.show()

# Save animation
#org.saveAnim(directory='Results', frames= 1000)

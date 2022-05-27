from Organoid import Organoid
from ClExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np



# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
# org.evolution(T=10, file = 'Organoid_mid.csv', mode='transcription')
org.evolution(T=40, dim=2, ignore=['transcription'])

# Plot the result
plt.figure()
org.cellPlot(radius='mean')
plt.show()

# Save animation
#org.saveAnim(directory='Results', frames= 200)

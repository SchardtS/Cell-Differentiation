from Organoid2D import Organoid
from ClExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np

# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
org.evolution(T=10, file = 'Organoid_mid.csv', mode='transcription')

# Plot the result
plt.figure()
org.cellPlot(org.G, size=0)
#plt.savefig('x_to_y.pdf', transparent=True)
#plt.savefig('x_to_y.png', transparent=True)

#org.timePlot()
plt.show()
#nofN = len(org.N[org.N > org.G])
#nofG = len(org.G[org.G >= org.N])
#print(nofN / nofG, nofN, nofG)

#org.saveGIF(directory='Results', frames=1000, mode='NANOG')

#dat = ExpData('Data/extendedRawDataICMOrganoidsWFate.csv')
#dat.sliderPlot_HTML(file='24h_and_48h.html')
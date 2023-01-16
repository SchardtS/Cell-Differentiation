from Organoid import Organoid
#from ExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np



# Instantiate organoid class
org = Organoid()
#org.signal = 'dispersion'
#
## Run simulation for specified amount of time. If not specified its 24 hours
org.evolution(T=250,  dim=2, ignore=['transcription'])
print(org.nofCells)
#
## Plot the result
org.cellPlot()
plt.savefig('Results/Organoid.png', transparent=True)
#plt.show()
#
#print(min(org.N), max(org.N))
#print(min(org.G), max(org.G))
#plt.figure(figsize=(10,10))
#org.circularPlot(TF='GATA6', nofPlots=8, bounds=[0,0.1], radius='mean', size=0)
#plt.show()

# Save animation
org.saveAnim(directory='Results', frames= 1000)

#dat = ExpData('Data/includingSurfaceDistance/extendedRawDataICMOrganoids.csv')
#dat.combinedPlot('48h', 1000, file=None)
#plt.show()

#import matplotlib.pyplot as plt
#import numpy as np

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

#n = 100

#for i in range(100):
#    x = np.random.uniform(0,1)
#    y = np.random.uniform(0,1)
#    z = np.random.uniform(0,1)
#    ax.scatter(x, y, z, color='black')

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

#plt.show()
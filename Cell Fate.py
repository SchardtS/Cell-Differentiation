from Organoid2D import Organoid
#from Organoid3D import Organoid
from ExpData import ExpData
import matplotlib.pyplot as plt
import numpy as np

# Instantiate organoid class
#org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
#org.evolution(T=26)

# Plot the result
#org.cellPlot(org.N)
#plt.figure()
#org.cellPlot(org.N, radius='mean')
#plt.show()

""" org.moran()
print(org.Morans_I)
plt.figure()
org.pcf()
plt.savefig('Results/Cell Fate - Division/pcf.png', transparent = True) 
plt.savefig('Results/Cell Fate - Division/pcf.pdf', transparent = True) 
org.saveData(directory='Results/Cell Fate - Division/')
org.saveAnim(directory='Results/Cell Fate - Division/', frames=400, fps=30)


##### static geometry #####
N0 = org.r_N/org.gamma_N*3/4
G0 = org.r_N/org.gamma_N*3/4
org.u = np.append(np.random.normal(N0, N0*0.01, org.nofCells),
                  np.random.normal(G0, G0*0.01, org.nofCells))
org.N = org.u[:org.nofCells]
org.G = org.u[org.nofCells:]

org.evolution(T=26, mode = 'transcription')
org.saveData(directory='Results/Cell Fate - Static/')
org.moran()
print(org.Morans_I)
plt.figure()
org.pcf()
plt.savefig('Results/Cell Fate - Static/pcf.png', transparent = True)
plt.savefig('Results/Cell Fate - Static/pcf.pdf', transparent = True)  """

""" dat = ExpData('Data/includingSurfaceDistance/extendedRawDataICMOrganoids.csv')
dat.info(3)
dat.pcf_bounds(1, 10)
dat.moran_bounds(1, 10)
print(dat.moran[1])
plt.show() """

from ExpData import ExpData
import matplotlib.pyplot as plt

dat = ExpData('Data/includingSurfaceDistance/extendedRawDataICMOrganoids.csv')
for i in range(76):
    ID = i+1
    if ID < 10:
        num = '0'+str(ID)
    else:
        num = str(ID)
    if dat.stage[dat.id == ID][0] == '24h':
        dat.fullPlot_HTML(ID, 1000, file='Results/HTML PLots/24h/Organoid ID = ' + num + '.html')
    else:
        dat.fullPlot_HTML(ID, 1000, file='Results/HTML PLots/48h/Organoid ID = ' + num + '.html')
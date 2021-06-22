from Organoid2D_new import Organoid
import matplotlib.pyplot as plt

# Instantiate organoid class
org = Organoid()

# Run simulation for specified amount of time. If not specified its 24 hours
org.evolution(T=20, file = 'Organoid_mid.csv', mode='transcription')

# Plot the result
org.cellPlot(org.N)
plt.show()

#org.saveData(directory='Results/Cell Fate/')
#org.saveAnim(directory='Results/Cell Fate/', frames=200, fps=30)
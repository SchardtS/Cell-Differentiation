import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 1000
x = np.linspace(0,1,N)
t = np.linspace(0,1,N)

a = 1
u0 = np.array([0]*len(x))
u0[x > 0.1] = 1
u0[x > 0.3] = 0

dx = x[1]-x[0]
dt = t[1]-t[0]
CFL = a*dt/dx
print(CFL)

ufull = np.empty([len(x),len(t)])
ufull[:,0] = u0
for i in range(1,len(t)):
    uL = np.roll(u0,1)
    uR = np.roll(u0,-1)

    u = u0 - dt/dx/2*a*(uR-uL) + dt**2/dx**2/2*a**2*(uR-2*u0+uL)
    #u = u0 - dt/dx*a*(u0-uL)
    u0 = u
    ufull[:,i] = u

""" Dxx = np.zeros([len(x),len(x)])
for i in range(len(x)):
    for j in range(len(x)):
        Dxx[i,i] = -2
        Dxx[i,i-1] = 1
        if i+1 > len(x)-1:
            Dxx[i,0] = 1
        else:
            Dxx[i,i+1] = 1
        
E = np.eye(len(x))
MAT = E-a*dt/dx**2*Dxx
for i in range(1,len(t)):
    u = np.linalg.solve(MAT, u0)
    u0 = u
    ufull[:,i] = u """

fig, ax = plt.subplots()
def update(i):
    plt.cla()
    plt.rc('font', size=14)
    plt.plot(x,ufull[:,i], linewidth=2)
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.ylim(-0.05,1.05)

ani = FuncAnimation(fig, update, frames=N, interval=1, blit=False)
plt.show()
#ani.save('Results/diffusion.mp4', fps=70, dpi=200)
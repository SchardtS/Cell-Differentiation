import numpy as np
import matplotlib.pyplot as plt

eta_a = np.exp(5)
eta_b = np.exp(6)

b = 0.02
a = np.linspace(0,0.1,1000)
p_A = lambda delta: eta_a*a*(1 + eta_b*delta*b)/(1 + eta_a*a*(1 + eta_b*delta*b) + eta_b*b)

plt.figure()
plt.rc('font', size=14)
plt.plot(a, p_A(0), 'c--', label = '$\eta_{ab} = 0$', lw=3)
plt.plot(a, p_A(1/2), 'c', label = '$\eta_{ab} = 1/2$', lw=3)
plt.plot(a, p_A(1), 'k', label = '$\eta_{ab} = 1$', lw=3)
plt.plot(a, p_A(2), 'm', label = '$\eta_{ab} = 2$', lw=3)
plt.legend()
plt.xlabel('$a$')
plt.ylabel('$p_A(a)$')
plt.savefig('Results/probability_interaction.png')
plt.savefig('Results/probability_interaction.pdf')
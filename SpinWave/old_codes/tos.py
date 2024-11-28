import numpy as np
import matplotlib.pyplot as plt

kx = np.linspace(0,0.1,100,endpoint=False)
X,Y = np.meshgrid(kx,kx)

U = 1/2*np.arctanh((np.cos(X)+np.cos(Y))/(4-np.cos(X)-np.cos(Y)))

plt.figure()
s = plt.contourf(X,Y,U)
plt.colorbar(s)
plt.show()

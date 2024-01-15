import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

chi = 0.2096
steps = 1000
all_Sx = []
list_N = [10,16,18,20,26,32]
for N in list_N:
    A_M = np.zeros(N+1)
    for M in range(-N//2,N//2+1):
        A_M[N//2+M] = np.sqrt(binom(N,N//2+M)/2**N)

    list_t = np.linspace(0,100,steps)
    S_x = np.zeros(steps,dtype=complex)
    for i in range(steps):
        S_x[i] = 0
        for M in range(-N//2,N//2+1):
            if not M==-N//2:
                S_x[i] += 1/2*np.conjugate(A_M[N//2+M])*A_M[N//2+M-1]*np.exp(-1j*list_t[i]/2/chi/N*(2*M-1))
            if not M==N//2:
                S_x[i] += 1/2*np.conjugate(A_M[N//2+M])*A_M[N//2+M+1]*np.exp(1j*list_t[i]/2/chi/N*(2*M+1))
    all_Sx.append(S_x)
#Plot
plt.figure()
for i in range(len(list_N)):
    plt.plot(list_t,-np.real(all_Sx[i])/2,label=str(list_N[i]))
plt.legend()
plt.xlabel(r'$t/J$',size=15)
plt.ylabel(r'$\langle S_x\rangle$',size=15)
plt.show()

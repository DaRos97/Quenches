import numpy as np
from scipy.special import binom

tau = 5     #>=1
N = 16       #Even
print("N: ",N," and tau: ",tau)

#Compute norm M=0 (GS)
N_0 = tau*2**(N-2)
if tau > 1:
    for k in range(N//2+1):
        for d in range(1,tau):
            temp = binom(N,2*k)*(tau-d)*np.cos(k*d/tau*2*np.pi)
            N_0 += temp
Norm_0 = np.sqrt(N_0)/2**((N-3)/2)
print(Norm_0)

#Compute amplitude 0
A_0 = 0+0*1j
for t in range(tau):
    for k in range(N//2):
        A_0 += binom(N,2*k)*np.exp(-1j*t/tau*k*2*np.pi)
A_0 /= Norm_0
A_0 /= 2**N
print("A_0: ",A_0)

#Compute norm M=1
N_1 = 0
for t in range(tau):
    for tp in range(tau):
        temp = np.exp(1j*(t-tp)/tau*np.pi)
        p1 = 0
        for k in range((N-2)//2):
            p1 += binom(N-2,2*k+1)*temp**(2*k+1)
        p1 *= (N-1)*temp
        p2 = 0
        for k in range((N-2)//2+1):
            p2 += binom(N-1,2*k)*temp**(2*k)
        N_1 += p1+p2
Norm_1 = np.sqrt(N_1)/np.sqrt(N)/2**((N-2)/2)/Norm_0
print(Norm_1)
if abs(np.imag(Norm_1))>1e-10:
    print("Error in Norm")
    exit()

#Compute amplitude 1
A_1 = 0+0*1j
for t in range(tau):
    for k in range((N-2)//2+1):
        temp = np.exp(-1j*t/tau*np.pi*2*k)
        A_1 += binom(N-1,2*k)*temp
A_1 /= Norm_0*Norm_1*2**(N-1)
print("A_1: ",A_1)

#Compute norm M=-1
N_m1 = 0
for t in range(tau):
    for tp in range(tau):
        temp = np.exp(1j*(t-tp)/tau*np.pi)
        p1 = 0
        for k in range((N-2)//2):
            p1 += binom(N-2,2*k+1)*temp**(2*k+1)
        p1 *= (N-1)
        p2 = 0
        for k in range((N-2)//2+1):
            p2 += binom(N-1,2*k+1)*temp**(2*k+1)
        N_m1 += (p1+p2)*temp
Norm_m1 = np.sqrt(N_m1)/np.sqrt(N)/2**((N-2)/2)/Norm_0
print(Norm_m1)
if abs(np.imag(Norm_m1))>1e-10:
    print("Error in Norm")
    exit()

#Compute amplitude -1
A_m1 = 0+0*1j
for t in range(tau):
    temp = np.exp(-1j*t/tau*np.pi)
    for k in range((N-2)//2+1):
        A_m1 += binom(N-1,2*k+1)*temp**(2*k+2)
A_m1 /= Norm_m1*Norm_0*2**(N-1)
print("A_m1: ",A_m1)

#Compute norm M=2
N_2 = 0
for t in range(tau):
    for tp in range(tau):
        gamma = np.exp(1j*(t-tp)/tau*np.pi)
        p1 = 0
        for k in range((N-4)//2+1):
            p1 += binom(N-4,2*k)*gamma**(2*k)
        p1 *= (2+(N-2)*(N-3)*gamma**2)
        p2 = 0
        for k in range((N-4)//2):
            p2 += binom(N-4,2*k+1)*gamma**(2*k+1)
        p2 *= 4*(N-2)*gamma
        N_2 += p1+p2
Norm_2 = np.sqrt(N_2)*np.sqrt((N-1)/N**3/2**(N-2))/Norm_1/Norm_0
print(Norm_2)
if abs(np.imag(Norm_2))>1e-10:
    print("Error in Norm")
    exit()

#Compute amplitude 2
A_2 = 0+0*1j
for t in range(tau):
    for k in range((N-2)//2+1):
        temp = np.exp(-1j*t/tau*np.pi*2*k)
        A_2 += binom(N-2,2*k)*temp
A_2 *= (N-1)/N/2**(N-1)/Norm_2/Norm_1/Norm_0
print("A_2: ",A_2)

#
A_M = [A_0,A_1,A_2]
N_M = [N_0,N_1,N_2]

chi = 0.2096
steps = 100
list_t = np.linspace(0,70,steps)
S_x = np.zeros(steps,dtype=complex)
for i in range(steps):
    S_x[i] = 0
    for M in range(-2,3):
        if not M==-2:
            S_x[i] += 1/2*np.conjugate(A_M[abs(M)])*A_M[abs(M-1)]*np.exp(-1j*list_t[i]/2/chi/N*(2*M-1))
        if not M==2:
            S_x[i] += 1/2*np.conjugate(A_M[abs(M)])*A_M[abs(M+1)]*np.exp(1j*list_t[i]/2/chi/N*(2*M+1))

#Plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(list_t,np.real(S_x))
plt.show()














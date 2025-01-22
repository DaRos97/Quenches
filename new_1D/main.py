import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.linalg import expm
import scipy

plot_fidelity = 0
plot_populations = 1

g_ = 10 #MHz
h_ = 15 #MHz
full_time_ramp = 0.5 #ramp time in ms
full_time_measure = 0.8     #measure time in ms
time_step = 0.001    #time step of ramp

N = 42  #sites
Nt = 2000 #time steps after ramp

stop_ratio_list = np.linspace(0.1,1,10)
time_list = np.linspace(0,full_time_measure,Nt)

def compute_H(g_nn,h_field,N_sites):
    ham = np.zeros((N_sites,N_sites))
    for i in range(N_sites-1):
        ham[i,i+1] = g_nn
        ham[i+1,i] = g_nn
        ham[i,i] = h_field*(-1)**(i+1)
    ham[-1,-1] = h_field*(-1)**(N_sites)
    #PBC
    ham[-1,0] = -g_nn
    ham[0,-1] = -g_nn
    return ham

stop_ratio = 1
ramp_time = full_time_ramp
time_steps = int(ramp_time//time_step)   #steps in time evolution
g = np.zeros(time_steps)
h = np.zeros(time_steps)
for i in range(time_steps):
    g[i] = 0 + g_*stop_ratio/(time_steps-1)*i
    h[i] = h_ - h_*stop_ratio/(time_steps-1)*i

time_evolved_psi_fn = "data/time_evolved_psi_"+str(N)+'_'+"{:.2f}".format(full_time_ramp)+'.npy'
fidelity_fn = "data/fidelity_"+str(N)+'_'+"{:.2f}".format(full_time_ramp)+'.npy'
if not Path(time_evolved_psi_fn).is_file(): #Time evolution and fidelity along the ramp
    psi = np.zeros((time_steps,N,N),dtype=complex)    #wavefunction at all time steps of the ramp
    fidelity = np.zeros(time_steps)
    for i in range(time_steps):
        if i==0:
            psi[i] = np.linalg.eigh(compute_H(g[i],h[i],N))[1]
        else:
            exp_H = expm(-1j*2*np.pi*compute_H(g[i],h[i],N)*time_step)
            for m in range(N):  #evolve each mode independently
                psi[i,:,m] = exp_H @ psi[i-1,:,m]
        #Fidelity wrt real GS
        E_GS,psi_GS = scipy.linalg.eigh(compute_H(g[i],h[i],N))
        fidelity[i] = np.absolute(np.linalg.det(psi[i,:,:N//2].T.conj()@psi_GS[:,:N//2]))**2
    #Save
    np.save(time_evolved_psi_fn,psi)
    np.save(fidelity_fn,fidelity)
else:
    time_evolved_psi = np.load(time_evolved_psi_fn)
    fidelity = np.load(fidelity_fn)

if plot_fidelity:   #Plot fidelity along the ramp and quenched values of h and g
    fig = plt.figure()
    ax = fig.add_subplot()
    l1 = ax.plot(np.linspace(0,ramp_time,time_steps),fidelity,color='r',label="fidelity")
    ax.set_ylabel("Fidelity wrt GS")
    ax.set_xlabel("time (ms)")
    ax_r = ax.twinx()
    l2 = ax_r.plot(np.linspace(0,ramp_time,time_steps),g,color='g',label="g")
    l3 = ax_r.plot(np.linspace(0,ramp_time,time_steps),h,color='b',label="h")
    ax_r.set_ylabel("coupling g and field h (MHz)")
    ax.set_title("Full ramp fidelity")
    ls = l1+l2+l3
    labels = [l.get_label() for l in ls]
    ax.legend(ls,labels)
    plt.savefig("figures/ramp_fidelity_"+str(N)+'_'+"{:.2f}".format(full_time_ramp)+'.png')
    plt.show()
    exit()
#
if plot_populations:
    pop = np.zeros((time_steps,N))
    H = compute_H(g[-1],h[-1],N)
    evals,evecs = np.linalg.eigh(H)
    for i_t in range(time_steps):
        for k in range(N):
            pop[i,k] = np.sum(time_evolved_psi[i_t,:,k]@time_evolved_psi[i_t].conj())
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(N//2):
        ax.plot(np.linspace(0,ramp_time,time_steps),pop[:,i],color='r')
    plt.show()
    exit()

fig = plt.figure(figsize=(20, 15))
for i_sr in range(len(stop_ratio_list)):
    stop_ratio = stop_ratio_list[i_sr]
    g = g_*stop_ratio
    h = h_*(1-stop_ratio)
    H = compute_H(g,h,N)
    evals,evecs = np.linalg.eigh(H)
    #Time evolution
    U_1 = [np.exp(1j*2*np.pi*evals[:N//2]*time) for time in time_list]
    U_2 = [np.exp(-1j*2*np.pi*evals[N//2:]*time) for time in time_list]
    #
    rho1 = np.array([np.einsum('ik,kj,k->ij',evecs[:,:N//2],evecs[:,:N//2].T.conj(),U_1[i_t],optimize=True) for i_t in range(Nt)])
    rho2 = np.array([np.einsum('ik,kj,k->ij',evecs[:,N//2:],evecs[:,N//2:].T.conj(),U_2[i_t],optimize=True) for i_t in range(Nt)])
    #Correlator in space-time
    ZZ = np.zeros((N,Nt),dtype=complex)
    for i in range(N):
        ZZ[i] = np.array([2*1j*np.imag(rho1[i_t,i,0]*rho2[i_t,i,0]) for i_t in range(Nt)])
    #Fourier transform -> momentum-frequency
    ZZ_ft = np.fft.fftshift(np.fft.fft2(ZZ)) / np.sqrt(N*Nt)
    kx = np.fft.fftshift(np.fft.fftfreq(N, d=1)) * np.pi
    omega = np.fft.fftshift(np.fft.fftfreq(Nt, d=full_time_measure / Nt))
    #Plot
    ax = fig.add_subplot(2,5,i_sr+1)
    pm = ax.pcolormesh(kx, omega, np.abs(ZZ_ft).T, shading='auto', cmap='magma')
    if i_sr in [4,9]:
        label_cm = 'Magnitude of Fourier Transform'
    else:
        label_cm = ''
    plt.colorbar(pm,label=label_cm)
    ax.set_ylim(-50,50)
    if i_sr>4:
        ax.set_xlabel('Momentum ($k_x$)')
    if i_sr in [0,5]:
        ax.set_ylabel('Frequency $\omega$ (MHz)')
    ax.set_title('Stop ratio '+"{:.1f}".format(stop_ratio))
#    plt.savefig("figures/1D_GS_100_"+"{:.1f}".format(stop_ratio)+".png")
fig.tight_layout()
plt.show()
#    exit()




























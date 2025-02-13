import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
from tqdm import tqdm
from scipy.linalg import expm
import scipy
import sys

plot_fidelity = 0
plot_populations = 0
plot_correlators = 1
savefig_corr = False

use_time_evolved = 1
add_time_ev = 0

save_time_evolved = True
save_data = False

g_ = 10 #MHz
h_ = 15 #MHz
full_time_ramp = 0.5 if len(sys.argv)<2 else int(sys.argv[1])/1000#ramp time in ms
full_time_measure = 0.8     #measure time in ms
time_step = full_time_ramp/500
#time_step = 0.001    #time step of ramp

N = 42  #sites
Nt = 401#2000 #time steps after ramp
Nomega = 2000
omega = np.linspace(-250,250,Nomega)

print(N,full_time_ramp,time_step)

stop_ratio_list = np.linspace(0.1,1,10)
time_list = np.linspace(0,full_time_measure,Nt)

def compute_H(g_nn,h_field,N_sites):
    """NxN fermion Hamiltonian with PBC"""
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

def compute_populations(psi_t,psi_GS):
    """Compute the populations of the single modes"""
    N = psi_t.shape[1]
    res = np.zeros(N)
    for k in range(N):
        res[k] = np.sum((np.absolute(psi_t.T.conj() @ psi_GS[:,k])**2)[:N//2])
    return res

def compute_populations2(psi_t,psi_GS):
    """Compute the populations of the single modes, v2"""
    alpha = psi_t[:,:N//2]
    beta = psi_GS
    result = np.real(np.diagonal(beta.T.conj()@alpha@alpha.T.conj()@beta))
    return result

stop_ratio = 1
ramp_time = full_time_ramp
time_steps = int(ramp_time//time_step)   #steps in time evolution
g = np.zeros(time_steps)
h = np.zeros(time_steps)
for i in range(time_steps):
    g[i] = 0 + g_*stop_ratio/(time_steps-1)*i
    h[i] = h_ - h_*stop_ratio/(time_steps-1)*i

time_evolved_psi_fn = "data/time_evolved_psi_"+str(N)+'_'+"{:.3f}".format(full_time_ramp)+'_'+"{:.5f}".format(time_step)+'.npy'
fidelity_fn = "data/fidelity_"+str(N)+'_'+"{:.3f}".format(full_time_ramp)+'_'+"{:.5f}".format(time_step)+'.npy'
pop_fn = "data/populations_"+str(N)+'_'+"{:.3f}".format(full_time_ramp)+'_'+"{:.5f}".format(time_step)+'.npy'
if not Path(time_evolved_psi_fn).is_file(): #Time evolution and fidelity along the ramp
    time_evolved_psi = np.zeros((time_steps,N,N),dtype=complex)    #wavefunction at all time steps of the ramp
    fidelity = np.zeros(time_steps)
    pop = np.zeros((time_steps,N))
    print("Time evolution")
    for it in tqdm(range(time_steps)):
        if it==0:
            time_evolved_psi[it] = np.linalg.eigh(compute_H(g[it],h[it],N))[1]      #exact GS at t=0
        else:
            exp_H = expm(-1j*2*np.pi*compute_H(g[it],h[it],N)*time_step)
            for m in range(N):  #evolve each mode independently
                time_evolved_psi[it,:,m] = exp_H @ time_evolved_psi[it-1,:,m]
        #Fidelity wrt real GS
        E_GS,psi_GS = scipy.linalg.eigh(compute_H(g[it],h[it],N))
        fidelity[it] = np.absolute(np.linalg.det(time_evolved_psi[it,:,:N//2].T.conj()@psi_GS[:,:N//2]))**2
        #Occupation of populations
        pop[it] = compute_populations2(time_evolved_psi[it],psi_GS)
    if save_time_evolved:
        np.save(time_evolved_psi_fn,time_evolved_psi)
        np.save(fidelity_fn,fidelity)
        np.save(pop_fn,pop)
else:
    time_evolved_psi = np.load(time_evolved_psi_fn)
    fidelity = np.load(fidelity_fn)
    pop = np.load(pop_fn)

def time_evolve(alpha,beta,e,t):
    result = np.einsum('iq,ir,r,jr->jq',alpha,beta.conj(),np.exp(-1j*2*np.pi*t*e),beta)
    return result

def get_Gij(alpha,beta,U):
    """Compute correlator <c_i(t)^\dag c_j(0)>"""
    return np.array([(alpha@alpha.T.conj()@beta@np.diag(U_t[i_t])@beta.T.conj()).T for i_t in range(len(U))])

def get_Hij(alpha,beta,U):
    """Compute correlator <c_i(t) c_j(0)^\dag>"""
    return np.array([N/2*beta@np.diag(U_t[i_t].conj())@beta.T.conj() - beta@np.diag(U_t[i_t].conj())@beta.T.conj()@alpha@alpha.T.conj() for i_t in range(len(U))])

corr_fn = "data/correlators_"+str(N)+'_'+"{:.3f}".format(full_time_ramp)+'_'+"{:.5f}".format(time_step)+'_'+"{:.3f}".format(full_time_measure)+'.npy'
if not Path(corr_fn).is_file():
    txt_wf = 'time evolved wavefunction' if use_time_evolved else 'ground state wavefunction'
    print("Computing correlation function with "+txt_wf+': ')
    ZZ_fts = np.zeros((len(stop_ratio_list),N,Nomega),dtype=complex)
    for i_sr in tqdm(range(len(stop_ratio_list))):
        stop_ratio = stop_ratio_list[i_sr]
        i_t = int(time_steps*stop_ratio)-1
        E_GS,beta = np.linalg.eigh( compute_H(g[i_t],h[i_t],N) )
#        _,beta = np.linalg.eigh(compute_H(g[-1],h[-1],N))
        if use_time_evolved and 1:
            alpha = np.copy(time_evolved_psi[i_t][:,:N//2])
            if add_time_ev:
                extra_time = full_time_ramp*(1-stop_ratio)  #in ms
                alpha = time_evolve(alpha,beta,E_GS,extra_time)
        else:
            alpha = np.copy(beta[:,:N//2])
        #Time evolution measurement
        U_t = [np.exp(-1j*2*np.pi*E_GS[:]*time) for time in time_list]
        #
        new = True
        if new:
            G_ij = get_Gij(alpha,beta,U_t)
            H_ij = get_Hij(alpha,beta,U_t)
            #rho1 = np.array([(alpha@alpha.T.conj()@beta@np.diag(U_t[i_t])@beta.T.conj())[0,:] for i_t in range(Nt)])
            #rho1c = np.array([alpha[0]@alpha.T.conj()@beta@np.diag(U_t[i_t])@beta.T.conj() for i_t in range(Nt)])
            #rho1b = np.array([beta[0]@beta.T.conj()@alpha@alpha.T.conj()@beta@np.diag(U_2[i_t])@beta.T.conj() for i_t in range(Nt)])
            #rho1a = np.array([np.einsum('ab,ac,ic,d,eb,ed,c->i',alpha.conj(),beta,beta.conj(),beta[0],alpha,beta.conj(),U_2[i_t],optimize=True) for i_t in range(Nt)])
            #rho2 = np.array([N/2*(beta@np.diag(U_t[i_t].conj())@beta.T.conj())[:,0] - (beta@np.diag(U_t[i_t].conj())@beta.T.conj()@alpha@alpha.T.conj())[:,0] for i_t in range(Nt)])
            #rho2b = np.array([N/2*beta@np.diag(U_1[i_t])@beta[0].T.conj() - beta@np.diag(U_1[i_t])@beta.T.conj()@alpha@alpha.T.conj()@beta@beta[0].T.conj() for i_t in range(Nt)])
            #rho2a = np.array([np.einsum('ab,ac,id,d,ec,eb,d->i',alpha.conj(),beta,beta,beta[0].conj(),beta.conj(),alpha,U_1[i_t],optimize=True)
            #                 -np.einsum('ab,ac,id,c,ed,eb,d->i',alpha.conj(),beta,beta,beta[0].conj(),beta.conj(),alpha,U_1[i_t],optimize=True) for i_t in range(Nt)])
        else:
            #With populations
            #rho1 = np.array([np.einsum('ik,kj,k,k->ij',evecs[:,:],evecs[:,:].T.conj(),U_1[i_t],pop_k,optimize=True) for i_t in range(Nt)])
            #rho2 = np.array([np.einsum('ik,kj,k,k->ij',evecs[:,:],evecs[:,:].T.conj(),U_2[i_t],1-pop_k,optimize=True) for i_t in range(Nt)])
            evecs = beta
            rho1 = np.array([evecs[:,:N//2]@np.diag(U_1[i_t][:N//2])@evecs[:,:N//2].T.conj() for i_t in range(Nt)])
            rho2 = np.array([evecs[:,N//2:]@np.diag(U_2[i_t][N//2:])@evecs[:,N//2:].T.conj() for i_t in range(Nt)])
        #
        #Correlator in space-time
        ZZ = np.zeros((N,Nt),dtype=complex)
        for i in range(N):
            if new:
                ZZ[i] = np.array([2*1j*np.imag(G_ij[i_t,i,0]*H_ij[i_t,i,0]) for i_t in range(Nt)])/N*2
            else:
                ZZ[i] = np.array([2*1j*np.imag(rho1[i_t,i,0]*rho2[i_t,i,0]) for i_t in range(Nt)])
        ZZ_fts[i_sr] = np.fft.fftshift(np.fft.fft2(ZZ,[N,Nomega]))
    if save_data:
        np.save(corr_fn,ZZ_fts)
else:
    ZZ_fts = np.load(corr_fn)

if plot_correlators:
    #Plot
    fig = plt.figure(figsize=(17, 8))
    txt_title = 'time evolved wavefunction' if use_time_evolved else 'ground state wavefunction'
    plt.suptitle("Total ramp time: "+str(int(full_time_ramp*1000))+" ns, "+txt_title)
    kx = np.fft.fftshift(np.fft.fftfreq(N,d=1))
    for i_sr in range(len(stop_ratio_list)):
        stop_ratio = stop_ratio_list[i_sr]
        #Plot
        ax = fig.add_subplot(2,5,i_sr+1)
    #    ax = fig.add_subplot()
        pm = ax.pcolormesh(kx, omega, np.abs(ZZ_fts[i_sr]).T, shading='auto', cmap='magma')
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
        if 0: #plot fit
            i_t = int(time_steps*stop_ratio)-1
            en_m = np.sqrt(h[i_t]**2+4*g[i_t]**2*np.cos(kx*2*np.pi+np.pi/2)**2)+abs(h[i_t])
            en_M = 2*np.sqrt(h[i_t]**2+4*g[i_t]**2*np.cos(kx*np.pi+np.pi/2)**2)
            en_ex = en_m-2*abs(h[i_t])
            ax.plot(kx,en_m,color='r')
            ax.plot(kx,-en_m,color='r')
            ax.plot(kx,en_M,color='g')
            ax.plot(kx,-en_M,color='g')
            ax.plot(kx,en_ex,color='b')
    #    plt.show()

    fig.tight_layout()
    txt_wf = 'ramp_WF' if use_time_evolved else 'GS_WF'
    if savefig_corr:
        plt.savefig("figures/"+corr_fn[5:-4]+'.png')
    plt.show()
    exit()


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
#    plt.savefig("figures/ramp_fidelity_"+str(N)+'_'+"{:.3f}".format(full_time_ramp)+'.png')

if plot_populations:
    E_GS,psi_GS = scipy.linalg.eigh(compute_H(g[-1],h[1],N))
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0,1,N))#N//2-1))
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(N//2,N):#N//2-1):
        ax.plot(np.linspace(0,ramp_time,time_steps),pop[:,i],color=colors[i-N//2])
    ax.set_ylabel('Modes occupations')
    ax.set_xlabel('time (ms)')
    norm = Normalize(vmin=E_GS[N//2], vmax=E_GS[N-1])
    sm = ScalarMappable(cmap=cmap, norm=norm)  # Create a ScalarMappable for the colorbar
    sm.set_array([])  # The ScalarMappable needs an array, even if unused
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Energy of modes")
    plt.savefig("figures/"+pop_fn[5:-4]+'.png')

if plot_fidelity or plot_populations or plot_correlators:
    plt.show()

























